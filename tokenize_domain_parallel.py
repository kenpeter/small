#!/usr/bin/env python3
"""Generic parallel domain tokenizer — classifies into easy/medium/hard tiers and writes .bin shards.

Usage: python3 tokenize_domain_parallel.py --domains math,code,synth,web --workers 4
"""
import sys, os, gc, time, re, json, multiprocessing, importlib.util
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq

os.environ["TOKENIZERS_PARALLELISM"] = "false"

ROOT = Path("/home/kenpeter/work/data")
SEQ_LEN = 2048
FLUSH_TOKENS = 100_000_000

# Domain configs (column: parquet field holding the document text)
DOMAIN_CONFIGS = {
    "math": {
        "sources": [ROOT / "_raw_original" / "finemath-3plus", ROOT / "_raw_original" / "open-web-math"],
        "column": "text",
    },
    "code": {
        "sources": [ROOT / "_raw_original" / "github-code"],
        "column": "content",
    },
    "synth": {
        "sources": [ROOT / "_raw_original" / "cosmopedia"],
        "column": "text",
    },
    "web": {
        "sources": [ROOT / "_raw_original" / "fineweb-edu"],
        "column": "text",
    },
}

# Import classify_document once at module level
spec = importlib.util.spec_from_file_location("filter_all", ROOT / "filter_all.py")
filter_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(filter_mod)
classify_document = filter_mod.classify_document

def log(msg, log_file=None):
    ts = time.strftime('%H:%M:%S')
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    if log_file:
        with open(log_file, "a") as f:
            f.write(line + "\n")

def worker_func(worker_id, domain, column, file_paths, out_base, log_file):
    """Process files for a domain, write .bin shards."""
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")
    log(f"[W{worker_id}/{domain}] Tokenizer loaded, {len(file_paths)} files", log_file)

    out_dirs = {}
    for t in ["easy", "medium", "hard"]:
        d = out_base / f"_shards_{domain}_{t}" / f"worker_{worker_id}"
        d.mkdir(parents=True, exist_ok=True)
        out_dirs[t] = d

    def next_shard(tier):
        existing = sorted(out_dirs[tier].glob("*.bin"))
        if existing:
            nums = [int(re.search(r'train-(\d+)\.bin', p.name).group(1)) for p in existing if re.search(r'train-(\d+)\.bin', p.name)]
            return max(nums) + 1 if nums else 0
        return 0

    shard_counters = {t: next_shard(t) for t in ["easy", "medium", "hard"]}
    tier_buffers = {t: [] for t in ["easy", "medium", "hard"]}
    tier_totals = {t: 0 for t in ["easy", "medium", "hard"]}
    stats = {"total": 0, "kept": 0, "quality_sum": 0.0}
    wp = f"W{worker_id}"

    def flush_tier(tier):
        buf = tier_buffers[tier]
        if not buf:
            return
        flat = np.concatenate(buf)
        flat = flat[: (len(flat) // SEQ_LEN) * SEQ_LEN]
        if len(flat) > 0:
            out_path = out_dirs[tier] / f"train-{shard_counters[tier]:06d}.bin"
            flat.tofile(str(out_path))
            shard_counters[tier] += 1
            log(f"  [{wp}/{domain}/{tier}] Shard: {len(flat):,} tokens -> {out_path.name}", log_file)
        tier_buffers[tier] = []
        del flat
        gc.collect()

    def append_tokens(tier, toks):
        if not toks:
            return
        arr = np.array(toks, dtype=np.uint16)
        tier_buffers[tier].append(arr)
        tier_totals[tier] += len(arr)
        total = sum(len(a) for a in tier_buffers[tier])
        if total >= FLUSH_TOKENS:
            flush_tier(tier)

    for pf_path in file_paths:
        fname = pf_path.name
        try:
            pf = pq.ParquetFile(str(pf_path))
            n_rows = pf.metadata.num_rows
        except Exception as e:
            log(f"[{wp}/{domain}] ❌ Skipping {fname}: {e}", log_file)
            continue

        t0 = time.time()
        log(f"[{wp}/{domain}] Processing {fname} ({n_rows} rows)...", log_file)
        file_kept = 0
        keep_count = 0

        for batch_idx, batch in enumerate(pf.iter_batches(batch_size=1000, columns=[column])):
            for row in batch.to_pylist():
                stats["total"] += 1
                text = row.get(column, "")
                if not isinstance(text, str):
                    text = str(text) if text is not None else ""

                ok, reason, q_score, tier = classify_document(text, domain)
                if not ok:
                    continue

                stats["kept"] += 1
                file_kept += 1
                stats["quality_sum"] += q_score

                toks = tokenizer.encode(text, add_special_tokens=False)
                append_tokens(tier, toks)

            keep_count += 1
            if keep_count % 5 == 0:
                elapsed = time.time() - t0
                pct = (batch_idx * 1000) / n_rows * 100
                mem = [f"{t}:{sum(len(a) for a in tier_buffers[t])//1000000}M" for t in ["easy","medium","hard"]]
                log(f"[{wp}/{domain}] {batch_idx*1000}/{n_rows} rows ({pct:.0f}%) | kept={file_kept} | "
                    f"totals: easy={tier_totals['easy']//1000000}M med={tier_totals['medium']//1000000}M "
                    f"hard={tier_totals['hard']//1000000}M | buf=[{', '.join(mem)}] | {elapsed:.0f}s", log_file)

        elapsed = time.time() - t0
        log(f"[{wp}/{domain}] Done {fname}: {file_kept} kept in {elapsed:.1f}s", log_file)

    for t in ["easy", "medium", "hard"]:
        flush_tier(t)

    log(f"[{wp}/{domain}] FINAL: kept={stats['kept']} | tokens: easy={tier_totals['easy']:,} "
        f"med={tier_totals['medium']:,} hard={tier_totals['hard']:,}", log_file)

def consolidate_shards(domain, log_file):
    """Merge worker shards into consolidated train-*.bin files."""
    import shutil
    for tier in ["easy", "medium", "hard"]:
        tier_dir = ROOT / f"_shards_{domain}_{tier}"
        worker_dirs = sorted(tier_dir.glob("worker_*"))
        if not worker_dirs:
            continue

        all_shards = []
        for wd in worker_dirs:
            all_shards.extend(sorted(wd.glob("*.bin")))

        def sort_key(p):
            parts = p.stem.split("-")
            return int(parts[1])
        all_shards.sort(key=sort_key)

        log(f"[{domain}/consolidate] {tier}: merging {len(all_shards)} shards from {len(worker_dirs)} workers", log_file)

        next_idx = 0
        existing = sorted(tier_dir.glob("train-*.bin"))
        if existing:
            nums = [int(re.search(r'train-(\d+)\.bin', p.name).group(1)) for p in existing if re.search(r'train-(\d+)\.bin', p.name)]
            if nums:
                next_idx = max(nums) + 1

        for shard_path in all_shards:
            new_name = f"train-{next_idx:06d}.bin"
            dst = tier_dir / new_name
            shutil.move(str(shard_path), str(dst))
            next_idx += 1

        for wd in worker_dirs:
            remaining = list(wd.glob("*"))
            if not remaining:
                wd.rmdir()

        final_shards = sorted(tier_dir.glob("train-*.bin"))
        total_tokens = sum(len(np.fromfile(str(s), dtype=np.uint16)) for s in final_shards)
        log(f"[{domain}/consolidate] {tier}: {len(final_shards)} shards, {total_tokens:,} tokens", log_file)

def run_domain(domain, num_workers):
    """Run parallel tokenization for one domain."""
    config = DOMAIN_CONFIGS[domain]
    log_file = ROOT / f"tokenize_{domain}_parallel.log"
    log_file.unlink(missing_ok=True)

    # Collect all parquet files from source dirs
    parquet_files = []
    for src in config["sources"]:
        parquet_files.extend(sorted(src.glob("*.parquet")))
    
    log(f"[{domain}] Total {len(parquet_files)} parquet files, {num_workers} workers", log_file)

    # Split files among workers (round-robin)
    chunks = [[] for _ in range(num_workers)]
    for i, pf in enumerate(parquet_files):
        chunks[i % num_workers].append(pf)

    for wi, chunk in enumerate(chunks):
        log(f"[{domain}] Worker {wi}: {len(chunk)} files", log_file)

    t_start = time.time()
    # Clear existing worker dirs
    out_base = ROOT
    for t in ["easy", "medium", "hard"]:
        d = out_base / f"_shards_{domain}_{t}"
        for wd in sorted(d.glob("worker_*")):
            for f in wd.glob("*.bin"):
                f.unlink()

    with multiprocessing.Pool(num_workers) as pool:
        pool.starmap(worker_func, [
            (wi, domain, config["column"], chunk, out_base, log_file)
            for wi, chunk in enumerate(chunks)
        ])

    elapsed = time.time() - t_start
    log(f"[{domain}] === ALL DONE in {elapsed:.0f}s ({elapsed/60:.1f}min) ===", log_file)
    consolidate_shards(domain, log_file)
    log(f"[{domain}] Done!", log_file)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--domains", type=str, default="math,code,synth")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    domains = [d.strip() for d in args.domains.split(",")]
    log(f"Starting domains: {domains} with {args.workers} workers each")

    # Run sequentially (one after another) to avoid CPU overcommit
    for domain in domains:
        log(f"\n{'='*60}")
        log(f"Starting {domain} with {args.workers} workers")
        log(f"{'='*60}")
        run_domain(domain, args.workers)
