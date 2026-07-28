#!/usr/bin/env python3
"""Parallel web tier tokenization — spawns N workers, each processing a chunk of parquet files."""

import sys, os, gc, time, re, json, subprocess, multiprocessing
from pathlib import Path
from collections import Counter
import numpy as np
import pyarrow.parquet as pq

os.environ["TOKENIZERS_PARALLELISM"] = "false"

ROOT = Path("/home/kenpeter/work/data")
SRC_DIR = ROOT / "_raw_original" / "fineweb-edu"
OUT_BASE = ROOT  # _shards_web_{tier} dirs live here
TIERS = ["easy", "medium", "hard"]
SEQ_LEN = 2048
FLUSH_TOKENS = 100_000_000
NUM_WORKERS = 8

LOG_FILE = Path("/home/kenpeter/work/data/tokenize_web_parallel.log")

def log(msg):
    ts = time.strftime('%H:%M:%S')
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")

def worker_func(worker_id, file_paths):
    """Process a list of parquet files, write .bin shards with worker prefix."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("filter_all", ROOT / "filter_all.py")
    filter_all = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(filter_all)
    classify_document = filter_all.classify_document

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")
    log(f"[W{worker_id}] Tokenizer loaded, {len(file_paths)} files to process")

    # Worker-specific output dirs to avoid conflicts
    out_dirs = {}
    for t in TIERS:
        d = OUT_BASE / f"_shards_web_{t}" / f"worker_{worker_id}"
        d.mkdir(parents=True, exist_ok=True)
        out_dirs[t] = d

    # Find existing shards for this worker
    def next_shard(tier):
        existing = sorted(out_dirs[tier].glob("*.bin"))
        if existing:
            nums = [int(re.search(r'train-(\d+)\.bin', p.name).group(1)) for p in existing if re.search(r'train-(\d+)\.bin', p.name)]
            return max(nums) + 1 if nums else 0
        return 0

    shard_counters = {t: next_shard(t) for t in TIERS}
    tier_buffers = {t: [] for t in TIERS}
    tier_totals = {t: 0 for t in TIERS}
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
            log(f"  [{wp}/{tier}] Shard: {len(flat):,} tokens -> {out_path.name}")
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
            log(f"[{wp}] ❌ Skipping {fname}: {e}")
            continue

        t0 = time.time()
        log(f"[{wp}] Processing {fname} ({n_rows} rows)...")
        file_kept = 0

        for batch_idx, batch in enumerate(pf.iter_batches(batch_size=1000, columns=["text"])):
            for row in batch.to_pylist():
                stats["total"] += 1
                text = row.get("text", "")
                if not isinstance(text, str):
                    text = str(text) if text is not None else ""

                ok, reason, q_score, tier = classify_document(text, "web")
                if not ok:
                    continue

                stats["kept"] += 1
                file_kept += 1
                stats["quality_sum"] += q_score

                toks = tokenizer.encode(text, add_special_tokens=False)
                append_tokens(tier, toks)

            if batch_idx % 5 == 0:
                elapsed = time.time() - t0
                pct = (batch_idx * 1000) / n_rows * 100
                mem = [f"{t}:{sum(len(a) for a in tier_buffers[t])//1000000}M" for t in TIERS]
                log(f"[{wp}] {batch_idx*1000}/{n_rows} rows ({pct:.0f}%) | kept={file_kept} | "
                    f"totals: easy={tier_totals['easy']//1000000}M med={tier_totals['medium']//1000000}M "
                    f"hard={tier_totals['hard']//1000000}M | buf=[{', '.join(mem)}] | {elapsed:.0f}s")

        elapsed = time.time() - t0
        log(f"[{wp}] Done {fname}: {file_kept} kept in {elapsed:.1f}s")

    # Flush remaining
    for t in TIERS:
        flush_tier(t)

    log(f"[{wp}] FINAL: kept={stats['kept']} | tokens: easy={tier_totals['easy']:,} "
        f"med={tier_totals['medium']:,} hard={tier_totals['hard']:,}")

def consolidate_shards():
    """Merge all worker_*/train-*.bin into consolidated train-*.bin files per tier."""
    import shutil
    for tier in TIERS:
        tier_dir = OUT_BASE / f"_shards_web_{tier}"
        worker_dirs = sorted(tier_dir.glob("worker_*"))
        if not worker_dirs:
            continue

        all_shards = []
        for wd in worker_dirs:
            all_shards.extend(sorted(wd.glob("*.bin")))

        # Sort by (worker_id, shard_idx)
        def sort_key(p):
            parts = p.stem.split("-")  # train-{shard}
            return int(parts[1])
        all_shards.sort(key=sort_key)

        log(f"[consolidate] {tier}: merging {len(all_shards)} shards from {len(worker_dirs)} workers")

        # Rename/move to tier dir with sequential naming
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
            log(f"[consolidate] {tier}: {shard_path.name} -> {new_name}")
            next_idx += 1

        # Clean up empty worker dirs
        for wd in worker_dirs:
            remaining = list(wd.glob("*"))
            if not remaining:
                wd.rmdir()
                log(f"[consolidate] Removed empty {wd}")

        # Final count
        final_shards = sorted(tier_dir.glob("train-*.bin"))
        total_tokens = sum(len(np.fromfile(str(s), dtype=np.uint16)) for s in final_shards)
        log(f"[consolidate] {tier}: {len(final_shards)} shards, {total_tokens:,} tokens")

if __name__ == "__main__":
    import shutil
    # Clear old log
    LOG_FILE.unlink(missing_ok=True)

    parquet_files = sorted(SRC_DIR.glob("*.parquet"))
    log(f"Total {len(parquet_files)} parquet files, splitting across {NUM_WORKERS} workers")

    # Split files among workers
    chunks = [[] for _ in range(NUM_WORKERS)]
    for i, pf in enumerate(parquet_files):
        chunks[i % NUM_WORKERS].append(pf)

    for wi, chunk in enumerate(chunks):
        log(f"  Worker {wi}: {len(chunk)} files")

    # Spawn workers
    t_start = time.time()
    with multiprocessing.Pool(NUM_WORKERS) as pool:
        results = pool.starmap(worker_func, [(wi, chunk) for wi, chunk in enumerate(chunks)])

    elapsed = time.time() - t_start
    log(f"\n=== ALL WORKERS DONE in {elapsed:.0f}s ({elapsed/60:.1f}min) ===")
    log("Consolidating shards...")

    # Consolidate
    consolidate_shards()
    log("Done!")
