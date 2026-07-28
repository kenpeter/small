#!/usr/bin/env python3
"""Memory-efficient web tier classification + tokenization.
Uses filter_all.py's classify_document but flushes every 100M tokens per tier."""

import sys, os, gc, time, re, json, string
from pathlib import Path
from collections import Counter
from typing import Tuple
import numpy as np

os.environ["TOKENIZERS_PARALLELISM"] = "false"

ROOT = Path("/home/kenpeter/work/data")
SRC_DIR = ROOT / "_raw_original" / "fineweb-edu"
OUT_DIRS = {
    "easy":   ROOT / "_shards_web_easy",
    "medium": ROOT / "_shards_web_medium",
    "hard":   ROOT / "_shards_web_hard",
}
for d in OUT_DIRS.values():
    d.mkdir(parents=True, exist_ok=True)

SEQ_LEN = 2048
FLUSH_TOKENS=100_000_000  # 100M tokens per shard

LOG_FILE = Path("/home/kenpeter/work/data/tokenize_web_tiers.log")
def log(msg):
    ts = time.strftime('%H:%M:%S')
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")

# ─── Import classification from filter_all ───
sys.path.insert(0, str(ROOT))
import importlib.util
spec = importlib.util.spec_from_file_location("filter_all", ROOT / "filter_all.py")
filter_all = importlib.util.module_from_spec(spec)
spec.loader.exec_module(filter_all)

classify_document = filter_all.classify_document

# ─── Tokenizer ───
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")
log("Tokenizer loaded")

# Resume tracking
COMPLETED_FILE = ROOT / "tokenize_web_tiers_completed.txt"
completed = set()
if COMPLETED_FILE.exists():
    completed = set(line.strip() for line in COMPLETED_FILE.read_text().splitlines() if line.strip())
    log(f"Resume: {len(completed)} parquets already completed")

parquet_files = sorted(SRC_DIR.glob("*.parquet"))
parquet_files = [p for p in parquet_files if p.name not in completed]
log(f"Found {len(parquet_files)} parquet files ({len(completed)} already done)")

# Find next shard index per tier
import re as re_module
def next_shard_idx(tier):
    existing = sorted(OUT_DIRS[tier].glob("*.bin"))
    if existing:
        nums = [int(re_module.search(r'train-(\d+)\.bin', p.name).group(1)) for p in existing if re_module.search(r'train-(\d+)\.bin', p.name)]
        return max(nums) + 1 if nums else 0
    return 0

shard_counters = {t: next_shard_idx(t) for t in ["easy", "medium", "hard"]}
tier_buffers = {t: [] for t in ["easy", "medium", "hard"]}
tier_totals = {t: 0 for t in ["easy", "medium", "hard"]}
stats = {"total": 0, "kept": 0, "reasons": Counter(), "quality_sum": 0.0}

def flush_tier(tier):
    buf = tier_buffers[tier]
    if not buf:
        return
    flat = np.concatenate(buf)
    flat = flat[: (len(flat) // SEQ_LEN) * SEQ_LEN]
    if len(flat) > 0:
        out_path = OUT_DIRS[tier] / f"train-{shard_counters[tier]:06d}.bin"
        flat.tofile(str(out_path))
        shard_counters[tier] += 1
        log(f"  [{tier}] Shard: {len(flat):,} tokens -> {out_path.name}")
    tier_buffers[tier] = []
    del flat
    gc.collect()

def append_tokens(tier, toks):
    if not toks:
        return
    arr = np.array(toks, dtype=np.uint16)
    tier_buffers[tier].append(arr)
    tier_totals[tier] += len(arr)
    # Check if need to flush
    total = sum(len(a) for a in tier_buffers[tier])
    if total >= FLUSH_TOKENS:
        flush_tier(tier)

import pyarrow.parquet as pq

for pf_path in parquet_files:
    fname = pf_path.name
    try:
        pf = pq.ParquetFile(str(pf_path))
        n_rows = pf.metadata.num_rows
    except Exception as e:
        log(f"  ❌ Skipping {fname}: {e}")
        continue
    
    t0 = time.time()
    log(f"Processing {fname} ({n_rows} rows)...")
    file_kept = 0
    file_total = 0
    
    for batch_idx, batch in enumerate(pf.iter_batches(batch_size=1000, columns=["text"])):
        for row in batch.to_pylist():
            file_total += 1
            stats["total"] += 1
            text = row.get("text", "")
            if not isinstance(text, str):
                text = str(text) if text is not None else ""
            
            ok, reason, q_score, tier = classify_document(text, "web")
            if not ok:
                stats["reasons"][reason] += 1
                continue
            
            stats["kept"] += 1
            file_kept += 1
            stats["quality_sum"] += q_score
            
            toks = tokenizer.encode(text, add_special_tokens=False)
            append_tokens(tier, toks)
        
        if batch_idx % 5 == 0:
            elapsed = time.time() - t0
            pct = (batch_idx * 1000) / n_rows * 100
            mem = [f"{t}:{sum(len(a) for a in tier_buffers[t])//1000000}M" for t in ["easy","medium","hard"]]
            log(f"  {batch_idx*1000}/{n_rows} rows ({pct:.0f}%) | kept={file_kept} | "
                f"totals: easy={tier_totals['easy']//1000000}M med={tier_totals['medium']//1000000}M hard={tier_totals['hard']//1000000}M | "
                f"buf=[{', '.join(mem)}] | {elapsed:.0f}s")
    
    elapsed = time.time() - t0
    log(f"  Done {fname}: {file_kept}/{file_total} kept in {elapsed:.1f}s")
    with open(COMPLETED_FILE, "a") as cf:
        cf.write(pf_path.name + "\n")

# Flush remaining
for tier in ["easy", "medium", "hard"]:
    flush_tier(tier)

# Summary
total_kept = stats["kept"]
total_all = stats["total"]
rate = total_kept / max(total_all, 1) * 100
avg_q = stats["quality_sum"] / max(total_kept, 1)
print(f"\n=== FINAL ===")
print(f"  Docs: {total_kept:,}/{total_all:,} kept ({rate:.1f}%) | avg_quality={avg_q:.3f}")
for t in ["easy", "medium", "hard"]:
    print(f"  {t}: {tier_totals[t]:,} tokens ({tier_totals[t]/1e9:.2f}B)")
print(f"  Top rejects: {', '.join(f'{r}({c})' for r,c in stats['reasons'].most_common(5))}")
print(f"  Shard counters: easy={shard_counters['easy']} medium={shard_counters['medium']} hard={shard_counters['hard']}")
