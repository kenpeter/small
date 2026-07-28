#!/usr/bin/env python3
"""Continue tokenizing remaining fineweb-edu parquets → append to _shards_web_easy."""
import os, sys, gc, time
from pathlib import Path
import numpy as np
from transformers import AutoTokenizer
import pyarrow.parquet as pq

os.environ["TOKENIZERS_PARALLELISM"] = "false"

SRC_DIR = Path("/home/kenpeter/work/data/_raw_original/fineweb-edu")
OUT_DIR = Path("/home/kenpeter/work/data/_shards_web_easy")
SEQ_LEN = 2048

log = lambda msg: print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")
log("Tokenizer loaded")

parquet_files = sorted(SRC_DIR.glob("*.parquet"))
log(f"Found {len(parquet_files)} parquet files to process")

# Find next shard index from existing shards
existing = sorted(OUT_DIR.glob("train-*.bin"))
shard_idx = 0
if existing:
    import re
    nums = [int(re.search(r'train-(\d+)\.bin', p.name).group(1)) for p in existing]
    shard_idx = max(nums) + 1
log(f"Starting shard index: {shard_idx}")

buf = []
total_tokens = 0

for pf_path in parquet_files:
    fname = pf_path.name
    log(f"Processing {fname}...")
    try:
        pf = pq.ParquetFile(str(pf_path))
        n_rows = pf.metadata.num_rows
    except Exception as e:
        log(f"  ❌ Skipping corrupt file {fname}: {e}")
        continue
    t0 = time.time()

    for batch_idx, batch in enumerate(pf.iter_batches(batch_size=2000, columns=["text"])):
        texts = [row.get("text", "") or "" for row in batch.to_pylist()]
        texts = [t[:50000] for t in texts]

        encoded = tokenizer(texts, add_special_tokens=False, truncation=False)
        for ids in encoded["input_ids"]:
            arr = np.array(ids, dtype=np.uint16)
            if len(arr) > 0:
                buf.append(arr)
                total_tokens += len(arr)

        while buf and sum(len(a) for a in buf) >= 100_000_000:
            chunks = []
            sz = 0
            while buf and sz < 100_000_000:
                a = buf.pop(0)
                chunks.append(a)
                sz += len(a)
            flat = np.concatenate(chunks)[: (sz // SEQ_LEN) * SEQ_LEN]
            if len(flat) > 0:
                out_path = OUT_DIR / f"train-{shard_idx:06d}.bin"
                flat.tofile(str(out_path))
                shard_idx += 1
                log(f"  Shard {shard_idx}: {len(flat):,} tokens -> {out_path.name}")
            del flat, chunks
            gc.collect()

        if batch_idx % 5 == 0:
            elapsed = time.time() - t0
            pct = (batch_idx * 2000) / n_rows * 100
            log(f"  {batch_idx*2000}/{n_rows} rows ({pct:.0f}%) | {total_tokens:,} total tokens | {elapsed:.0f}s")

        gc.collect()

    elapsed = time.time() - t0
    log(f"  Done {fname}: {n_rows} rows in {elapsed:.1f}s")
    gc.collect()

# Flush remaining buffer
if buf:
    flat = np.concatenate(buf)
    flat = flat[: (len(flat) // SEQ_LEN) * SEQ_LEN]
    if len(flat) > 0:
        out_path = OUT_DIR / f"train-{shard_idx:06d}.bin"
        flat.tofile(str(out_path))
        shard_idx += 1
        log(f"Final shard: {len(flat):,} tokens -> {out_path.name}")

log(f"DONE: {shard_idx} total shards, {total_tokens:,} total tokens")
