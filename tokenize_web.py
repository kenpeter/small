#!/usr/bin/env python3
"""Fast tokenize fineweb-edu parquets → _shards_web_easy .bin shards.
Skips heavy quality filtering — fineweb-edu is already pre-filtered."""
import os, sys, gc, time
from pathlib import Path
import numpy as np
from transformers import AutoTokenizer
import pyarrow.parquet as pq

os.environ["TOKENIZERS_PARALLELISM"] = "false"

SRC_DIR = Path("/home/kenpeter/work/data/_raw_original/fineweb-edu")
OUT_DIR = Path("/home/kenpeter/work/data/_shards_web_easy")
OUT_DIR.mkdir(parents=True, exist_ok=True)
SEQ_LEN = 2048

log = lambda msg: print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")
log("Tokenizer loaded")

parquet_files = sorted(SRC_DIR.glob("*.parquet"))
log(f"Found {len(parquet_files)} parquet files")

buf = []  # list of token arrays
shard_idx = 0
total_tokens = 0

for pf_path in parquet_files:
    fname = pf_path.name
    log(f"Processing {fname}...")
    pf = pq.ParquetFile(str(pf_path))
    n_rows = pf.metadata.num_rows
    t0 = time.time()

    for batch_idx, batch in enumerate(pf.iter_batches(batch_size=2000, columns=["text"])):
        texts = [row.get("text", "") or "" for row in batch.to_pylist()]
        texts = [t[:50000] for t in texts]  # truncate very long
        
        encoded = tokenizer(texts, add_special_tokens=False, truncation=False)
        for ids in encoded["input_ids"]:
            arr = np.array(ids, dtype=np.uint16)
            if len(arr) > 0:
                buf.append(arr)
                total_tokens += len(arr)
        
        # Flush full shards
        while buf and sum(len(a) for a in buf) >= 100_000_000:  # 100M tokens per shard
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

# Flush remaining
if buf:
    flat = np.concatenate(buf)
    flat = flat[: (len(flat) // SEQ_LEN) * SEQ_LEN]
    if len(flat) > 0:
        out_path = OUT_DIR / f"train-{shard_idx:06d}.bin"
        flat.tofile(str(out_path))
        shard_idx += 1
        log(f"Final shard {shard_idx}: {len(flat):,} tokens -> {out_path.name}")

log(f"DONE: {shard_idx} shards, {total_tokens:,} total tokens")
