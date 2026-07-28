#!/usr/bin/env python3
"""Create 8 input chunks of 800 docs each from fineweb-edu for reformatting."""
import json, random, os
from pathlib import Path
import pyarrow.parquet as pq

DATA = Path("/home/kenpeter/work/data")
OUT_DIR = DATA / "_reformatted"
OUT_DIR.mkdir(exist_ok=True)

# Use fineweb-edu from raw_original
source_dir = DATA / "_raw_original" / "fineweb-edu"
parquets = sorted(source_dir.glob("train-00000-of-00015.parquet"))
if not parquets:
    parquets = sorted(source_dir.glob("train-*.parquet"))

print(f"Found {len(parquets)} parquet files in {source_dir}")

# 8 chunks × 800 docs = 6400 total
N_CHUNKS = 8
DOCS_PER_CHUNK = 800
MIN_CHARS = 500
SEED = 42

random.seed(SEED)
pool = []

# Load from each parquet
for pq_path in parquets:
    try:
        pf = pq.ParquetFile(str(pq_path))
        # Get total rows if possible
        n_rows = pf.metadata.num_rows
        batch = next(pf.iter_batches(batch_size=n_rows, columns=["text"]))
        for r in batch.to_pylist():
            if isinstance(r.get("text"), str) and len(r["text"].strip()) >= MIN_CHARS:
                pool.append({"text": r["text"], "domain": "web"})
        print(f"  {pq_path.name}: +{sum(1 for r in batch.to_pylist() if isinstance(r.get('text'), str) and len(r['text'].strip()) >= MIN_CHARS)} valid")
    except Exception as e:
        print(f"  ⚠ {pq_path.name}: {e}")

print(f"\nTotal pool: {len(pool)} docs >= {MIN_CHARS} chars")
random.shuffle(pool)

# Take enough for 8 chunks
needed = N_CHUNKS * DOCS_PER_CHUNK
if len(pool) < needed:
    print(f"⚠ Only {len(pool)} docs available, needed {needed}. Reducing chunks.")
    N_CHUNKS = len(pool) // DOCS_PER_CHUNK
    needed = N_CHUNKS * DOCS_PER_CHUNK

pool = pool[:needed]

# Split into chunks
for i in range(N_CHUNKS):
    chunk = pool[i * DOCS_PER_CHUNK : (i + 1) * DOCS_PER_CHUNK]
    path = OUT_DIR / f"_input_chunk_{i}.jsonl"
    with open(path, "w") as f:
        for doc in chunk:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
    print(f"  Chunk {i}: {len(chunk)} docs → {path.name} ({path.stat().st_size/1024**2:.1f} MB)")

print(f"\n✅ Done: {N_CHUNKS} chunks × {DOCS_PER_CHUNK} docs = {len(pool)} total")
