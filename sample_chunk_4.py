#!/usr/bin/env python3
"""Sample 800 more docs for 4th chunk."""
import json, random
from pathlib import Path
import pyarrow.parquet as pq

random.seed(999)  # different seed for different docs
DOMAINS = {"web": "text", "math": "text", "synth": "text", "code": "content"}
N = 200  # 200 per domain = 800 total
MIN_CHARS = 500
MAX_FILES = 15

parquet_sources = {
    "web": sorted(Path("/home/kenpeter/work/data/_raw_original/fineweb-edu").glob("train-*.parquet")),
    "math": sorted(Path("/mnt/file_drive/data/finemath-3plus").glob("train-*.parquet"))[:8] +
            sorted(Path("/mnt/file_drive/data/open-web-math").glob("train-*.parquet"))[:7],
    "synth": sorted(Path("/mnt/file_drive/data/cosmopedia").glob("train-*.parquet")),
    "code": sorted(Path("/mnt/file_drive/data/github-code").glob("train-*.parquet")),
}

all_docs = []
for domain, parquets in parquet_sources.items():
    col = DOMAINS[domain]
    selected = random.sample(parquets, min(MAX_FILES, len(parquets)))
    n_per = max(5, N // len(selected))
    pool = []
    for p in selected:
        try:
            pf = pq.ParquetFile(str(p))
            batch = next(pf.iter_batches(batch_size=n_per * 10, columns=[col]))
            for r in batch.to_pylist():
                if isinstance(r.get(col), str) and len(r[col].strip()) >= MIN_CHARS:
                    pool.append(r[col])
        except: pass
    random.shuffle(pool)
    picked = pool[:N]
    for t in picked:
        all_docs.append({"text": t, "domain": domain})
    print(f"{domain}: {len(picked)} docs")

random.shuffle(all_docs)
out_dir = Path("/home/kenpeter/work/data/_reformatted")
path = out_dir / "_input_chunk_3.jsonl"
with open(path, "w") as f:
    for d in all_docs:
        f.write(json.dumps(d, ensure_ascii=False) + "\n")
print(f"Chunk 3: {len(all_docs)} docs ({path.stat().st_size/1024**2:.1f} MB)")
