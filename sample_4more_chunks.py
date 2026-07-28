#!/usr/bin/env python3
"""Sample 3,200 more docs, split into 4 chunks of 800."""
import json, random
from pathlib import Path
import pyarrow.parquet as pq

random.seed(7777)
DOMAINS = {"web": "text", "math": "text", "synth": "text", "code": "content"}
N_PER_DOMAIN = 200  # 200 x 4 domains = 800 per chunk
MIN_CHARS = 500
MAX_FILES = 15

sources = {
    "web":   sorted(Path("/home/kenpeter/work/data/_raw_original/fineweb-edu").glob("train-*.parquet")),
    "math":  sorted(Path("/mnt/file_drive/data/finemath-3plus").glob("train-*.parquet"))[:8] +
             sorted(Path("/mnt/file_drive/data/open-web-math").glob("train-*.parquet"))[:7],
    "synth": sorted(Path("/mnt/file_drive/data/cosmopedia").glob("train-*.parquet")),
    "code":  sorted(Path("/mnt/file_drive/data/github-code").glob("train-*.parquet")),
}

all_docs = []
for domain, parquets in sources.items():
    col = DOMAINS[domain]
    selected = random.sample(parquets, min(MAX_FILES, len(parquets)))
    n_per = max(5, N_PER_DOMAIN // len(selected))
    pool = []
    for p in selected:
        try:
            pf = pq.ParquetFile(str(p))
            batch = next(pf.iter_batches(batch_size=n_per * 10, columns=[col]))
            for r in batch.to_pylist():
                if isinstance(r.get(col), str) and len(r[col].strip()) >= MIN_CHARS:
                    pool.append(r[col])
        except:
            pass
    random.shuffle(pool)
    for t in pool[:N_PER_DOMAIN * 4]:  # need 800 per domain (200 x 4 chunks)
        all_docs.append({"text": t, "domain": domain})
    print(f"{domain}: {len(all_docs)} in pool")

random.shuffle(all_docs)
print(f"Total: {len(all_docs)} docs")

out_dir = Path("/home/kenpeter/work/data/_reformatted")
# Split into 4 chunks of ~800
cs = len(all_docs) // 4
for i in range(4):
    chunk = all_docs[i*cs:(i+1)*cs] if i < 3 else all_docs[i*cs:]
    idx = i + 4  # chunks 4,5,6,7
    path = out_dir / f"_input_chunk_{idx}.jsonl"
    with open(path, "w") as f:
        for d in chunk:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(f"Chunk {idx}: {len(chunk)} docs ({path.stat().st_size/1024**2:.1f} MB)")
