#!/usr/bin/env python3
"""Sample 600 per domain, split into 3 chunks."""
import json, random
from pathlib import Path
import pyarrow.parquet as pq

random.seed(42)
DOMAINS = {"web": "text", "math": "text", "synth": "text", "code": "content"}
N_PER_DOMAIN = 600
MIN_CHARS = 500
MAX_FILES = 20

domain_parquets = {
    "web":   sorted(Path("/home/kenpeter/work/data/_raw_original/fineweb-edu").glob("train-*.parquet")),
    "math":  sorted(Path("/mnt/file_drive/data/finemath-3plus").glob("train-*.parquet"))[:10] +
             sorted(Path("/mnt/file_drive/data/open-web-math").glob("train-*.parquet"))[:10],
    "synth": sorted(Path("/mnt/file_drive/data/cosmopedia").glob("train-*.parquet")),
    "code":  sorted(Path("/mnt/file_drive/data/github-code").glob("train-*.parquet")),
}

all_docs = []
for domain, parquets in domain_parquets.items():
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
            random.shuffle(pool)
        except Exception as e:
            print(f"  skip {p.name}: {e}")
    picked = pool[:N_PER_DOMAIN]
    for t in picked:
        all_docs.append({"text": t, "domain": domain})
    print(f"{domain}: {len(picked)} docs")

random.shuffle(all_docs)
print(f"Total: {len(all_docs)}")

out_dir = Path("/home/kenpeter/work/data/_reformatted")
cs = len(all_docs) // 3
for i in range(3):
    chunk = all_docs[i*cs:(i+1)*cs] if i < 2 else all_docs[i*cs:]
    path = out_dir / f"_input_chunk_{i}.jsonl"
    with open(path, "w") as f:
        for d in chunk:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(f"Chunk {i}: {len(chunk)} docs ({path.stat().st_size/1024**2:.1f} MB)")
