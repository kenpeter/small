#!/usr/bin/env python3
"""Sample 3,000 docs (750 per domain) and split into 3 chunks of 1,000 each."""
import json, os, random
from pathlib import Path

DATA = Path("/home/kenpeter/work/data")
RAW = DATA / "_raw_original"
N_PER_DOMAIN = 750  # 750 x 4 = 3,000 total
MIN_CHARS = 500
SEED = 42

COLUMN_MAP = {"web": "text", "math": "text", "synth": "text", "code": "content"}

domain_parquets = {
    "web":   sorted((RAW / "fineweb-edu").glob("train-*.parquet")),
    "math":  sorted((Path("/mnt/file_drive/data/finemath-3plus")).glob("train-*.parquet")) +
             sorted((Path("/mnt/file_drive/data/open-web-math")).glob("train-*.parquet")),
    "synth": sorted((Path("/mnt/file_drive/data/cosmopedia")).glob("train-*.parquet")),
    "code":  sorted((Path("/mnt/file_drive/data/github-code")).glob("train-*.parquet")),
}

MAX_FILES = 25
random.seed(SEED)
all_docs = []

for domain, parquets in domain_parquets.items():
    if not parquets:
        print(f"⚠ {domain}: no files")
        continue
    selected = random.sample(parquets, min(MAX_FILES, len(parquets)))
    n_per_file = max(5, N_PER_DOMAIN // len(selected))
    col = COLUMN_MAP[domain]
    domain_docs = []

    for pq_path in selected:
        try:
            import pyarrow.parquet as pq
            pf = pq.ParquetFile(str(pq_path))
            batch = next(pf.iter_batches(batch_size=n_per_file * 10, columns=[col]))
            rows = batch.to_pylist()
            candidates = [r[col] for r in rows
                          if isinstance(r.get(col), str) and len(r[col].strip()) >= MIN_CHARS]
            sampled = random.sample(candidates, min(n_per_file, len(candidates)))
            for text in sampled:
                domain_docs.append({"text": text, "domain": domain})
            print(f"  {domain}/{pq_path.name}: +{len(sampled)}")
        except Exception as e:
            print(f"  ⚠ {domain}/{pq_path.name}: {e}")

    random.shuffle(domain_docs)
    final = domain_docs[:N_PER_DOMAIN]
    all_docs.extend(final)
    print(f"  → {domain}: {len(final)} docs ({sum(len(d['text']) for d in final)//1000}K chars)")

random.shuffle(all_docs)
print(f"\n✅ Total: {len(all_docs)} docs")

# Split into 3 chunks of ~1,000 each
out_dir = DATA / "_reformatted"
out_dir.mkdir(exist_ok=True)
chunk_size = len(all_docs) // 3

for i in range(3):
    chunk = all_docs[i*chunk_size:(i+1)*chunk_size] if i < 2 else all_docs[i*chunk_size:]
    path = out_dir / f"_input_chunk_{i}.jsonl"
    with open(path, "w") as f:
        for doc in chunk:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
    print(f"  Chunk {i}: {len(chunk)} docs → {path.name} ({path.stat().st_size/1024**2:.1f} MB)")

print(f"\nTotal across chunks: {sum(1 for i in range(3) for _ in open(out_dir/f'_input_chunk_{i}.jsonl'))}")
