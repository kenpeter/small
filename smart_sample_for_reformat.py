#!/usr/bin/env python3
"""Smart sample up to 200 high-quality docs per domain for reformatting."""
import json, os, sys, random
from pathlib import Path

DATA = Path("/home/kenpeter/work/data")
RAW = DATA / "_raw_original"
N_PER_DOMAIN = 200
MIN_CHARS = 500
OUTPUT = DATA / "_reformatted" / "_input_sample.jsonl"
SEED = 42

# Find parquet files per domain
# Column name per domain (github-code uses "content" not "text")
COLUMN_MAP = {
    "web":   "text",
    "math":  "text",
    "synth": "text",
    "code":  "content",
}

domain_parquets = {
    "web":   sorted((RAW / "fineweb-edu").glob("train-*.parquet")),
    "math":  sorted((Path("/mnt/file_drive/data/finemath-3plus")).glob("train-*.parquet")) +
             sorted((Path("/mnt/file_drive/data/open-web-math")).glob("train-*.parquet")),
    "synth": sorted((Path("/mnt/file_drive/data/cosmopedia")).glob("train-*.parquet")),
    "code":  sorted((Path("/mnt/file_drive/data/github-code")).glob("train-*.parquet")),
}

# For each domain, sample docs evenly across all parquet files
# Only scan up to 20 parquet files per domain (random sample)
MAX_FILES = 20
random.seed(SEED)
total_sampled = 0

with open(OUTPUT, "w") as out:
    for domain, parquets in domain_parquets.items():
        if not parquets:
            print(f"⚠ {domain}: no parquet files found")
            continue

        # Randomly select up to MAX_FILES parquets
        selected = random.sample(parquets, min(MAX_FILES, len(parquets)))
        # How many per file?
        n_per_file = max(5, N_PER_DOMAIN // len(selected))
        domain_docs = []

        for pq_path in selected:
            try:
                import pyarrow.parquet as pq
                col = COLUMN_MAP[domain]
                pf = pq.ParquetFile(str(pq_path))
                batch = next(pf.iter_batches(batch_size=n_per_file * 10, columns=[col]))
                rows = batch.to_pylist()
                candidates = [r[col] for r in rows
                              if isinstance(r.get(col), str) and len(r[col].strip()) >= MIN_CHARS]
                sampled = random.sample(candidates, min(n_per_file, len(candidates)))
                for text in sampled:
                    domain_docs.append({"text": text, "domain": domain})
                print(f"  {domain}/{pq_path.name}: sampled {len(sampled)}/{len(candidates)} valid")
            except Exception as e:
                print(f"  ⚠ {domain}/{pq_path.name}: {e}")
                continue

        # Final sample to exactly N_PER_DOMAIN
        random.shuffle(domain_docs)
        final = domain_docs[:N_PER_DOMAIN]
        for doc in final:
            out.write(json.dumps(doc, ensure_ascii=False) + "\n")
        total_sampled += len(final)
        print(f"  → {domain}: {len(final)} docs ({sum(len(d['text']) for d in final)//1000}K chars)")

print(f"\n✅ Total: {total_sampled} docs → {OUTPUT}")
print(f"   Sizes: {os.path.getsize(OUTPUT)/1024**2:.1f} MB")
