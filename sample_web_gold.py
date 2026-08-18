#!/usr/bin/env python3
"""Sample best-of-best web docs from QRP staging parquets for web-gold reformatting.

Selects docs with highest composite quality (educational_value + writing_style),
writes them to a JSONL for the reformat stage. Dedup by text hash.
"""
import sys, json, random, hashlib
from pathlib import Path
import pyarrow.parquet as pq
from glob import glob

SRC_DIR = Path("/mnt/file_drive/_qrp_staging")
OUT = Path("/home/kenpeter/work/data/_web_gold/sample.jsonl")
N_TARGET = 2000        # total docs to sample
COMPOSITE_P90 = 2.26   # ~top 10% on educ+writing
MIN_LEN = 400          # chars, skip too-short
FILES = 120            # parquet files to scan


def main():
    files = sorted(glob(str(SRC_DIR / "*.parquet")))
    random.seed(42)
    random.shuffle(files)
    files = files[:FILES]
    print(f"Scanning {len(files)} parquets...", flush=True)

    candidates = []  # (composite, text)
    seen = set()
    for fp in files:
        try:
            t = pq.read_table(fp, columns=["text", "educational_value_average",
                                           "writing_style_average", "length"])
        except Exception as e:
            print(f"  skip {fp}: {e}", flush=True)
            continue
        ev = t.column("educational_value_average").to_pylist()
        ws = t.column("writing_style_average").to_pylist()
        texts = t.column("text").to_pylist()
        for i, txt in enumerate(texts):
            txt = (txt or "").strip()
            if len(txt) < MIN_LEN or len(txt) > 6000:
                continue
            h = hashlib.md5(txt[:300].encode()).hexdigest()
            if h in seen:
                continue
            seen.add(h)
            comp = (ev[i] or 0) + (ws[i] or 0)
            if comp >= COMPOSITE_P90:
                candidates.append((comp, txt))
        print(f"  {Path(fp).name}: {len(candidates)} candidates so far", flush=True)

    candidates.sort(reverse=True)  # best first
    picked = candidates[:N_TARGET]
    print(f"Selected {len(picked)} best-of-best docs (composite >= {COMPOSITE_P90:.2f}, "
          f"top {COMPOSITE_P90:.2f} cut)", flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        for i, (comp, txt) in enumerate(picked):
            f.write(json.dumps({"idx": i, "text": txt,
                                "composite": round(comp, 3)}) + "\n")
    print(f"Wrote {OUT} ({OUT.stat().st_size/1e6:.1f} MB)", flush=True)


if __name__ == "__main__":
    main()
