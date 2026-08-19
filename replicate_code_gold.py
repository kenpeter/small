#!/usr/bin/env python3
"""Replicate the verified code_gold corpus to build real volume for 'heavy use'.

The correct-solution corpus is compact (241 docs ≈ 26K tokens ≈ 12 seqs) — too
small to register in a 20M-sample epoch. Replication drills the verified-correct
patterns hard (repetition is exactly how greedy internalizes correct base cases),
bringing code_gold to a real pool comparable to code_hard.

Usage: venv/bin/python replicate_code_gold.py --factor 150
"""
import json, argparse, random
from pathlib import Path
import numpy as np
from transformers import AutoTokenizer

TOKENIZER_NAME = "HuggingFaceTB/SmolLM2-135M"
SRC = Path("/home/kenpeter/work/data/_code_gold_qwen/code_gold_expanded.jsonl")
OUT_DIR = Path("/home/kenpeter/work/data/_shards_code_gold")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--factor", type=int, default=150)
    args = ap.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, trust_remote_code=True)
    docs = []
    with open(SRC) as f:
        for line in f:
            try:
                docs.append(json.loads(line)["text"])
            except Exception:
                continue
    seq_list = []
    for t in docs:
        ids = np.array(tokenizer(t, add_special_tokens=False)["input_ids"], dtype=np.uint16)
        # split each doc into 2048-token sequences (pad last)
        for s in range(0, len(ids), 2048):
            chunk = ids[s:s+2048]
            if len(chunk) < 2048:
                chunk = np.pad(chunk, (0, 2048 - len(chunk)))
            seq_list.append(chunk)
    # replicate with re-shuffle for variety in ordering
    all_seqs = []
    for _ in range(args.factor):
        random.shuffle(seq_list)
        all_seqs.extend(seq_list)
    random.shuffle(all_seqs)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # write as one big bin (uint16)
    arr = np.stack(all_seqs).ravel()
    out = OUT_DIR / "codegold-00000.bin"
    arr.tofile(str(out))
    print(f"{len(seq_list)} unique seqs × {args.factor} = {len(all_seqs)} seqs, "
          f"{len(arr):,} tokens → {out} ({out.stat().st_size/1e6:.1f} MB)")

if __name__ == "__main__":
    main()
