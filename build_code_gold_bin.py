#!/usr/bin/env python3
"""Tokenize + replicate the MERGED expanded code_gold corpus into the .bin shard.

Combines tokenize_code_gold.py + replicate_code_gold.py for the merged corpus.
Usage: venv/bin/python build_code_gold_bin.py --factor 150
"""
import json, argparse, random
from pathlib import Path
import numpy as np
from transformers import AutoTokenizer

TOKENIZER_NAME = "HuggingFaceTB/SmolLM2-135M"
SRC = Path("/home/kenpeter/work/data/_code_gold_hq/code_gold_merged_expanded.jsonl")
OUT_DIR = Path("/home/kenpeter/work/data/_shards_code_gold")
SEQ_LEN = 2048

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--factor", type=int, default=150)
    args = ap.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, trust_remote_code=True)
    docs = []
    with open(SRC) as f:
        for line in f:
            try:
                t = json.loads(line)["text"]
                if t:
                    docs.append(t)
            except Exception:
                continue
    print(f"{len(docs)} unique docs")
    seq_list = []
    total_tok = 0
    for t in docs:
        ids = np.array(tokenizer(t, add_special_tokens=False)["input_ids"], dtype=np.uint16)
        total_tok += len(ids)
        for s in range(0, len(ids), SEQ_LEN):
            chunk = ids[s:s+SEQ_LEN]
            if len(chunk) < SEQ_LEN:
                chunk = np.pad(chunk, (0, SEQ_LEN - len(chunk)))
            seq_list.append(chunk)
    print(f"{len(seq_list)} unique seqs, {total_tok:,} unique tokens (~{total_tok*args.factor:,} replicated)")

    all_seqs = []
    for _ in range(args.factor):
        random.shuffle(seq_list)
        all_seqs.extend(seq_list)
    random.shuffle(all_seqs)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    arr = np.stack(all_seqs).ravel()
    out = OUT_DIR / "codegold-00000.bin"
    arr.tofile(str(out))
    print(f"{len(all_seqs):,} seqs × 2048 = {len(arr):,} tokens → {out} ({out.stat().st_size/1e6:.0f} MB)")

if __name__ == "__main__":
    main()
