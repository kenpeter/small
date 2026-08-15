#!/usr/bin/env python3
"""Tokenize gold_set.jsonl (3 methods + 5 variants docs) into pretraining .bin shards.

Writes uint16 token arrays (no header) into _shards_gold/, the same format the
stratified loader reads (np.fromfile(dtype=np.uint16), seq chunks of 2048).
"""
import os, sys, json, time
from pathlib import Path
import numpy as np
from transformers import AutoTokenizer

TOKENIZER_NAME = "HuggingFaceTB/SmolLM2-135M"
INPUT_FILE = Path("/home/kenpeter/work/data/_gold/gold_combined.jsonl")
OUT_DIR = Path("/home/kenpeter/work/data/_shards_gold")
TOKENS_PER_SHARD = 50_000_000  # 100MB per shard (uint16)


def main():
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, trust_remote_code=True)
    print(f"  Vocab: {len(tokenizer)}", flush=True)

    shard_buf = []      # list of token arrays for current shard
    shard_idx = 0
    total_tokens = 0
    n_docs = 0
    skipped = 0

    with open(INPUT_FILE) as f:
        for line in f:
            doc = json.loads(line)
            text = doc.get("text", "")
            if not text or len(text) < 50:
                skipped += 1
                continue
            n_docs += 1
            encoded = tokenizer(text, add_special_tokens=False, truncation=False)
            ids = np.array(encoded["input_ids"], dtype=np.uint16)
            shard_buf.append(ids)
            total_tokens += len(ids)
            while shard_buf and sum(len(a) for a in shard_buf) >= TOKENS_PER_SHARD:
                all_tokens = np.concatenate(shard_buf)
                shard_buf = []
                out_path = OUT_DIR / f"gold-{shard_idx:05d}.bin"
                all_tokens[:TOKENS_PER_SHARD].tofile(str(out_path))
                print(f"  wrote {out_path.name} ({TOKENS_PER_SHARD:,} tokens)", flush=True)
                shard_idx += 1
                if len(all_tokens) > TOKENS_PER_SHARD:
                    shard_buf.append(all_tokens[TOKENS_PER_SHARD:])

    if shard_buf:
        all_tokens = np.concatenate(shard_buf)
        out_path = OUT_DIR / f"gold-{shard_idx:05d}.bin"
        all_tokens.tofile(str(out_path))
        print(f"  wrote {out_path.name} ({len(all_tokens):,} tokens)", flush=True)
        shard_idx += 1

    print(f"Done: {n_docs} docs, {total_tokens:,} tokens, {shard_idx} shards, "
          f"{time.time()-t0:.1f}s (skipped {skipped})", flush=True)


if __name__ == "__main__":
    main()
