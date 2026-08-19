#!/usr/bin/env python3
"""Tokenize code_gold_qwen.jsonl into pretraining .bin shards (uint16 tokens,
2048-seq chunks, same format as other tiers). Mirror of tokenize_gold.py."""
import json, time
from pathlib import Path
import numpy as np
from transformers import AutoTokenizer

TOKENIZER_NAME = "HuggingFaceTB/SmolLM2-135M"
INPUT_FILE = Path("/home/kenpeter/work/data/_code_gold_qwen/code_gold_qwen.jsonl")
OUT_DIR = Path("/home/kenpeter/work/data/_shards_code_gold")
TOKENS_PER_SHARD = 50_000_000

def main():
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, trust_remote_code=True)
    print(f"  Vocab: {len(tokenizer)}", flush=True)
    shard_buf = []
    shard_idx = 0
    total_tokens = 0
    n_docs = 0
    with open(INPUT_FILE) as f:
        for line in f:
            doc = json.loads(line)
            text = doc.get("text", "")
            if not text:
                continue
            n_docs += 1
            ids = np.array(tokenizer(text, add_special_tokens=False)["input_ids"], dtype=np.uint16)
            shard_buf.append(ids)
            total_tokens += len(ids)
            while shard_buf and sum(len(a) for a in shard_buf) >= TOKENS_PER_SHARD:
                all_tokens = np.concatenate(shard_buf)
                shard_buf = []
                out = OUT_DIR / f"codegold-{shard_idx:05d}.bin"
                all_tokens[:TOKENS_PER_SHARD].tofile(str(out))
                print(f"  wrote {out.name}", flush=True)
                shard_idx += 1
                if len(all_tokens) > TOKENS_PER_SHARD:
                    shard_buf.append(all_tokens[TOKENS_PER_SHARD:])
    if shard_buf:
        all_tokens = np.concatenate(shard_buf)
        out = OUT_DIR / f"codegold-{shard_idx:05d}.bin"
        all_tokens.tofile(str(out))
        print(f"  wrote {out.name} ({len(all_tokens):,} tokens)", flush=True)
        shard_idx += 1
    print(f"Done: {n_docs} docs, {total_tokens:,} tokens, {shard_idx} shards, {time.time()-t0:.1f}s", flush=True)

if __name__ == "__main__":
    main()
