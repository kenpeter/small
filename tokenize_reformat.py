#!/usr/bin/env python3
"""Tokenize reformatted textbook+QA JSONL into pretraining .bin shards."""
import os, sys, json, time, struct
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer

os.environ["TOKENIZERS_PARALLELISM"] = "false"

TOKENIZER_NAME = "HuggingFaceTB/SmolLM2-135M"
INPUT_FILES = [
    "/home/kenpeter/work/data/combined_textbook.jsonl",
    "/home/kenpeter/work/data/combined_qa.jsonl",
]
OUTPUT_DIR = Path("/home/kenpeter/work/data/_shards_reformat_easy")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TOKENS_PER_SHARD = 50_000_000  # ~100MB per shard
MAX_TEXT_LEN = 50_000
BATCH_SIZE = 500

def main():
    print("=" * 60)
    print("TOKENIZE REFORMATTED DATA \u2192 .bin SHARDS")
    print("=" * 60)

    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, trust_remote_code=True)
    print(f"  Vocab: {len(tokenizer)}")

    shard_buf = []
    shard_idx = 0
    total_tokens = 0
    total_docs = 0

    for jsonl_path in INPUT_FILES:
        path = Path(jsonl_path)
        if not path.exists():
            print(f"  \u26a0 {path.name} not found, skipping")
            continue
        print(f"\n\U0001f4c4 {path.name} ({path.stat().st_size/1024**2:.1f} MB)...")

        with open(path, 'r', encoding='utf-8') as f:
            batch_texts = []
            for line in f:
                data = json.loads(line)
                text = data.get('text', '')[:MAX_TEXT_LEN]
                if len(text) < 20:
                    continue
                batch_texts.append(text)

                if len(batch_texts) >= BATCH_SIZE:
                    encoded = tokenizer(batch_texts, add_special_tokens=False, truncation=False)
                    for ids in encoded["input_ids"]:
                        arr = np.array(ids, dtype=np.uint16)
                        shard_buf.append(arr)
                        total_tokens += len(arr)
                        total_docs += 1

                    while shard_buf and sum(len(a) for a in shard_buf) >= TOKENS_PER_SHARD:
                        shard_idx = _flush_shard(shard_buf, shard_idx)

                    batch_texts = []

            # Flush remaining batch
            if batch_texts:
                encoded = tokenizer(batch_texts, add_special_tokens=False, truncation=False)
                for ids in encoded["input_ids"]:
                    arr = np.array(ids, dtype=np.uint16)
                    shard_buf.append(arr)
                    total_tokens += len(arr)
                    total_docs += 1

                while shard_buf and sum(len(a) for a in shard_buf) >= TOKENS_PER_SHARD:
                    shard_idx = _flush_shard(shard_buf, shard_idx)

        print(f"  {total_tokens:,} tok so far")

    # Flush final remainder
    if shard_buf:
        all_tokens = np.concatenate(shard_buf)
        out_path = OUTPUT_DIR / f"train-{shard_idx:06d}.bin"
        all_tokens.tofile(str(out_path))
        size_mb = out_path.stat().st_size / 1e6
        print(f"\n  \u2192 Final shard {shard_idx}: {len(all_tokens):,} tok ({size_mb:.1f} MB)")
        shard_idx += 1

    print(f"\n{'='*60}")
    print(f"\u2705 DONE: {total_docs} docs, {total_tokens:,} tokens")
    print(f"   Shards: {shard_idx} files in {OUTPUT_DIR}")
    total_size = sum(f.stat().st_size for f in OUTPUT_DIR.glob("*.bin"))
    print(f"   Total size: {total_size/1024**3:.2f} GB")
    print(f"{'='*60}")


def _flush_shard(shard_buf, shard_idx):
    all_tokens = np.concatenate(shard_buf)
    if len(all_tokens) >= TOKENS_PER_SHARD:
        out_path = OUTPUT_DIR / f"train-{shard_idx:06d}.bin"
        all_tokens[:TOKENS_PER_SHARD].tofile(str(out_path))
        size_mb = out_path.stat().st_size / 1e6
        print(f"  \u2192 Shard {shard_idx}: {TOKENS_PER_SHARD:,} tok ({size_mb:.1f} MB)")
        remainder = [all_tokens[TOKENS_PER_SHARD:]] if len(all_tokens) > TOKENS_PER_SHARD else []
        shard_buf[:] = remainder
        return shard_idx + 1
    shard_buf[:] = [all_tokens]
    return shard_idx


if __name__ == "__main__":
    main()
