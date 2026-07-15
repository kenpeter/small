#!/usr/bin/env python3
"""
Tokenize filtered JSONL into binary shards.
Target: 3.3B tokens, best quality first.
"""
import os, sys, json, glob, time
from pathlib import Path
import numpy as np
from transformers import AutoTokenizer

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# === CONFIG ===
TOKENIZER_NAME = "HuggingFaceTB/cosmo2-tokenizer"
DATA_ROOT = Path("/home/kenpeter/work/data")
FILTERED_DIR = DATA_ROOT / "_filtered_new"
OUTPUT_DIR = DATA_ROOT / "_shards_v2"  # New clean shards
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Token count target
token_target = 3_300_000_000  # 3.3B tokens
# Tokens per shard (uint16 = 2 bytes per token)
# 256M tokens = 512MB per shard
TOKENS_PER_SHARD = 256_000_000

# Priority order: best quality first
DATASET_PRIORITY = [
    "fineweb-edu",      # 3M docs, diverse educational
    "finemath-3plus",   # 2M docs, math/science
    "finemath",         # 17K docs, math problems
    "open-web-math",    # 33K docs, advanced math
    "stack-python",      # 26K docs, code
    "cosmopedia",        # 5.5K docs, synthetic (lowest priority)
]

MAX_TEXT_LEN = 50_000  # Cap doc length
BATCH_SIZE = 2000

def tokenize_dataset(jsonl_files, tokenizer, target_tokens, shard_buf, shard_idx):
    """Tokenize one dataset's JSONL files, return tokens and updated state."""
    total_tokens = 0
    total_docs = 0
    
    for jsonl_path in jsonl_files:
        print(f"  Tokenizing {jsonl_path.name}...")
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            batch_texts = []
            for line in f:
                data = json.loads(line)
                text = data.get('text', '')[:MAX_TEXT_LEN]
                if len(text) < 50:
                    continue
                batch_texts.append(text)
                
                if len(batch_texts) >= BATCH_SIZE:
                    # Tokenize batch
                    encoded = tokenizer(batch_texts, add_special_tokens=False, truncation=False)
                    for ids in encoded["input_ids"]:
                        arr = np.array(ids, dtype=np.uint16)
                        shard_buf.append(arr)
                        total_tokens += len(arr)
                        total_docs += 1
                    
                    # Flush if buffer full
                    while shard_buf and sum(len(a) for a in shard_buf) >= TOKENS_PER_SHARD:
                        shard_idx = _flush_shard(shard_buf, shard_idx, OUTPUT_DIR, TOKENS_PER_SHARD)
                    
                    batch_texts = []
                    
                    if total_tokens >= target_tokens:
                        return total_tokens, total_docs, shard_buf, shard_idx
        
        # Flush remaining batch
        if batch_texts:
            encoded = tokenizer(batch_texts, add_special_tokens=False, truncation=False)
            for ids in encoded["input_ids"]:
                arr = np.array(ids, dtype=np.uint16)
                shard_buf.append(arr)
                total_tokens += len(arr)
                total_docs += 1
            
            while shard_buf and sum(len(a) for a in shard_buf) >= TOKENS_PER_SHARD:
                shard_idx = _flush_shard(shard_buf, shard_idx, OUTPUT_DIR, TOKENS_PER_SHARD)
        
        if total_tokens >= target_tokens:
            return total_tokens, total_docs, shard_buf, shard_idx
    
    return total_tokens, total_docs, shard_buf, shard_idx

def _flush_shard(shard_buf, shard_idx, out_dir, shard_size):
    """Concatenate buffers and write a shard."""
    all_tokens = np.concatenate(shard_buf)
    
    if len(all_tokens) >= shard_size:
        # Write exactly shard_size tokens
        out_path = out_dir / f"shard_{shard_idx:06d}.bin"
        all_tokens[:shard_size].tofile(str(out_path))
        print(f"  → Shard {shard_idx}: {shard_size:,} tokens ({out_path.stat().st_size / 1e6:.1f} MB)")
        
        # Keep remainder
        remainder = [all_tokens[shard_size:]] if len(all_tokens) > shard_size else []
        shard_buf[:] = remainder
        return shard_idx + 1
    
    shard_buf[:] = [all_tokens]
    return shard_idx

def main():
    print("=" * 60)
    print("TOKENIZE FILTERED DATA → 3.3B TOKENS")
    print("=" * 60)
    
    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, trust_remote_code=True)
    print(f"  Vocab size: {len(tokenizer)}")
    
    # State
    shard_buf = []
    shard_idx = 0
    total_tokens = 0
    total_docs = 0
    
    # Tokenize in priority order
    for ds in DATASET_PRIORITY:
        ds_dir = FILTERED_DIR / ds
        if not ds_dir.exists():
            print(f"\n[SKIP] {ds}: directory not found")
            continue
        
        jsonl_files = sorted(ds_dir.glob("*.jsonl"))
        if not jsonl_files:
            print(f"\n[SKIP] {ds}: no JSONL files")
            continue
        
        remaining = token_target - total_tokens
        if remaining <= 0:
            print(f"\n[STOP] Target reached. Skipping {ds}.")
            break
        
        print(f"\n[{ds.upper()}] {len(jsonl_files)} files | Target: {remaining:,} more tokens")
        
        ds_tokens, ds_docs, shard_buf, shard_idx = tokenize_dataset(
            jsonl_files, tokenizer, remaining, shard_buf, shard_idx
        )
        total_tokens += ds_tokens
        total_docs += ds_docs
        
        print(f"  Added: {ds_tokens:,} tokens | {ds_docs:,} docs")
        print(f"  Running total: {total_tokens:,} / {token_target:,} ({100*total_tokens/token_target:.1f}%)")
    
    # Flush final partial shard
    if shard_buf and len(shard_buf[0]) > 0:
        final = np.concatenate(shard_buf)
        out_path = OUTPUT_DIR / f"shard_{shard_idx:06d}.bin"
        final.tofile(str(out_path))
        print(f"\n  → Final shard {shard_idx}: {len(final):,} tokens")
        shard_idx += 1
        total_tokens += len(final)
    
    # Summary
    print("\n" + "=" * 60)
    print("TOKENIZATION COMPLETE")
    print("=" * 60)
    print(f"Total shards: {shard_idx}")
    print(f"Total tokens: {total_tokens:,}")
    print(f"Total docs: {total_docs:,}")
    print(f"Avg tokens/doc: {total_tokens//max(total_docs,1):,}")
    print(f"Output: {OUTPUT_DIR}")
    
    # List shards
    shards = sorted(OUTPUT_DIR.glob("shard_*.bin"))
    print(f"\nShard files:")
    for s in shards:
        size_mb = s.stat().st_size / 1e6
        tokens = int(size_mb * 1e6 / 2)  # uint16 = 2 bytes
        print(f"  {s.name}: {size_mb:.1f} MB (~{tokens:,} tokens)")

if __name__ == "__main__":
    main()
