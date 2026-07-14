"""
Sample 1.6B tokens from existing 49 shards.
Creates new filtered shards without deleting originals.
"""
import os, json, gc
from pathlib import Path
import numpy as np

DATA_DIR = Path("/home/kenpeter/work/data")
SAMPLE_DIR = DATA_DIR / "sampled_1.6b"
SAMPLE_DIR.mkdir(exist_ok=True)

TARGET_TOKENS = int(1.6e9)  # 1.6 billion tokens
SHARD_SIZE = int(1_073_741_824)  # 1B tokens per output shard
SEQ_LEN = 2048  # sequences of 2048 tokens

def sample_shard(shard_path, sample_ratio):
    """Sample sequences from a shard."""
    tokens = np.memmap(str(shard_path), dtype=np.uint16, mode='r')
    total_tokens = len(tokens)
    
    # Calculate how many sequences to keep
    num_seqs = total_tokens // SEQ_LEN
    keep_seqs = int(num_seqs * sample_ratio)
    
    if keep_seqs == 0:
        return np.array([], dtype=np.uint16)
    
    # Random sample
    np.random.seed(42)  # reproducible
    keep_indices = np.random.choice(num_seqs, keep_seqs, replace=False)
    keep_indices.sort()
    
    # Extract sequences
    sampled = []
    for idx in keep_indices:
        start = idx * SEQ_LEN
        sampled.extend(tokens[start:start + SEQ_LEN])
    
    del tokens
    gc.collect()
    
    return np.array(sampled, dtype=np.uint16)

def main():
    # Find all existing shards
    shards = sorted(DATA_DIR.glob("shard_*.bin"))
    print(f"Found {len(shards)} existing shards")
    
    # Calculate total tokens
    total_tokens = sum(Path(s).stat().st_size // 2 for s in shards)
    print(f"Total tokens available: {total_tokens:,}")
    
    # Calculate sample ratio
    sample_ratio = TARGET_TOKENS / total_tokens
    print(f"Sample ratio: {sample_ratio:.2%}")
    
    # Sample from each shard
    buffer = np.empty(SHARD_SIZE, dtype=np.uint16)
    buf_pos = 0
    shard_idx = 0
    total_sampled = 0
    
    for i, shard_path in enumerate(shards):
        print(f"\n[{i+1}/{len(shards)}] Sampling from {shard_path.name}...")
        
        sampled = sample_shard(shard_path, sample_ratio)
        n = len(sampled)
        
        if n == 0:
            continue
        
        # Add to buffer
        start = 0
        while start < n:
            rem = SHARD_SIZE - buf_pos
            chunk = sampled[start:start + rem]
            buffer[buf_pos:buf_pos + len(chunk)] = chunk
            buf_pos += len(chunk)
            start += len(chunk)
            
            if buf_pos >= SHARD_SIZE:
                # Write shard
                sp = SAMPLE_DIR / f"shard_{shard_idx:06d}.bin"
                buffer.tofile(sp)
                print(f"  Written {sp.name}")
                shard_idx += 1
                buf_pos = 0
        
        total_sampled += n
        print(f"  Sampled {n:,} tokens (total: {total_sampled:,})")
        
        if total_sampled >= TARGET_TOKENS:
            print(f"\nReached target of {TARGET_TOKENS:,} tokens!")
            break
    
    # Flush remaining
    if buf_pos > 0:
        sp = SAMPLE_DIR / f"shard_{shard_idx:06d}.bin"
        buffer[:buf_pos].tofile(sp)
        print(f"  Written {sp.name} ({buf_pos:,} tokens)")
        shard_idx += 1
    
    print(f"\n✅ Done!")
    print(f"   Sampled tokens: {total_sampled:,}")
    print(f"   Output shards: {shard_idx}")
    print(f"   Location: {SAMPLE_DIR}")
    
    # Save metadata
    meta = {
        "source_shards": len(shards),
        "sampled_tokens": int(total_sampled),
        "output_shards": shard_idx,
        "target_tokens": TARGET_TOKENS,
        "sample_ratio": sample_ratio
    }
    with open(SAMPLE_DIR / "sample_info.json", "w") as f:
        json.dump(meta, f, indent=2)

if __name__ == "__main__":
    main()
