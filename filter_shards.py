"""
Filter existing shards by quality metrics.
Keep top ~75% of data by perplexity/quality score.
"""
import os, json, gc
from pathlib import Path
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

DATA_DIR = Path("/home/kenpeter/work/data")
CHECKPOINT_DIR = Path("/home/kenpeter/work/checkpoints")
FILTERED_DIR = DATA_DIR / "filtered"
FILTERED_DIR.mkdir(exist_ok=True)

# Config
TARGET_TOKENS = int(1.5e9)  # 1.5B tokens from existing data
SHARD_SIZE = int(1_073_741_824)  # 1B tokens per shard
KEEP_RATIO = 0.75  # Keep top 75% by quality

def score_sequence(tokens, model, tokenizer, device):
    """Score a sequence by perplexity (lower = better)."""
    if len(tokens) < 100:
        return float('inf')
    
    # Convert to tensor
    input_ids = torch.tensor([tokens[:2048]], device=device)  # Cap at 2048
    
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss.item()
    
    return loss  # Lower loss = higher quality

def filter_shard(shard_path, model, tokenizer, device, keep_ratio=0.75):
    """Filter a single shard, keeping top sequences by quality."""
    print(f"  Loading {shard_path.name}...")
    
    # Load shard
    tokens = np.memmap(str(shard_path), dtype=np.uint16, mode='r')
    total_tokens = len(tokens)
    
    # Split into sequences (8192 tokens each for scoring)
    seq_len = 8192
    num_seqs = total_tokens // seq_len
    
    print(f"    {num_seqs} sequences to score...")
    
    scores = []
    for i in range(min(num_seqs, 100)):  # Sample first 100 sequences for speed
        start = i * seq_len
        seq = tokens[start:start + seq_len].tolist()
        score = score_sequence(seq, model, tokenizer, device)
        scores.append((i, score))
        
        if (i + 1) % 20 == 0:
            print(f"    Scored {i+1}/{min(num_seqs, 100)}...")
    
    # Sort by score (lower is better)
    scores.sort(key=lambda x: x[1])
    
    # Keep top sequences
    keep_count = int(len(scores) * keep_ratio)
    keep_indices = set(idx for idx, _ in scores[:keep_count])
    
    print(f"    Keeping {keep_count}/{len(scores)} sequences")
    
    # Build filtered tokens
    filtered_tokens = []
    for i in range(num_seqs):
        if i in keep_indices:
            start = i * seq_len
            filtered_tokens.extend(tokens[start:start + seq_len])
    
    # Save filtered shard
    output_path = FILTERED_DIR / f"{shard_path.stem}_filtered.bin"
    filtered_array = np.array(filtered_tokens, dtype=np.uint16)
    filtered_array.tofile(output_path)
    
    print(f"    Saved {output_path.name}: {len(filtered_tokens):,} tokens")
    
    del tokens, filtered_array
    gc.collect()
    
    return len(filtered_tokens)

def main():
    # Load model for scoring
    print("Loading model for quality scoring...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load from checkpoint if available
    ckpt_path = CHECKPOINT_DIR / "checkpoint_best.pt"
    if ckpt_path.exists():
        print(f"  Loading from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        
        # Import model class
        import sys
        sys.path.insert(0, str(Path(__file__).parent))
        from train import ModelConfig, Upfar
        
        cfg = ModelConfig()
        model = Upfar(cfg).to(device)
        model.load_state_dict(ckpt['model'])
        model.eval()
        print(f"  Loaded trained model (step {ckpt.get('step', 'unknown')})")
    else:
        print("  No checkpoint found, using random init model")
        tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M", trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-135M", trust_remote_code=True).to(device)
        model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M", trust_remote_code=True)
    
    # Find all shards
    shards = sorted(DATA_DIR.glob("shard_*.bin"))
    print(f"\nFound {len(shards)} shards to filter")
    
    total_kept = 0
    for i, shard in enumerate(shards):
        print(f"\n[{i+1}/{len(shards)}] Processing {shard.name}...")
        kept = filter_shard(shard, model, tokenizer, device, KEEP_RATIO)
        total_kept += kept
        
        if total_kept >= TARGET_TOKENS:
            print(f"\nReached target of {TARGET_TOKENS:,} tokens. Stopping.")
            break
    
    print(f"\n✅ Done! Kept {total_kept:,} tokens from {i+1} shards")
    print(f"   Filtered data in: {FILTERED_DIR}")

if __name__ == "__main__":
    main()
