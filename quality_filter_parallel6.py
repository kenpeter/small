"""
6-worker parallel quality filter.
Gopher + FineWeb + C4 filters with realistic thresholds.
Target: 15-20 min for 1.6B tokens.
"""
import os, gc, string
from pathlib import Path
from collections import Counter
import numpy as np
from multiprocessing import Pool, cpu_count
from transformers import AutoTokenizer

DATA_DIR = Path("/home/kenpeter/work/data")
OUTPUT_DIR = DATA_DIR / "filtered_best_1.6b"
OUTPUT_DIR.mkdir(exist_ok=True)

TARGET_TOKENS = 1_600_000_000
SEQ_LEN = 2048
N_WORKERS = 6  # One per physical core

STOP_WORDS = {'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'i', 
              'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at'}
BAD_WORDS = {'http', 'https', 'www', 'com', 'html'}


def gopher_filter(text):
    """Gopher quality filter - relaxed for web text."""
    words = text.split()
    if len(words) == 0:
        return False
    word_count = len(words)
    if word_count < 50 or word_count > 100000:
        return False
    avg_word_len = sum(len(w) for w in words) / word_count
    if avg_word_len < 2.5 or avg_word_len > 12:
        return False
    symbol_words = sum(1 for w in words if any(c in string.punctuation for c in w))
    if symbol_words / word_count > 0.30:
        return False
    lines = text.split('\n')
    bullet_lines = sum(1 for l in lines if l.strip().startswith(('-', '*', '•', '·')))
    if len(lines) > 0 and bullet_lines / len(lines) > 0.70:
        return False
    ellipsis_lines = sum(1 for l in lines if l.strip().endswith('...'))
    if len(lines) > 0 and ellipsis_lines / len(lines) > 0.40:
        return False
    alpha_words = sum(1 for w in words if any(c.isalpha() for c in w))
    if alpha_words / word_count < 0.10:
        return False
    unique_stops = len(set(w.lower() for w in words) & STOP_WORDS)
    if unique_stops < 2:
        return False
    return True


def fineweb_filter(text):
    """FineWeb quality filter."""
    lines = text.split('\n')
    if len(lines) == 0:
        return False
    punct_ending = sum(1 for l in lines if l.strip() and l.strip()[-1] in '.!?;')
    if len(lines) > 0 and punct_ending / len(lines) < 0.08:
        return False
    short_lines = sum(1 for l in lines if len(l) < 30)
    if len(lines) > 0 and short_lines / len(lines) > 0.80:
        return False
    if len(text) > 0:
        char_counts = Counter(text)
        filtered_counts = {c: n for c, n in char_counts.items() if c not in ' \n\t\r'}
        if filtered_counts:
            max_dup = max(filtered_counts.values()) / len(text)
            if max_dup > 0.12:
                return False
    newline_count = text.count('\n')
    if len(text) > 0 and newline_count / len(text) > 0.40:
        return False
    return True


def c4_filter(text):
    """C4 quality filter."""
    words = text.lower().split()
    lines = text.split('\n')
    if any(bw in words for bw in BAD_WORDS):
        return False
    if len(lines) < 2:
        return False
    short_paras = sum(1 for l in lines if len(l.split()) < 5)
    if len(lines) > 0 and short_paras / len(lines) > 0.95:
        return False
    return True


def filter_shard_worker(args):
    """Worker function - processes one shard."""
    shard_path, shard_idx = args
    
    # Each worker loads its own tokenizer
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")
    
    tokens = np.memmap(str(shard_path), dtype=np.uint16, mode='r')
    num_seqs = len(tokens) // SEQ_LEN
    
    kept_tokens = []
    kept = rejected = 0
    
    for i in range(num_seqs):
        start = i * SEQ_LEN
        seq_tokens = tokens[start:start + SEQ_LEN]
        
        try:
            text = tokenizer.decode(seq_tokens, skip_special_tokens=True)
        except:
            rejected += 1
            continue
        
        if gopher_filter(text) and fineweb_filter(text) and c4_filter(text):
            kept_tokens.extend(seq_tokens.tolist())
            kept += 1
        else:
            rejected += 1
        
        if (i + 1) % 50000 == 0:
            print(f"  Shard {shard_idx}: {i+1}/{num_seqs} ({100*(i+1)/num_seqs:.1f}%)", flush=True)
    
    del tokens
    gc.collect()
    
    # Save to temp file
    temp_path = OUTPUT_DIR / f"_temp_shard_{shard_idx:06d}.bin"
    if kept_tokens:
        np.array(kept_tokens, dtype=np.uint16).tofile(temp_path)
    
    return {
        'shard': shard_idx,
        'temp_file': str(temp_path),
        'kept': kept,
        'rejected': rejected,
        'tokens': len(kept_tokens)
    }


def main():
    print(f"6-Worker Parallel Quality Filter")
    print(f"=================================")
    print(f"Workers: {N_WORKERS} (one per physical core)")
    print(f"Target: {TARGET_TOKENS:,} tokens")
    print()
    
    # Find all shards
    shards = sorted(DATA_DIR.glob("shard_*.bin"))
    print(f"Found {len(shards)} shards total")
    
    # We need ~2 shards for 1.6B tokens (at 94% keep rate)
    # But process in parallel for speed
    shards_to_process = shards[:3]  # Process 3 shards, keep ~1.6B
    print(f"Processing first {len(shards_to_process)} shards in parallel")
    print()
    
    # Process with 6 workers
    print(f"Starting {N_WORKERS} workers...")
    with Pool(N_WORKERS) as pool:
        results = pool.map(filter_shard_worker, [(s, i) for i, s in enumerate(shards_to_process)])
    
    # Combine results
    print(f"\nCombining results...")
    total_kept = sum(r['kept'] for r in results)
    total_rejected = sum(r['rejected'] for r in results)
    total_tokens = sum(r['tokens'] for r in results)
    
    # Merge temp files into final shards (1B tokens each)
    output_idx = 0
    buffer = []
    
    for r in results:
        temp_path = Path(r['temp_file'])
        if temp_path.exists():
            data = np.fromfile(temp_path, dtype=np.uint16)
            buffer.extend(data.tolist())
            temp_path.unlink()  # Delete temp file
            
            while len(buffer) >= 1_073_741_824:
                out_path = OUTPUT_DIR / f"shard_{output_idx:06d}.bin"
                np.array(buffer[:1_073_741_824], dtype=np.uint16).tofile(out_path)
                print(f"  Saved {out_path.name}")
                buffer = buffer[1_073_741_824:]
                output_idx += 1
    
    # Save remaining
    if buffer:
        out_path = OUTPUT_DIR / f"shard_{output_idx:06d}.bin"
        np.array(buffer, dtype=np.uint16).tofile(out_path)
        print(f"  Saved {out_path.name}")
    
    # Print stats
    print(f"\n{'='*60}")
    print(f"DONE!")
    print(f"{'='*60}")
    print(f"Sequences kept: {total_kept:,} ({100*total_kept/(total_kept+total_rejected):.1f}%)")
    print(f"Sequences rejected: {total_rejected:,}")
    print(f"Total tokens: {total_tokens:,}")
    print(f"Output directory: {OUTPUT_DIR}")
    
    # List output files
    output_files = sorted(OUTPUT_DIR.glob("shard_*.bin"))
    print(f"\nOutput files ({len(output_files)}):")
    for f in output_files:
        size = f.stat().st_size
        print(f"  {f.name}: {size/1e9:.2f} GB")


if __name__ == "__main__":
    main()
