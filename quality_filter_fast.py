"""
Fast quality filter - process just enough shards for 1.6B tokens.
Combines Gopher + FineWeb + C4 with realistic thresholds.
"""
import os, re, json, gc, string
from pathlib import Path
from collections import Counter
import numpy as np
from transformers import AutoTokenizer

DATA_DIR = Path("/home/kenpeter/work/data")
OUTPUT_DIR = DATA_DIR / "filtered_best_1.6b"
OUTPUT_DIR.mkdir(exist_ok=True)

TARGET_TOKENS = 1_600_000_000
SEQ_LEN = 2048

STOP_WORDS = {'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'i', 'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at'}
BAD_WORDS = {'http', 'https', 'www', 'com', 'html'}


def gopher_filter(text):
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


def filter_shard(shard_path, tokenizer):
    """Filter a single shard, return kept tokens."""
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
        
        if (i + 1) % 10000 == 0:
            print(f"  {i+1}/{num_seqs}: kept {kept} ({100*kept/(kept+rejected):.1f}%), rejected {rejected}", flush=True)
    
    del tokens
    gc.collect()
    
    return np.array(kept_tokens, dtype=np.uint16), kept, rejected


def main():
    print("Quality Filter: Fast Mode (2 shards = ~1.6B tokens)", flush=True)
    
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")
    shards = sorted(DATA_DIR.glob("shard_*.bin"))[:2]  # Only first 2 shards
    print(f"Processing {len(shards)} shards to get ~1.6B tokens", flush=True)
    
    total_kept = 0
    total_rejected = 0
    all_kept_tokens = []
    output_idx = 0
    
    for shard_path in shards:
        print(f"\nProcessing {shard_path.name}...", flush=True)
        filtered, kept, rejected = filter_shard(shard_path, tokenizer)
        
        total_kept += kept
        total_rejected += rejected
        all_kept_tokens.extend(filtered.tolist())
        
        print(f"  Kept {len(filtered):,} tokens from this shard", flush=True)
        
        # Save in 1B chunks
        while len(all_kept_tokens) >= 1_073_741_824:
            output_path = OUTPUT_DIR / f"shard_{output_idx:06d}.bin"
            to_save = np.array(all_kept_tokens[:1_073_741_824], dtype=np.uint16)
            to_save.tofile(output_path)
            print(f"  Saved {output_path.name}: {len(to_save):,} tokens", flush=True)
            all_kept_tokens = all_kept_tokens[1_073_741_824:]
            output_idx += 1
        
        if len(all_kept_tokens) >= TARGET_TOKENS:
            break
    
    # Save remaining
    if all_kept_tokens:
        output_path = OUTPUT_DIR / f"shard_{output_idx:06d}.bin"
        to_save = np.array(all_kept_tokens, dtype=np.uint16)
        to_save.tofile(output_path)
        print(f"  Saved {output_path.name}: {len(to_save):,} tokens", flush=True)
    
    total_tokens = total_kept * SEQ_LEN
    print(f"\n{'='*60}", flush=True)
    print(f"DONE! Kept {total_kept:,} sequences ({100*total_kept/(total_kept+total_rejected):.1f}%)", flush=True)
    print(f"Total tokens: {total_tokens:,}", flush=True)
    print(f"Output: {OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
