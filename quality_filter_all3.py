"""
Fast quality filter combining Gopher + FineWeb-Edu + C4 algorithms.
Decodes tokens → applies heuristics → keeps best sequences.
Target: 10-15 min for 49 shards → 1.6B high-quality tokens.
"""
import os, re, json, gc, string
from pathlib import Path
from collections import Counter
import numpy as np
from transformers import AutoTokenizer
from tqdm import tqdm

DATA_DIR = Path("/home/kenpeter/work/data")
OUTPUT_DIR = DATA_DIR / "filtered_best_1.6b"
OUTPUT_DIR.mkdir(exist_ok=True)

# Target: 1.6B tokens
TARGET_TOKENS = 1_600_000_000
SEQ_LEN = 2048  # sequences of 2048 tokens

# Stop words for Gopher filter
STOP_WORDS = {
    'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'i',
    'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at',
    'this', 'but', 'his', 'by', 'from', 'they', 'we', 'say', 'her', 'she',
    'or', 'an', 'will', 'my', 'one', 'all', 'would', 'there', 'their'
}

# Bad words for C4 filter (minimal set)
BAD_WORDS = {'http', 'https', 'www', 'com', 'html', 'wp', 'img', 'src'}


def gopher_filter(text):
    """
    DeepMind Gopher quality filter.
    Returns (is_good, reason)
    """
    words = text.split()
    if len(words) == 0:
        return False, "empty"
    
    # Word count check (50-100,000 words)
    word_count = len(words)
    if word_count < 50:
        return False, "too_short"
    if word_count > 100000:
        return False, "too_long"
    
    # Average word length (3-10 chars)
    avg_word_len = sum(len(w) for w in words) / word_count
    if avg_word_len < 3 or avg_word_len > 10:
        return False, "avg_word_len"
    
    # Symbol words check (<10%)
    symbol_words = sum(1 for w in words if any(c in string.punctuation for c in w))
    if symbol_words / word_count > 0.10:
        return False, "too_many_symbols"
    
    # Bullet line check (<10% lines start with bullet)
    lines = text.split('\n')
    bullet_lines = sum(1 for l in lines if l.strip().startswith(('-', '*', '•', '·')))
    if len(lines) > 0 and bullet_lines / len(lines) > 0.10:
        return False, "too_many_bullets"
    
    # Ellipsis line check (<30% lines end with ...)
    ellipsis_lines = sum(1 for l in lines if l.strip().endswith('...'))
    if len(lines) > 0 and ellipsis_lines / len(lines) > 0.30:
        return False, "too_many_ellipsis"
    
    # Non-alpha words check (<80%)
    alpha_words = sum(1 for w in words if any(c.isalpha() for c in w))
    if alpha_words / word_count < 0.20:  # Less than 20% alpha = more than 80% non-alpha
        return False, "too_few_alpha_words"
    
    # Stop words check (≥2 unique stop words)
    unique_stops = len(set(w.lower() for w in words) & STOP_WORDS)
    if unique_stops < 2:
        return False, "too_few_stop_words"
    
    return True, "pass"


def fineweb_filter(text):
    """
    FineWeb-Edu quality filter.
    Returns (is_good, reason)
    """
    lines = text.split('\n')
    if len(lines) == 0:
        return False, "empty"
    
    # Line ending with punctuation (≥12%)
    punct_ending = sum(1 for l in lines if l.strip() and l.strip()[-1] in '.!?;')
    if len(lines) > 0 and punct_ending / len(lines) < 0.12:
        return False, "low_punct_ending"
    
    # Short lines check (<67% lines <30 chars)
    short_lines = sum(1 for l in lines if len(l) < 30)
    if len(lines) > 0 and short_lines / len(lines) > 0.67:
        return False, "too_many_short_lines"
    
    # Character repetition check (<1% duplicate chars)
    char_counts = Counter(text)
    total_chars = len(text)
    if total_chars > 0:
        max_dup_ratio = max(char_counts.values()) / total_chars
        if max_dup_ratio > 0.01:
            return False, "char_repetition"
    
    # Newline ratio check (<30%)
    newline_count = text.count('\n')
    if total_chars > 0 and newline_count / total_chars > 0.30:
        return False, "too_many_newlines"
    
    return True, "pass"


def c4_filter(text):
    """
    C4 quality filter (simplified).
    Returns (is_good, reason)
    """
    words = text.lower().split()
    
    # Bad words check
    if any(bw in words for bw in BAD_WORDS):
        return False, "bad_words"
    
    # Line count check (at least 3 lines)
    lines = text.split('\n')
    if len(lines) < 3:
        return False, "too_few_lines"
    
    # Short paragraph check
    short_paras = sum(1 for l in lines if len(l.split()) < 5)
    if len(lines) > 0 and short_paras / len(lines) > 0.90:
        return False, "too_many_short_paras"
    
    return True, "pass"


def apply_all_filters(text):
    """Apply all 3 filters. Returns (is_good, reasons)."""
    reasons = []
    
    # Gopher
    good, reason = gopher_filter(text)
    if not good:
        reasons.append(f"gopher:{reason}")
        return False, reasons
    
    # FineWeb
    good, reason = fineweb_filter(text)
    if not good:
        reasons.append(f"fineweb:{reason}")
        return False, reasons
    
    # C4
    good, reason = c4_filter(text)
    if not good:
        reasons.append(f"c4:{reason}")
        return False, reasons
    
    return True, ["all_pass"]


def filter_shard(shard_path, tokenizer, stats):
    """Filter a single shard, keeping sequences that pass all filters."""
    tokens = np.memmap(str(shard_path), dtype=np.uint16, mode='r')
    total_tokens = len(tokens)
    num_seqs = total_tokens // SEQ_LEN
    
    kept_tokens = []
    rejected_counts = Counter()
    
    for i in range(num_seqs):
        start = i * SEQ_LEN
        seq_tokens = tokens[start:start + SEQ_LEN]
        
        # Decode to text
        try:
            text = tokenizer.decode(seq_tokens, skip_special_tokens=True)
        except:
            rejected_counts["decode_error"] += 1
            continue
        
        # Apply all filters
        is_good, reasons = apply_all_filters(text)
        
        if is_good:
            kept_tokens.extend(seq_tokens.tolist())
            stats["kept"] += 1
        else:
            for r in reasons:
                rejected_counts[r] += 1
            stats["rejected"] += 1
        
        stats["total"] += 1
        
        if (i + 1) % 1000 == 0:
            print(f"  Processed {i+1}/{num_seqs} seqs, kept {stats['kept']}, rejected {stats['rejected']}")
    
    del tokens
    gc.collect()
    
    return np.array(kept_tokens, dtype=np.uint16), rejected_counts


def main():
    print("=" * 60)
    print("Quality Filter: Gopher + FineWeb + C4")
    print("=" * 60)
    
    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")
    
    # Find all shards
    shards = sorted(DATA_DIR.glob("shard_*.bin"))
    print(f"\nFound {len(shards)} shards")
    
    # Stats
    stats = {"total": 0, "kept": 0, "rejected": 0}
    all_rejected = Counter()
    
    total_kept_tokens = 0
    output_shard_idx = 0
    current_buffer = []
    
    for shard_path in tqdm(shards, desc="Filtering shards"):
        print(f"\nProcessing {shard_path.name}...")
        
        # Filter shard
        filtered_tokens, rejected_counts = filter_shard(shard_path, tokenizer, stats)
        all_rejected.update(rejected_counts)
        
        # Add to buffer
        current_buffer.extend(filtered_tokens.tolist())
        total_kept_tokens += len(filtered_tokens)
        
        # Save when buffer reaches 1B tokens
        while len(current_buffer) >= 1_073_741_824:
            output_path = OUTPUT_DIR / f"shard_{output_shard_idx:06d}.bin"
            to_save = np.array(current_buffer[:1_073_741_824], dtype=np.uint16)
            to_save.tofile(output_path)
            print(f"  Saved {output_path.name}: {len(to_save):,} tokens")
            current_buffer = current_buffer[1_073_741_824:]
            output_shard_idx += 1
        
        # Stop if we have enough tokens
        if total_kept_tokens >= TARGET_TOKENS:
            print(f"\nReached target of {TARGET_TOKENS:,} tokens!")
            break
    
    # Save remaining buffer
    if current_buffer:
        output_path = OUTPUT_DIR / f"shard_{output_shard_idx:06d}.bin"
        to_save = np.array(current_buffer, dtype=np.uint16)
        to_save.tofile(output_path)
        print(f"  Saved {output_path.name}: {len(to_save):,} tokens")
    
    # Print stats
    print("\n" + "=" * 60)
    print("FILTERING COMPLETE")
    print("=" * 60)
    print(f"\nTotal sequences: {stats['total']:,}")
    print(f"Kept: {stats['kept']:,} ({100*stats['kept']/max(stats['total'],1):.1f}%)")
    print(f"Rejected: {stats['rejected']:,} ({100*stats['rejected']/max(stats['total'],1):.1f}%)")
    print(f"\nTotal tokens kept: {total_kept_tokens:,}")
    print(f"\nRejection reasons:")
    for reason, count in all_rejected.most_common(10):
        print(f"  {reason}: {count:,}")
    
    print(f"\nOutput directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
