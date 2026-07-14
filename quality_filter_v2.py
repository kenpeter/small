"""
Quality filter v2: Gopher + FineWeb + C4 with realistic thresholds.
Relaxes overly strict filters that reject all web text.
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

TARGET_TOKENS = 1_600_000_000
SEQ_LEN = 2048

STOP_WORDS = {
    'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'i',
    'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at',
    'this', 'but', 'his', 'by', 'from', 'they', 'we', 'say', 'her', 'she',
    'or', 'an', 'will', 'my', 'one', 'all', 'would', 'there', 'their'
}

BAD_WORDS = {'http', 'https', 'www', 'com', 'html'}


def gopher_filter(text):
    """Gopher filter with relaxed thresholds."""
    words = text.split()
    if len(words) == 0:
        return False, "empty"
    
    word_count = len(words)
    if word_count < 50:  # Must have some substance
        return False, "too_short"
    if word_count > 100000:  # Not a giant document
        return False, "too_long"
    
    # Avg word length (3-10 chars) - reasonable for English
    avg_word_len = sum(len(w) for w in words) / word_count
    if avg_word_len < 2.5 or avg_word_len > 12:
        return False, "avg_word_len"
    
    # Symbol words - RELAXED from 10% to 30%
    symbol_words = sum(1 for w in words if any(c in string.punctuation for c in w))
    if symbol_words / word_count > 0.30:
        return False, "too_many_symbols"
    
    # Bullet lines - RELAXED from 10% to 70%
    lines = text.split('\n')
    bullet_lines = sum(1 for l in lines if l.strip().startswith(('-', '*', '•', '·')))
    if len(lines) > 0 and bullet_lines / len(lines) > 0.70:
        return False, "too_many_bullets"
    
    # Ellipsis lines - RELAXED from 30% to 40%
    ellipsis_lines = sum(1 for l in lines if l.strip().endswith('...'))
    if len(lines) > 0 and ellipsis_lines / len(lines) > 0.40:
        return False, "too_many_ellipsis"
    
    # Alpha words - RELAXED from 20% to 10%
    alpha_words = sum(1 for w in words if any(c.isalpha() for c in w))
    if alpha_words / word_count < 0.10:
        return False, "too_few_alpha_words"
    
    # Stop words - at least 2 unique stop words
    unique_stops = len(set(w.lower() for w in words) & STOP_WORDS)
    if unique_stops < 2:
        return False, "too_few_stop_words"
    
    return True, "pass"


def fineweb_filter(text):
    """FineWeb filter with realistic thresholds."""
    lines = text.split('\n')
    if len(lines) == 0:
        return False, "empty"
    
    # Line ending with punctuation (≥8%)
    punct_ending = sum(1 for l in lines if l.strip() and l.strip()[-1] in '.!?;')
    if len(lines) > 0 and punct_ending / len(lines) < 0.08:
        return False, "low_punct_ending"
    
    # Short lines (≤80%)
    short_lines = sum(1 for l in lines if len(l) < 30)
    if len(lines) > 0 and short_lines / len(lines) > 0.80:
        return False, "too_many_short_lines"
    
    # Character repetition - exclude space/newline, check others <12%
    if len(text) > 0:
        char_counts = Counter(text)
        # Exclude whitespace from dup check
        filtered_counts = {c: n for c, n in char_counts.items() if c not in ' \n\t\r'}
        if filtered_counts:
            max_dup = max(filtered_counts.values()) / len(text)
            if max_dup > 0.12:
                return False, "char_repetition"
    
    # Newline ratio (≤40%)
    newline_count = text.count('\n')
    if len(text) > 0 and newline_count / len(text) > 0.40:
        return False, "too_many_newlines"
    
    return True, "pass"


def c4_filter(text):
    """C4 filter - simplified."""
    words = text.lower().split()
    lines = text.split('\n')
    
    # Bad words check
    if any(bw in words for bw in BAD_WORDS):
        return False, "bad_words"
    
    # At least 2 lines
    if len(lines) < 2:
        return False, "too_few_lines"
    
    # Not all short paragraphs
    short_paras = sum(1 for l in lines if len(l.split()) < 5)
    if len(lines) > 0 and short_paras / len(lines) > 0.95:
        return False, "too_many_short_paras"
    
    return True, "pass"


def apply_all_filters(text):
    """Apply all 3 filters."""
    reasons = []
    
    good, reason = gopher_filter(text)
    if not good:
        reasons.append(f"gopher:{reason}")
        return False, reasons
    
    good, reason = fineweb_filter(text)
    if not good:
        reasons.append(f"fineweb:{reason}")
        return False, reasons
    
    good, reason = c4_filter(text)
    if not good:
        reasons.append(f"c4:{reason}")
        return False, reasons
    
    return True, ["all_pass"]


def filter_shard(shard_path, tokenizer, stats, max_seqs=None):
    """Filter a single shard."""
    tokens = np.memmap(str(shard_path), dtype=np.uint16, mode='r')
    total_tokens = len(tokens)
    num_seqs = total_tokens // SEQ_LEN
    
    if max_seqs:
        num_seqs = min(num_seqs, max_seqs)
    
    kept_tokens = []
    rejected_counts = Counter()
    
    for i in range(num_seqs):
        start = i * SEQ_LEN
        seq_tokens = tokens[start:start + SEQ_LEN]
        
        try:
            text = tokenizer.decode(seq_tokens, skip_special_tokens=True)
        except:
            rejected_counts["decode_error"] += 1
            continue
        
        is_good, reasons = apply_all_filters(text)
        
        if is_good:
            kept_tokens.extend(seq_tokens.tolist())
            stats["kept"] += 1
        else:
            for r in reasons:
                rejected_counts[r] += 1
            stats["rejected"] += 1
        
        stats["total"] += 1
        
        if (i + 1) % 5000 == 0:
            keep_rate = stats["kept"] / max(stats["total"], 1)
            print(f"  {i+1}/{num_seqs}: kept {stats['kept']} ({keep_rate*100:.1f}%), rejected {stats['rejected']}", flush=True)
    
    del tokens
    gc.collect()
    
    return np.array(kept_tokens, dtype=np.uint16), rejected_counts


def main():
    print("=" * 60, flush=True)
    print("Quality Filter v2: Relaxed Thresholds", flush=True)
    print("=" * 60, flush=True)
    
    print("\nLoading tokenizer...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")
    
    shards = sorted(DATA_DIR.glob("shard_*.bin"))
    print(f"\nFound {len(shards)} shards", flush=True)
    
    stats = {"total": 0, "kept": 0, "rejected": 0}
    all_rejected = Counter()
    
    total_kept_tokens = 0
    output_shard_idx = 0
    current_buffer = []
    
    for shard_path in tqdm(shards, desc="Filtering shards"):
        print(f"\nProcessing {shard_path.name}...")
        
        filtered_tokens, rejected_counts = filter_shard(shard_path, tokenizer, stats)
        all_rejected.update(rejected_counts)
        
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
        
        if total_kept_tokens >= TARGET_TOKENS:
            print(f"\nReached target of {TARGET_TOKENS:,} tokens!")
            break
    
    # Save remaining
    if current_buffer:
        output_path = OUTPUT_DIR / f"shard_{output_shard_idx:06d}.bin"
        to_save = np.array(current_buffer, dtype=np.uint16)
        to_save.tofile(output_path)
        print(f"  Saved {output_path.name}: {len(to_save):,} tokens")
    
    # Stats
    print("\n" + "=" * 60)
    print("FILTERING COMPLETE")
    print("=" * 60)
    print(f"\nTotal sequences: {stats['total']:,}")
    print(f"Kept: {stats['kept']:,} ({100*stats['kept']/max(stats['total'],1):.1f}%)")
    print(f"Rejected: {stats['rejected']:,} ({100*stats['rejected']/max(stats['total'],1):.1f}%)")
    print(f"\nTotal tokens kept: {total_kept_tokens:,}")
    print(f"\nTop rejection reasons:")
    for reason, count in all_rejected.most_common(10):
        print(f"  {reason}: {count:,}")
    print(f"\nOutput: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
