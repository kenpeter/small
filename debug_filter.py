"""Debug the quality filters."""
import numpy as np
import string
from collections import Counter
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")
tokens = np.memmap("/home/kenpeter/work/data/shard_000000.bin", dtype=np.uint16, mode='r')

STOP_WORDS = {'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'i'}

# Test first 5 sequences
for seq_idx in range(5):
    start = seq_idx * 2048
    seq = tokens[start:start+2048]
    text = tokenizer.decode(seq, skip_special_tokens=True)
    words = text.split()
    lines = text.split('\n')
    
    print(f"\n=== Sequence {seq_idx} ===")
    print(f"Words: {len(words)}, Lines: {len(lines)}")
    
    # Gopher checks
    if len(words) < 50:
        print("  FAIL: too_short")
        continue
    
    avg_word_len = sum(len(w) for w in words) / len(words)
    print(f"  Avg word len: {avg_word_len:.2f}", end="")
    if avg_word_len < 2.5 or avg_word_len > 12:  # RELAXED
        print(" FAIL")
        continue
    else:
        print(" PASS")
    
    symbol_words = sum(1 for w in words if any(c in string.punctuation for c in w))
    print(f"  Symbol words: {symbol_words/len(words)*100:.1f}%", end="")
    if symbol_words / len(words) > 0.30:  # RELAXED from 10% to 30%
        print(" FAIL")
        continue
    else:
        print(" PASS")
    
    bullet_lines = sum(1 for l in lines if l.strip().startswith(('-', '*', '•', '·')))
    print(f"  Bullet lines: {bullet_lines/len(lines)*100:.1f}%", end="")
    if len(lines) > 0 and bullet_lines / len(lines) > 0.70:  # RELAXED to 70%
        print(" FAIL")
        continue
    else:
        print(" PASS")
    
    ellipsis_lines = sum(1 for l in lines if l.strip().endswith('...'))
    print(f"  Ellipsis lines: {ellipsis_lines/len(lines)*100:.1f}%", end="")
    if len(lines) > 0 and ellipsis_lines / len(lines) > 0.40:  # RELAXED from 30%
        print(" FAIL")
        continue
    else:
        print(" PASS")
    
    alpha_words = sum(1 for w in words if any(c.isalpha() for c in w))
    print(f"  Alpha words: {alpha_words/len(words)*100:.1f}%", end="")
    if alpha_words / len(words) < 0.10:  # RELAXED from 20%
        print(" FAIL")
        continue
    else:
        print(" PASS")
    
    unique_stops = len(set(w.lower() for w in words) & STOP_WORDS)
    print(f"  Stop words: {unique_stops}", end="")
    if unique_stops < 2:
        print(" FAIL")
        continue
    else:
        print(" PASS")
    
    # FineWeb checks
    punct_ending = sum(1 for l in lines if l.strip() and l.strip()[-1] in '.!?;')
    print(f"  Punct ending: {punct_ending/len(lines)*100:.1f}%", end="")
    if punct_ending / len(lines) < 0.08:  # RELAXED from 12%
        print(" FAIL")
        continue
    else:
        print(" PASS")
    
    short_lines = sum(1 for l in lines if len(l) < 30)
    print(f"  Short lines: {short_lines/len(lines)*100:.1f}%", end="")
    if short_lines / len(lines) > 0.80:  # RELAXED from 67%
        print(" FAIL")
        continue
    else:
        print(" PASS")
    
    char_counts = Counter(text)
    max_dup_ratio = max(char_counts.values()) / len(text)
    print(f"  Char dup (all): {max_dup_ratio*100:.2f}%", end="")
    
    # Exclude space/newline from char dup check
    filtered_counts = {c: n for c, n in char_counts.items() if c not in ' \n\t\r'}
    if filtered_counts:
        max_dup_filtered = max(filtered_counts.values()) / len(text)
        print(f", (no space): {max_dup_filtered*100:.2f}%", end="")
        if max_dup_filtered > 0.12:  # 12% threshold for non-space chars
            print(" FAIL")
            continue
    print(" PASS")
    
    print("  === ALL PASS ===")
