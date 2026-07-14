"""Quick test of quality filters on first 100 sequences."""
import numpy as np
import string
from collections import Counter
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")
tokens = np.memmap("/home/kenpeter/work/data/shard_000000.bin", dtype=np.uint16, mode='r')

STOP_WORDS = {'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'i', 'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at'}
BAD_WORDS = {'http', 'https', 'www', 'com', 'html'}

def gopher_filter(text):
    words = text.split()
    if len(words) == 0:
        return False, "empty"
    word_count = len(words)
    if word_count < 50:
        return False, "too_short"
    if word_count > 100000:
        return False, "too_long"
    avg_word_len = sum(len(w) for w in words) / word_count
    if avg_word_len < 2.5 or avg_word_len > 12:
        return False, "avg_word_len"
    symbol_words = sum(1 for w in words if any(c in string.punctuation for c in w))
    if symbol_words / word_count > 0.30:
        return False, "too_many_symbols"
    lines = text.split('\n')
    bullet_lines = sum(1 for l in lines if l.strip().startswith(('-', '*', '•', '·')))
    if len(lines) > 0 and bullet_lines / len(lines) > 0.70:
        return False, "too_many_bullets"
    ellipsis_lines = sum(1 for l in lines if l.strip().endswith('...'))
    if len(lines) > 0 and ellipsis_lines / len(lines) > 0.40:
        return False, "too_many_ellipsis"
    alpha_words = sum(1 for w in words if any(c.isalpha() for c in w))
    if alpha_words / word_count < 0.10:
        return False, "too_few_alpha_words"
    unique_stops = len(set(w.lower() for w in words) & STOP_WORDS)
    if unique_stops < 2:
        return False, "too_few_stop_words"
    return True, "pass"

def fineweb_filter(text):
    lines = text.split('\n')
    if len(lines) == 0:
        return False, "empty"
    punct_ending = sum(1 for l in lines if l.strip() and l.strip()[-1] in '.!?;')
    if len(lines) > 0 and punct_ending / len(lines) < 0.08:
        return False, "low_punct_ending"
    short_lines = sum(1 for l in lines if len(l) < 30)
    if len(lines) > 0 and short_lines / len(lines) > 0.80:
        return False, "too_many_short_lines"
    if len(text) > 0:
        char_counts = Counter(text)
        filtered_counts = {c: n for c, n in char_counts.items() if c not in ' \n\t\r'}
        if filtered_counts:
            max_dup = max(filtered_counts.values()) / len(text)
            if max_dup > 0.12:
                return False, "char_repetition"
    newline_count = text.count('\n')
    if len(text) > 0 and newline_count / len(text) > 0.40:
        return False, "too_many_newlines"
    return True, "pass"

def c4_filter(text):
    words = text.lower().split()
    lines = text.split('\n')
    if any(bw in words for bw in BAD_WORDS):
        return False, "bad_words"
    if len(lines) < 2:
        return False, "too_few_lines"
    short_paras = sum(1 for l in lines if len(l.split()) < 5)
    if len(lines) > 0 and short_paras / len(lines) > 0.95:
        return False, "too_many_short_paras"
    return True, "pass"

kept = 0
rejected = Counter()

for i in range(100):
    start = i * 2048
    seq = tokens[start:start+2048]
    text = tokenizer.decode(seq, skip_special_tokens=True)
    
    g, rg = gopher_filter(text)
    f, rf = fineweb_filter(text)
    c, rc = c4_filter(text)
    
    if g and f and c:
        kept += 1
    else:
        if not g: rejected[f"gopher:{rg}"] += 1
        elif not f: rejected[f"fineweb:{rf}"] += 1
        elif not c: rejected[f"c4:{rc}"] += 1

print(f"Tested 100 sequences:")
print(f"  Kept: {kept} ({kept}%)")
print(f"  Rejected: {100-kept}")
print(f"  Rejection breakdown:")
for r, c in rejected.most_common():
    print(f"    {r}: {c}")
