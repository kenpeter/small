#!/usr/bin/env python3
"""Smarter preamble stripping - detect preamble paragraphs by signal word density."""
import json, re
from pathlib import Path

SIGNAL_WORDS = re.compile(
    r'\b(okay|ok|alright|well|let me|i need|i will|i shall|i can|i start|the user|'
    r'the text|the provided|the given|the original|the article|the passage|the source|'
    r'first,|firstly|second,|secondly|next,|finally,|first i|first we|'
    r'as an ai|as an assistant|as an expert|as an educator|'
    r'i should|i want|i am going|i\'ll start|i\'m going|this text|this document|'
    r'looking at|reading through|understanding the|analyzing the|examining the)',
    re.IGNORECASE
)

def strip_preamble_smart(text, fmt):
    """Remove preamble using signal word density on paragraphs."""
    # For QA: find first Q: 
    if fmt == "qa":
        m = re.search(r'(?:^|\n)(Q[.:]?\s)', text)
        if m:
            text = text[m.start():].strip()
        return text
    
    # For textbook: strip preamble paragraphs
    # A preamble paragraph has high density of signal words
    paragraphs = re.split(r'\n\n+', text)
    
    # If only one paragraph, treat entire text differently
    if len(paragraphs) <= 1:
        # Check if the single paragraph is preamble
        words = text.split()
        if len(words) > 3:  # meaningful text
            signal_count = len(SIGNAL_WORDS.findall(text))
            if signal_count > 0 and signal_count / max(len(words), 1) > 0.03:
                return ""  # return empty, whole thing is preamble
        return text
    
    # Find first non-preamble paragraph
    start_idx = 0
    for idx, para in enumerate(paragraphs):
        para = para.strip()
        if not para:
            continue
        
        words = para.split()
        if len(words) < 5:
            # Short paragraph - could be a heading
            continue
        
        signal_count = len(SIGNAL_WORDS.findall(para))
        
        # A paragraph is preamble if >15% of words are signal words
        # Or if it starts with certain patterns
        ratio = signal_count / len(words)
        starts_with_preamble = bool(re.match(
            r'^(Okay|Ok|Alright|Well|So[,!\s]|First[,!\s]|Now[,!\s]|'
            r'Let me|I need|I will|I shall|I can|I start|I should|I want|'
            r'The user|The text|The provided|The given|The original|'
            r'As an|Looking at|Reading through|Understanding|Analyzing)',
            para
        ))
        
        if ratio < 0.15 and not starts_with_preamble:
            # Not preamble - this is where content starts
            start_idx = idx
            break
    
    # Rejoin from the first content paragraph
    result = '\n\n'.join(paragraphs[start_idx:]).strip()
    return result

# Process all chunks
base = Path("/home/kenpeter/work/data")
total_text = 0
total_qa = 0
removed_text = 0
removed_qa = 0

for i in range(8):
    d = base / f"_reformatted_chunk_{i}"
    if not d.exists():
        continue
    
    for fmt_name in ["textbook", "qa"]:
        f = d / f"{fmt_name}.jsonl"
        if not f.exists():
            continue
        
        with open(f) as fh:
            docs = [json.loads(line) for line in fh]
        
        cleaned = []
        removed = 0
        for doc in docs:
            text = doc.get("text", "")
            stripped = strip_preamble_smart(text, fmt_name)
            if stripped and len(stripped) >= 50:
                doc["text"] = stripped
                cleaned.append(doc)
            else:
                removed += 1
        
        # Overwrite
        with open(f, "w") as fh:
            for doc in cleaned:
                fh.write(json.dumps(doc, ensure_ascii=False) + "\n")
        
        if fmt_name == "textbook":
            total_text += len(docs)
            removed_text += removed
        else:
            total_qa += len(docs)
            removed_qa += removed
        
        print(f"Chunk {i} {fmt_name}: {len(docs)} -> {len(cleaned)} ({removed} removed)")

print(f"\nTotal textbook: {total_text} -> {total_text - removed_text}")
print(f"Total QA: {total_qa} -> {total_qa - removed_qa}")

# Show samples
print("\n=== SAMPLE CLEANED ===")
for i in range(8):
    for fmt in ["textbook", "qa"]:
        f = base / f"_reformatted_chunk_{i}" / f"{fmt}.jsonl"
        if f.exists():
            with open(f) as fh:
                first = json.loads(fh.readline())
                text = first.get("text", "")
                tok = first.get("tokens_out", 0)
                print(f"Chunk {i} {fmt} (tok={tok}): {text[:200]}")
                print()
                break
    break  # Just first chunk
