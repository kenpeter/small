#!/usr/bin/env python3
"""Fold the 513 exec-verified high-quality LeetCode solutions into code_gold.

Builds code_gold_hq_expanded.jsonl (513 HQ completions + naming variants), then a
merged code_gold_merged_expanded.jsonl = existing Qwen-expanded (241) + HQ-expanded.
This is the text corpus that tokenize_code_gold.py → replicate_code_gold.py consume.

Usage: venv/bin/python fold_hq_code_gold.py
"""
import json, re
from pathlib import Path

HQ = Path("/home/kenpeter/work/data/_code_gold_hq/code_gold_hq_merged.jsonl")
QWEN_EXP = Path("/home/kenpeter/work/data/_code_gold_qwen/code_gold_expanded.jsonl")
OUT_HQ_EXP = Path("/home/kenpeter/work/data/_code_gold_hq/code_gold_hq_expanded.jsonl")
OUT_MERGED = Path("/home/kenpeter/work/data/_code_gold_hq/code_gold_merged_expanded.jsonl")

VARIANT_SETS = [
    {("arr", "data"), ("nums", "numbers"), ("s", "text"), ("lst", "items")},
    {("arr", "items"), ("nums", "values"), ("target", "goal"), ("data", "payload")},
    {("arr", "a"), ("nums", "nms"), ("lo", "low"), ("hi", "high"), ("left", "l"), ("right", "r")},
    {("count", "cnt"), ("total", "tot"), ("result", "res"), ("res", "out"), ("node", "nd")},
]

PY_KEYWORDS = {"def","return","if","elif","else","for","while","in","not","and","or",
               "True","False","None","class","self","import","from","as","lambda",
               "is","pass","break","continue","try","except","raise","with","yield",
               "global","nonlocal","assert","del"}

def rename_code(code, mapping):
    out = code
    for old, new in mapping:
        if old == new or old in PY_KEYWORDS or not old.isidentifier():
            continue
        out = re.sub(rf"\b{re.escape(old)}\b", new, out)
    return out

def expand(docs_out, doc_list, extra_fields=lambda d: {}):
    seen = set()
    out_docs = []
    for d in doc_list:
        base = d["text"] if "text" in d else d["completion"]
        variants = {base}
        for vs in VARIANT_SETS:
            variants.add(rename_code(base, vs))
        for v in variants:
            if v in seen:
                continue
            seen.add(v)
            meta = {"text": v}
            meta.update(extra_fields(d))
            out_docs.append(meta)
    return out_docs

def main():
    # HQ docs
    hq = []
    with open(HQ) as f:
        for line in f:
            try:
                hq.append(json.loads(line))
            except Exception:
                continue
    hq_docs = expand(None, hq, lambda d: {"source": "hq", "task_id": d.get("task_id",""),
                                          "difficulty": d.get("difficulty","")})
    with open(OUT_HQ_EXP, "w") as f:
        for d in hq_docs:
            f.write(json.dumps(d) + "\n")
    print(f"HQ: {len(hq)} verified → {len(hq_docs)} expanded variants → {OUT_HQ_EXP}")

    # Existing Qwen expanded
    qwen = []
    with open(QWEN_EXP) as f:
        for line in f:
            try:
                qwen.append(json.loads(line))
            except Exception:
                continue

    # Merge: HQ first, then qwen (dedupe by text)
    seen_text = set(d["text"] for d in hq_docs)
    merged = list(hq_docs)
    qwen_added = 0
    for d in qwen:
        t = d.get("text","")
        if t and t not in seen_text:
            seen_text.add(t)
            merged.append(d)
            qwen_added += 1
    seen2 = set()
    merged_uniq = []
    for d in merged:
        if d["text"] in seen2:
            continue
        seen2.add(d["text"])
        merged_uniq.append(d)
    with open(OUT_MERGED, "w") as f:
        for d in merged_uniq:
            f.write(json.dumps(d) + "\n")
    ntokest = sum(len(d["text"].split()) for d in merged_uniq)
    print(f"MERGED: {len(merged_uniq)} unique docs (HQ {len(hq_docs)} + qwen added {qwen_added}) "
          f"→ {OUT_MERGED} (~{ntokest:,} words est)")

if __name__ == "__main__":
    main()
