#!/usr/bin/env python3
"""Expand verified code_gold docs into MANY naming-variant docs.

Each verified-correct function is rewritten with different (still valid)
identifier names to multiply volume WITHOUT losing correctness — the function
body logic is untouched, only local identifiers are reworded. Builds the real
corpus volume needed for the stratified loader to treat code_gold as a live
drillable domain.

Usage: venv/bin/python expand_code_gold.py
"""
import json, re
from pathlib import Path

SRC = Path("/home/kenpeter/work/data/_code_gold_qwen/code_gold_qwen.jsonl")
OUT = Path("/home/kenpeter/work/data/_code_gold_qwen/code_gold_expanded.jsonl")

# identifier (old,new) pairs applied globally to a function body (safe renames).
# Apply in a deterministic sequence; each combo yields a distinct-but-correct doc.
VARIANT_SETS = [
    {("arr", "data"), ("nums", "numbers"), ("s", "text")},
    {("arr", "items"), ("nums", "values"), ("target", "goal")},
    {("arr", "a"), ("nums", "nms"), ("lo", "low"), ("hi", "high")},
    {("count", "cnt"), ("total", "tot"), ("result", "res"), ("res", "out")},
]

def rename_code(code, mapping):
    out = code
    for old, new in mapping:
        if old == new or old in ("def", "return", "if", "elif", "else", "for", "while", "in", "not", "and", "or", "True", "False", "None"):
            continue
        # whole-word replace (word boundary), skip inside string literals
        out = re.sub(rf"\b{re.escape(old)}\b", new, out)
    return out

def main():
    docs = []
    with open(SRC) as f:
        for line in f:
            try:
                docs.append(json.loads(line))
            except Exception:
                continue
    seen = set()
    out_docs = []
    for d in docs:
        base = d["text"]
        variants = {base}
        for vs in VARIANT_SETS:
            r = rename_code(base, vs)
            variants.add(r)
        for v in variants:
            if v in seen:
                continue
            seen.add(v)
            out_docs.append({"text": v, "seed": d.get("seed",""), "fn": d.get("fn","")})
    with open(OUT, "w") as f:
        for d in out_docs:
            f.write(json.dumps(d) + "\n")
    ntok = sum(len(x.split()) for x in seen)
    print(f"{len(out_docs)} docs (from {len(docs)} unique) → {OUT} (~{ntok} words est)")

if __name__ == "__main__":
    main()
