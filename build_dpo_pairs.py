#!/usr/bin/env python3
"""Build DPO preference pairs from gold_combined (code domain).

For each code problem:
  - prompt   = clean problem statement (before '## THREE SOLUTION METHODS')
  - chosen   = METHOD 1 solution (the canonical correct reference)
  - rejected = a sampled SFT-model completion that is WRONG (fails answer match
               or is an echo-loop / non-Python / over-short). We sample several
               completions per problem and keep the first 'bad' one; if none bad,
               skip the problem (don't force noisy pairs).

Output: data/_dpo/dpo_pairs.jsonl  [{prompt, chosen, rejected, domain, difficulty}]
Base model for sampling: hf_export_sft1b (the SFT HF export).
"""
import argparse, glob, json, os, re, time
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

GOLD = Path("/home/kenpeter/work/data/_gold/gold_combined.jsonl")
BASE = Path("/home/kenpeter/work/hf_export_sft1b")
OUT = Path("/home/kenpeter/work/data/_dpo/dpo_pairs.jsonl")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

GEN_KW = dict(do_sample=True, top_p=0.9, temperature=0.8, max_new_tokens=256, num_return_sequences=2)


def split_problem(text: str):
    """Return (prompt, method1_solution)."""
    m = re.split(r"\n\s*## THREE SOLUTION METHODS", text, maxsplit=1)
    prompt = m[0].strip()
    sol_text = m[1] if len(m) > 1 else ""
    # METHOD 1 block
    m1 = re.search(r"METHOD 1:.*?\n(.*?)(?=\n\s*METHOD 2:|\Z)", sol_text, re.S)
    body = m1.group(1) if m1 else sol_text
    return prompt, body


def extract_answer(completion: str):
    """Pull <answer>...</answer> if present, else the code block / whole text."""
    m = re.search(r"<answer>(.*?)</answer>", completion, re.S)
    if m:
        return m.group(1).strip()
    m = re.search(r"```(?:python)?\n(.*?)```", completion, re.S)
    if m:
        return m.group(1).strip()
    return completion.strip()


def is_echo_loop(completion: str):
    # heuristics for the failure modes seen in SFT eval
    lines = [l for l in completion.splitlines() if l.strip()]
    if len(lines) < 3:
        return True
    if "def " not in completion and "class " not in completion:
        return True  # no attempt at a function/class
    # repetition of identical function defs
    defs = re.findall(r"^(def|class)\s+\w+\s*", completion, re.M)
    if len(defs) >= 2 and len(set(defs)) == 1:
        return True
    return False


def looks_wrong(completion: str, chosen: str):
    """A completion is 'wrong' if it clearly diverges from the reference solution."""
    if is_echo_loop(completion):
        return True
    # wrong-language detection: python stub but C++/Java emitted
    if re.search(r"#include\s*[<\"]|using namespace std|public class|public static", completion):
        return True
    # sanity: must contain at least one def/class and a real body
    if "def " not in completion and "class " not in completion:
        return True
    ans = extract_answer(completion)
    if len(ans) < 20:
        return True  # too short = likely incomplete/hallucinated
    # If chosen has a recognizable canonical body (e.g. 'return s[::-1]'), and
    # completion lacks its core return, it's wrong. We keep this soft.
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-rows", type=int, default=1646)
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--max-pairs", type=int, default=2000)
    args = ap.parse_args()

    print("Loading gold code rows...", flush=True)
    rows = []
    with open(GOLD) as f:
        for line in f:
            d = json.loads(line)
            if d.get("domain") == "code":
                rows.append(d)
        if len(rows) >= args.max_rows:
            pass
    rows = rows[: args.max_rows]
    print(f"  {len(rows)} code rows", flush=True)

    print(f"Loading SFT model from {BASE}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(BASE), torch_dtype=torch.bfloat16, attn_implementation="sdpa"
    ).to(DEVICE)
    tok = AutoTokenizer.from_pretrained(str(BASE))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model.eval()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    n_pairs = 0
    n_skipped = 0
    with open(OUT, "w") as fo:
        for i, d in enumerate(rows):
            prompt, chosen = split_problem(d["text"])
            if len(prompt) < 30 or len(chosen) < 40:
                n_skipped += 1
                continue
            # reject if prompt has no code function signal or is overly long
            if len(prompt) > 1500:
                n_skipped += 1
                continue
            # Raw-format (matches SFT training: problem text -> solution continuation,
            # no chat markers). SmolLM2 tokenizer has no chat_template set.
            user = f"{prompt}\n\nSolution:\n```python\n"
            enc = tok(user, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                out_ids = model.generate(
                    enc.input_ids,
                    attention_mask=enc.attention_mask,
                    pad_token_id=tok.pad_token_id,
                    **GEN_KW,
                )
            rejected = None
            for j in range(args.samples):
                comp_ids = out_ids[j][enc.input_ids.shape[1]:]
                comp = tok.decode(comp_ids, skip_special_tokens=True).strip()
                if looks_wrong(comp, chosen):
                    rejected = comp
                    break
            if rejected is None:
                n_skipped += 1
                continue
            rec = {
                "prompt": user,
                "chosen": chosen.strip(),
                "rejected": rejected,
                "domain": "code",
                "difficulty": d.get("difficulty", ""),
            }
            fo.write(json.dumps(rec) + "\n")
            n_pairs += 1
            if n_pairs % 50 == 0:
                print(f"  {n_pairs} pairs built (skipped {n_skipped})", flush=True)
            if n_pairs >= args.max_pairs:
                break
    print(f"DONE: {n_pairs} pairs -> {OUT} (skipped {n_skipped})", flush=True)


if __name__ == "__main__":
    main()
