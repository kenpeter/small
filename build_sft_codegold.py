#!/usr/bin/env python3
"""Regenerate SFT shards from the exec-verified LeetCode gold set.

Reads code_gold_ALL_verified.jsonl (exec-verified problem->correct Solution),
splits each doc into prompt = problem statement, response = the class Solution
code, and tokenizes into the SFT .pt shard format the sft_gpu.py loader expects:

    {"input_ids": [...], "labels": [...], "dataset": "..."}

Prompt tokens -> labels -100 (masked); solution tokens -> labels = ids (trained).

Usage:
  ./venv/bin/python build_sft_codegold.py [--in PATH] [--out-dir DIR] [--dedupe]
"""
import argparse, json, time
from pathlib import Path
import torch
from transformers import AutoTokenizer

TOK_NAME = "HuggingFaceTB/SmolLM2-135M"
MAX_TOK = 2048
DATASET = "sft_gold_code_verified"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp",
                    default="/home/kenpeter/work/data/_code_gold_hq/code_gold_ALL_verified.jsonl")
    ap.add_argument("--out-dir", default="/mnt/file_drive/data/_sft_codegold_shards")
    ap.add_argument("--per-shard", type=int, default=1000)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(TOK_NAME, trust_remote_code=True)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Keep RAW verified dir + this final dir separate (no overwrites).
    shards, n, t0, kept, skipped = [], 0, time.time(), 0, 0
    seen = set()
    for line in open(args.inp):
        d = json.loads(line)
        task_id = d.get("task_id")
        if task_id in seen:
            skipped += 1
            continue
        seen.add(task_id)
        text = d.get("text", "")
        if "# Solution" not in text:
            skipped += 1
            continue
        prompt, sol = text.split("# Solution", 1)
        sol = sol.strip()
        if not prompt.strip() or not sol.strip():
            skipped += 1
            continue
        p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
        r_ids = tok(sol, add_special_tokens=False)["input_ids"]
        if len(p_ids) + len(r_ids) > MAX_TOK:
            over = len(p_ids) + len(r_ids) - MAX_TOK
            if over < len(p_ids):
                p_ids = p_ids[over:]
            else:
                p_ids, r_ids = [], r_ids[:MAX_TOK]
        ids = p_ids + r_ids
        labels = [-100] * len(p_ids) + r_ids
        shards.append({"input_ids": ids, "labels": labels, "dataset": DATASET})
        kept += 1
        if len(shards) >= args.per_shard:
            n += 1
            torch.save(shards, out_dir / f"sftcg_{n:05d}.pt")
            shards = []
            print(f"  shard {n}: {args.per_shard} ex, {(time.time()-t0)/60:.1f} min", flush=True)
    if shards:
        n += 1
        torch.save(shards, out_dir / f"sftcg_{n:05d}.pt")
        print(f"  shard {n}: {len(shards)} ex (final)", flush=True)
    print(f"DONE: kept={kept} skipped={skipped} shards={n} -> {out_dir}", flush=True)

if __name__ == "__main__":
    main()
