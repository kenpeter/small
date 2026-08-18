#!/usr/bin/env python3
"""Tokenize SFT gold JSONL → .pt shards (same format as _sft_final_shards).
Prompt tokens → labels -100; assistant response → labels = input_ids (trained).

Usage: ./venv/bin/python tokenize_sft_gold.py --in data/_sft_gold/gold.jsonl
"""
import argparse, json, os, time
from pathlib import Path
import torch
from transformers import AutoTokenizer

TOK_NAME = "HuggingFaceTB/SmolLM2-135M"
MAX_TOK = 2048

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="data/_sft_gold/gold.jsonl")
    ap.add_argument("--out-dir", default="/mnt/file_drive/data/_sft_gold_shards")
    ap.add_argument("--per-shard", type=int, default=1000)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(TOK_NAME, trust_remote_code=True)
    # Qwen printed with markdown-ish header; build chat-ish: question + answer
    # Simplest robust: prompt tokens then response tokens, no special chat template
    # (the base model was pretrained on plain text; SFT teaches format via data)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    shards = []
    n = 0
    t0 = time.time()
    for line in open(args.inp):
        d = json.loads(line)
        prompt = d["prompt"]
        resp = d["response"]
        p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
        r_ids = tok(resp, add_special_tokens=False)["input_ids"]
        # cap at MAX_TOK: keep the ANSWER (truncate prompt from the front)
        if len(p_ids) + len(r_ids) > MAX_TOK:
            over = len(p_ids) + len(r_ids) - MAX_TOK
            if over < len(p_ids):
                p_ids = p_ids[over:]
            else:
                p_ids = []
                r_ids = r_ids[:MAX_TOK]
        ids = p_ids + r_ids
        labels = [-100] * len(p_ids) + r_ids
        shards.append({"input_ids": ids, "labels": labels, "dataset": d["dataset"]})
        if len(shards) >= args.per_shard:
            n += 1
            torch.save(shards, out_dir / f"gold_{n:05d}.pt")
            shards = []
            print(f"  shard {n}: {args.per_shard} ex, {(time.time()-t0)/60:.1f} min", flush=True)
    if shards:
        n += 1
        torch.save(shards, out_dir / f"gold_{n:05d}.pt")
        print(f"  shard {n}: {len(shards)} ex (final)", flush=True)
    print(f"DONE: {n} shards → {out_dir}", flush=True)

if __name__ == "__main__":
    main()