"""
Tokenize SFT datasets with 5 parallel workers (one per dataset).
Each worker saves shards as shard_{dataset}_{idx}.pt independently.
"""
import os, sys, json, time, gc
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass

import numpy as np
import torch
from transformers import AutoTokenizer

SFT_STAGING = Path("/home/kenpeter/work/data/_sft_staging")
OUT_DIR = Path("/home/kenpeter/work/data/_sft_shards")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SHARD_SIZE = 100_000
MAX_SEQ_LEN = 2048

# ─── Chat formatters (same as before) ─────────────────────────────────
def format_chat(messages):
    text = ""
    for msg in messages:
        text += f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
    return text

def fmt_openhermes(ex):
    msgs = ex.get("messages", [])
    if not msgs: return None
    if msgs[0].get("role") != "system":
        msgs = [{"role": "system", "content": "You are a helpful assistant."}] + msgs
    return msgs

def fmt_openorca(ex):
    system = ex.get("system_prompt", "You are a helpful assistant.")
    q = ex.get("question", ex.get("prompt", ""))
    a = ex.get("response", ex.get("answer", ""))
    if not q or not a: return None
    return [{"role": "system", "content": system},
            {"role": "user", "content": q},
            {"role": "assistant", "content": a}]

def fmt_alpaca(ex):
    inst = ex.get("instruction", "")
    inp = ex.get("input", "")
    out = ex.get("output", "")
    if not inst or not out: return None
    if inp: inst += f"\n{inp}"
    return [{"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": inst},
            {"role": "assistant", "content": out}]

def fmt_ultrachat(ex):
    data = ex.get("data", [])
    if not data or len(data) < 2: return None
    msgs = [{"role": "system", "content": "You are a helpful assistant."}]
    for i, turn in enumerate(data):
        role = "user" if i % 2 == 0 else "assistant"
        msgs.append({"role": role, "content": turn})
    return msgs

DATASET_CONFIG = {
    "openhermes":  fmt_openhermes,
    "openorca":    fmt_openorca,
    "ultrachat":   fmt_ultrachat,
    "alpaca_gpt4": fmt_alpaca,
    "code_alpaca": fmt_alpaca,
}

def tokenize_one(tokenizer, messages, max_len=MAX_SEQ_LEN):
    full_text = ""
    role_boundaries = []
    for msg in messages:
        start = len(full_text)
        full_text += f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
        end = len(full_text)
        role_boundaries.append((start, end, msg["role"]))

    encoding = tokenizer(full_text, add_special_tokens=False, return_offsets_mapping=True)
    input_ids = encoding["input_ids"]
    offsets = encoding.get("offset_mapping", [])

    if len(input_ids) > max_len:
        input_ids = input_ids[:max_len]
        offsets = offsets[:max_len]

    labels = [-100] * len(input_ids)
    if offsets:
        for i, (cs, ce) in enumerate(offsets):
            for rstart, rend, role in role_boundaries:
                if cs >= rstart and ce <= rend and role == "assistant":
                    labels[i] = input_ids[i]
                    break
    return input_ids, labels

def process_dataset(name: str, formatter, tokenizer_name="HuggingFaceTB/SmolLM2-135M"):
    """Worker: tokenize one dataset, save shards."""
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    ds_dir = SFT_STAGING / name
    if not ds_dir.exists():
        return {"name": name, "status": "missing", "samples": 0}

    jsonl_files = sorted(ds_dir.glob("shard_*.jsonl"))
    total = 0
    all_samples = []
    shard_idx = 0

    for jfile in jsonl_files:
        with open(jfile, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    ex = json.loads(line)
                except json.JSONDecodeError:
                    continue
                messages = formatter(ex)
                if messages is None:
                    continue
                input_ids, labels = tokenize_one(tokenizer, messages, MAX_SEQ_LEN)
                if len(input_ids) < 2:
                    continue
                all_samples.append({"input_ids": input_ids, "labels": labels, "dataset": name})
                total += 1
                if len(all_samples) >= SHARD_SIZE:
                    out_path = OUT_DIR / f"shard_{name}_{shard_idx:05d}.pt"
                    torch.save(all_samples, out_path)
                    all_samples = []
                    shard_idx += 1
                    gc.collect()

    if all_samples:
        out_path = OUT_DIR / f"shard_{name}_{shard_idx:05d}.pt"
        torch.save(all_samples, out_path)
        shard_idx += 1

    elapsed = time.time() - t0
    return {"name": name, "status": "ok", "samples": total, "shards": shard_idx, "time": elapsed}

def main():
    print(f"SFT staging: {SFT_STAGING}")
    print(f"Output: {OUT_DIR}")
    print(f"Workers: {len(DATASET_CONFIG)} (one per dataset)\n")

    results = []
    with ProcessPoolExecutor(max_workers=6) as executor:
        futures = {executor.submit(process_dataset, name, fmt): name
                   for name, fmt in DATASET_CONFIG.items()}
        for future in as_completed(futures):
            results.append(future.result())

    print("\n" + "="*60)
    print("SFT TOKENIZATION SUMMARY")
    print("="*60)
    total_samples = 0
    total_shards = 0
    for r in results:
        if r["status"] == "ok":
            print(f"  ✅ {r['name']:15s} | {r['samples']:>10,} samples | {r['shards']:>4} shards | {r['time']:.0f}s")
            total_samples += r["samples"]
            total_shards += r["shards"]
        else:
            print(f"  ❌ {r['name']:15s} | {r['status']}")

    print(f"\n  TOTAL: {total_samples:,} samples | {total_shards} shards")
    print(f"  Output: {OUT_DIR}")

if __name__ == "__main__":
    main()
