#!/usr/bin/env python3
"""
HIGH-PERFORMANCE data preparation for SmolLM2-135M.
Uses batch tokenization + parallel workers to maximize throughput.
"""

import os
import sys
import json
import time
import signal
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer

# ── Config ───────────────────────────────────────────────────────────────────
DATA_DIR = Path(os.environ.get("DATA_DIR", "/home/kenpeter/work/data"))
CHECKPOINT_FILE = DATA_DIR / "prepare_checkpoint_fast.json"
SHARD_SIZE = 1_024_000_000  # ~1GB of uint16 tokens (~512M tokens)
BATCH_SIZE = 512            # tokenize this many texts at once
MAX_WORKERS = 6             # tokenizer threads (half your logical cores)

DATASETS = [
    ("HuggingFaceFW/fineweb-edu", None, 0.50),
    ("mlfoundations/dclm", "baseline_1.0", 0.20),
    ("HuggingFaceTB/stack-edu", None, 0.10),
    ("HuggingFaceTB/finemath", "finemath-3plus", 0.10),
    ("Infi-MM/Infimm-webmath", "Infimm-webmath-3plus", 0.05),
    ("HuggingFaceTB/cosmopedia", "stanford", 0.05),
]

_shutdown_requested = False
def _sig_handler(signum, frame):
    global _shutdown_requested
    _shutdown_requested = True
    print("\n[CTRL+C] Saving checkpoint and exiting...")

signal.signal(signal.SIGINT, _sig_handler)
signal.signal(signal.SIGTERM, _sig_handler)

# ── Helpers ──────────────────────────────────────────────────────────────────

def load_checkpoint():
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE) as f:
            return json.load(f)
    return {"total_tokens": 0, "shards": 0, "dataset_idx": 0, "dataset_samples_seen": 0}

def save_checkpoint(state):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    tmp = CHECKPOINT_FILE.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2)
    tmp.replace(CHECKPOINT_FILE)

def get_disk_usage_gb(path):
    st = os.statvfs(path)
    used = (st.f_blocks - st.f_bavail) * st.f_frsize
    total = st.f_blocks * st.f_frsize
    return used / (1024**3), total / (1024**3)

def write_shard(tokens, shard_id):
    path = DATA_DIR / f"shard_{shard_id:06d}.bin"
    tokens.astype(np.uint16).tofile(path)
    return path

def load_existing_shards():
    shards = sorted(DATA_DIR.glob("shard_*.bin"))
    total = sum(s.stat().st_size // 2 for s in shards)
    return len(shards), total

def load_dataset_robust(ds_name, config, split, max_retries=10):
    for attempt in range(max_retries):
        try:
            return load_dataset(ds_name, config, split=split, streaming=True)
        except Exception as e:
            wait = min(2 ** attempt, 300)
            print(f"   ⚠️  load_dataset failed ({attempt+1}/{max_retries}): {e}")
            time.sleep(wait)
    raise RuntimeError(f"Failed to load {ds_name}")

# ── Batch Tokenizer Worker ───────────────────────────────────────────────────

class BatchTokenizer:
    def __init__(self, tokenizer_name):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def tokenize_batch(self, texts):
        """Tokenize a batch of texts using the fast Rust path."""
        if not texts:
            return []
        enc = self.tokenizer(
            texts,
            add_special_tokens=False,
            truncation=False,
            padding=False,
            return_attention_mask=False,
        )
        return enc["input_ids"]

# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_gb", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    used_gb, total_gb = get_disk_usage_gb(DATA_DIR)
    avail_gb = total_gb - used_gb
    print(f"💾 Disk: {used_gb:.1f} GB used / {total_gb:.1f} GB total / {avail_gb:.1f} GB available")

    target_gb = args.max_gb or max(avail_gb - 20, 10)
    target_tokens = int(target_gb * (1024**3) / 2)
    print(f"🎯 Target: ~{target_gb:.1f} GB (~{target_tokens:,} tokens)")

    # Load tokenizer
    print("📥 Loading tokenizer...")
    tokenizer = BatchTokenizer("HuggingFaceTB/SmolLM2-135M")

    # Resume
    ckpt = load_checkpoint()
    existing_shards, existing_tokens = load_existing_shards()
    ckpt["shards"] = existing_shards
    ckpt["total_tokens"] = existing_tokens
    print(f"🔄 Resuming: {existing_shards} shards, {existing_tokens:,} tokens already written")

    if existing_tokens >= target_tokens:
        print("✅ Target already reached!")
        return

    buffer = np.empty(SHARD_SIZE, dtype=np.uint16)
    buf_pos = 0
    shard_id = existing_shards
    total_tokens = existing_tokens
    pbar = tqdm(total=target_tokens, initial=existing_tokens, unit="tok", unit_scale=True)

    # ThreadPool for parallel tokenization
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future = None
        batch_texts = []
        batch_count = 0

        for ds_idx, (ds_name, ds_config, weight) in enumerate(DATASETS):
            if ds_idx < ckpt.get("dataset_idx", 0):
                continue
            if _shutdown_requested:
                break

            dataset_target = int(target_tokens * weight)
            dataset_tokens = 0
            dataset_samples_seen = ckpt.get("dataset_samples_seen", 0) if ds_idx == ckpt.get("dataset_idx", 0) else 0

            for attempt in range(10):
                if _shutdown_requested:
                    break
                try:
                    print(f"\n📚 Loading dataset: {ds_name}" + (f" ({ds_config})" if ds_config else ""))
                    ds = load_dataset_robust(ds_name, ds_config, "train")

                    if dataset_samples_seen > 0:
                        print(f"   ⏩ Skipping {dataset_samples_seen:,} already-processed samples...")
                        ds = ds.skip(dataset_samples_seen)

                    for sample in ds:
                        if _shutdown_requested:
                            break

                        text = sample.get("text", sample.get("content", ""))
                        if not text:
                            dataset_samples_seen += 1
                            continue

                        batch_texts.append(text)
                        dataset_samples_seen += 1

                        # Submit batch for tokenization
                        if len(batch_texts) >= args.batch_size:
                            # If a previous batch is still running, wait for it
                            if future is not None:
                                token_ids_list = future.result()
                                for ids in token_ids_list:
                                    ids_np = np.array(ids, dtype=np.uint16)
                                    n = len(ids_np)
                                    dataset_tokens += n

                                    i = 0
                                    while i < n:
                                        space = SHARD_SIZE - buf_pos
                                        take = min(space, n - i)
                                        buffer[buf_pos:buf_pos + take] = ids_np[i:i + take]
                                        buf_pos += take
                                        i += take

                                        if buf_pos >= SHARD_SIZE:
                                            write_shard(buffer[:SHARD_SIZE], shard_id)
                                            shard_id += 1
                                            total_tokens += SHARD_SIZE
                                            buf_pos = 0
                                            pbar.update(SHARD_SIZE)

                                            save_checkpoint({
                                                "total_tokens": total_tokens,
                                                "shards": shard_id,
                                                "dataset_idx": ds_idx,
                                                "dataset_samples_seen": dataset_samples_seen,
                                            })

                                            if total_tokens >= target_tokens:
                                                break
                                    if total_tokens >= target_tokens:
                                        break

                            # Submit new batch
                            future = executor.submit(tokenizer.tokenize_batch, batch_texts[:])
                            batch_texts = []
                            batch_count += 1

                        if total_tokens >= target_tokens:
                            break

                    # Flush remaining batch
                    if batch_texts and not _shutdown_requested and total_tokens < target_tokens:
                        if future is not None:
                            future.result()
                        token_ids_list = tokenizer.tokenize_batch(batch_texts)
                        for ids in token_ids_list:
                            ids_np = np.array(ids, dtype=np.uint16)
                            n = len(ids_np)
                            dataset_tokens += n
                            i = 0
                            while i < n:
                                space = SHARD_SIZE - buf_pos
                                take = min(space, n - i)
                                buffer[buf_pos:buf_pos + take] = ids_np[i:i + take]
                                buf_pos += take
                                i += take
                                if buf_pos >= SHARD_SIZE:
                                    write_shard(buffer[:SHARD_SIZE], shard_id)
                                    shard_id += 1
                                    total_tokens += SHARD_SIZE
                                    buf_pos = 0
                                    pbar.update(SHARD_SIZE)
                                    save_checkpoint({
                                        "total_tokens": total_tokens,
                                        "shards": shard_id,
                                        "dataset_idx": ds_idx,
                                        "dataset_samples_seen": dataset_samples_seen,
                                    })
                                    if total_tokens >= target_tokens:
                                        break
                            if total_tokens >= target_tokens:
                                break
                        batch_texts = []

                    print(f"   → Got {dataset_tokens:,} tokens from {ds_name}")
                    ckpt["dataset_idx"] = ds_idx + 1
                    ckpt["dataset_samples_seen"] = 0
                    save_checkpoint(ckpt)
                    break  # success, exit retry loop

                except Exception as e:
                    print(f"\n⚠️  Error during {ds_name} (attempt {attempt+1}/10): {e}")
                    save_checkpoint({
                        "total_tokens": total_tokens,
                        "shards": shard_id,
                        "dataset_idx": ds_idx,
                        "dataset_samples_seen": dataset_samples_seen,
                    })
                    if _shutdown_requested:
                        break
                    if attempt < 9:
                        wait = min(2 ** attempt, 300)
                        print(f"   💾 Checkpoint saved. Retrying in {wait}s...")
                        time.sleep(wait)
                    else:
                        print("   ❌ Max retries reached.")
                        pbar.close()
                        return

    # Flush remaining buffer
    if buf_pos > 0 and not _shutdown_requested:
        write_shard(buffer[:buf_pos], shard_id)
        shard_id += 1
        total_tokens += buf_pos
        pbar.update(buf_pos)

    pbar.close()
    save_checkpoint({
        "total_tokens": total_tokens,
        "shards": shard_id,
        "dataset_idx": 999,
        "dataset_samples_seen": 0,
    })

    used_gb_after, _ = get_disk_usage_gb(DATA_DIR)
    print(f"\n✅ Done! Wrote {shard_id} shards, {total_tokens:,} tokens (~{total_tokens*2/(1024**3):.1f} GB)")
    print(f"💾 Disk usage: {used_gb_after:.1f} GB")

if __name__ == "__main__":
    main()
