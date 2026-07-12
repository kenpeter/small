#!/usr/bin/env python3
"""
Prepare training data for SmolLM2-135M replication.
Downloads, tokenizes, and packs data from HuggingFace datasets.
Robustly resumable via checkpoint files (sample-level precision).
Saves as memory-mappable uint16 binaries.
"""

import os
import sys
import json
import time
import signal
import argparse
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer

# ── Config ───────────────────────────────────────────────────────────────────
DATA_DIR = Path(os.environ.get("DATA_DIR", "/home/kenpeter/work/data"))
CHECKPOINT_FILE = DATA_DIR / "prepare_checkpoint.json"
CHUNK_SIZE = 2048          # sequence length
TOKENS_PER_CHUNK = CHUNK_SIZE * 1024 * 1024  # tokens per shard file (~1MB uint16)
CHUNK_TOKENS = TOKENS_PER_CHUNK // CHUNK_SIZE
SHARD_SIZE = CHUNK_TOKENS * CHUNK_SIZE  # tokens per .bin file (~1GB)

# Dataset mixture ratios (single-stage, from SmolLM2-135M paper)
DATASETS = [
    ("HuggingFaceFW/fineweb-edu", None, 0.50),          # web edu
    ("mlfoundations/dclm", "baseline_1.0", 0.20),       # web
    ("HuggingFaceTB/stack-edu", None, 0.10),              # code
    ("HuggingFaceTB/finemath", "finemath-3plus", 0.10),   # math
    ("Infi-MM/Infimm-webmath", "Infimm-webmath-3plus", 0.05),  # math
    ("HuggingFaceTB/cosmopedia", "stanford", 0.05),       # synthetic
]

# ── Globals for graceful shutdown ────────────────────────────────────────────
_shutdown_requested = False

def _sig_handler(signum, frame):
    global _shutdown_requested
    _shutdown_requested = True
    print("\n[CTRL+C] Shutdown requested — will save checkpoint and exit...")

signal.signal(signal.SIGINT, _sig_handler)
signal.signal(signal.SIGTERM, _sig_handler)

# ── Helpers ──────────────────────────────────────────────────────────────────

def load_checkpoint() -> dict:
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE) as f:
            return json.load(f)
    return {
        "total_tokens": 0,
        "shards": 0,
        "dataset_idx": 0,
        "dataset_samples_seen": 0,
    }

def save_checkpoint(state: dict):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    tmp = CHECKPOINT_FILE.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2)
    tmp.replace(CHECKPOINT_FILE)

def get_disk_usage_gb(path: Path) -> float:
    st = os.statvfs(path)
    used = (st.f_blocks - st.f_bavail) * st.f_frsize
    total = st.f_blocks * st.f_frsize
    return used / (1024**3), total / (1024**3)

def write_shard(tokens: np.ndarray, shard_id: int) -> Path:
    """Write a shard of uint16 tokens to disk."""
    path = DATA_DIR / f"shard_{shard_id:06d}.bin"
    tokens.astype(np.uint16).tofile(path)
    return path

def load_existing_shards() -> tuple[int, int]:
    """Count existing shards and total tokens."""
    shards = sorted(DATA_DIR.glob("shard_*.bin"))
    total_tokens = 0
    for s in shards:
        total_tokens += s.stat().st_size // 2  # uint16 = 2 bytes
    return len(shards), total_tokens

def load_dataset_with_retry(ds_name, ds_config, split, streaming, max_retries=5):
    """Load dataset with exponential backoff retry."""
    for attempt in range(max_retries):
        try:
            return load_dataset(ds_name, ds_config, split=split, streaming=streaming)
        except Exception as e:
            wait = min(2 ** attempt, 300)
            print(f"   ⚠️  load_dataset failed (attempt {attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                print(f"   ⏳ Retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise

# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Prepare SmolLM2-135M training data")
    parser.add_argument("--max_tokens", type=int, default=None,
                        help="Maximum tokens to generate (default: fill disk until 50GB left)")
    parser.add_argument("--max_gb", type=float, default=None,
                        help="Max GB of tokenized data to generate")
    parser.add_argument("--streaming", action="store_true", default=True,
                        help="Use streaming mode (default True)")
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    used_gb, total_gb = get_disk_usage_gb(DATA_DIR)
    avail_gb = total_gb - used_gb
    print(f"💾 Disk: {used_gb:.1f} GB used / {total_gb:.1f} GB total / {avail_gb:.1f} GB available")

    # Determine target
    if args.max_gb:
        target_gb = args.max_gb
    elif args.max_tokens:
        target_gb = args.max_tokens * 2 / (1024**3)
    else:
        target_gb = max(avail_gb - 20, 10)  # leave 20GB headroom
    target_tokens = int(target_gb * (1024**3) / 2)

    print(f"🎯 Target: ~{target_gb:.1f} GB of tokenized data (~{target_tokens:,} tokens)")

    # Load tokenizer
    print("📥 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Resume state
    ckpt = load_checkpoint()
    existing_shards, existing_tokens = load_existing_shards()
    ckpt["shards"] = existing_shards
    ckpt["total_tokens"] = existing_tokens
    dataset_samples_seen = ckpt.get("dataset_samples_seen", 0)
    print(f"🔄 Resuming: {existing_shards} shards, {existing_tokens:,} tokens already written")

    if existing_tokens >= target_tokens:
        print("✅ Target already reached!")
        return

    # Buffer for packing
    buffer = np.empty(SHARD_SIZE, dtype=np.uint16)
    buf_pos = 0
    shard_id = ckpt["shards"]
    total_tokens = existing_tokens
    pbar = tqdm(total=target_tokens, initial=existing_tokens, unit="tok", unit_scale=True)

    # Iterate datasets
    for ds_idx, (ds_name, ds_config, weight) in enumerate(DATASETS):
        if ds_idx < ckpt.get("dataset_idx", 0):
            continue
        if _shutdown_requested:
            break

        dataset_target = int(target_tokens * weight)
        dataset_tokens = 0

        # If resuming this exact dataset, restore sample offset
        if ds_idx == ckpt.get("dataset_idx", 0):
            dataset_samples_seen = ckpt.get("dataset_samples_seen", 0)
        else:
            dataset_samples_seen = 0

        # Robust retry loop: reloads dataset and skips seen samples on failure
        max_retries = 10
        for attempt in range(max_retries):
            if _shutdown_requested:
                break

            try:
                print(f"\n📚 Loading dataset: {ds_name}" + (f" ({ds_config})" if ds_config else ""))
                ds = load_dataset_with_retry(ds_name, ds_config, "train", args.streaming)

                # Resume from exact sample position using .skip()
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

                    # Tokenize (no padding, just raw tokens)
                    ids = tokenizer.encode(text, add_special_tokens=False)
                    ids_np = np.array(ids, dtype=np.uint16)
                    n = len(ids_np)
                    dataset_tokens += n
                    dataset_samples_seen += 1

                    # Pack into buffer
                    i = 0
                    while i < n:
                        space = SHARD_SIZE - buf_pos
                        take = min(space, n - i)
                        buffer[buf_pos:buf_pos + take] = ids_np[i:i + take]
                        buf_pos += take
                        i += take

                        if buf_pos >= SHARD_SIZE:
                            # Flush shard
                            path = write_shard(buffer[:SHARD_SIZE], shard_id)
                            shard_id += 1
                            total_tokens += SHARD_SIZE
                            buf_pos = 0
                            pbar.update(SHARD_SIZE)

                            # Checkpoint after EVERY shard for maximum resumability
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

                # Success: finished this dataset (or target reached)
                print(f"   → Got {dataset_tokens:,} tokens from {ds_name}")
                ckpt["dataset_idx"] = ds_idx + 1
                ckpt["dataset_samples_seen"] = 0
                save_checkpoint(ckpt)
                break  # exit retry loop

            except Exception as e:
                print(f"\n⚠️  Error during {ds_name} (attempt {attempt+1}/{max_retries}): {e}")
                save_checkpoint({
                    "total_tokens": total_tokens,
                    "shards": shard_id,
                    "dataset_idx": ds_idx,
                    "dataset_samples_seen": dataset_samples_seen,
                })
                if _shutdown_requested:
                    break
                if attempt < max_retries - 1:
                    wait_time = min(2 ** attempt, 300)  # exponential backoff, max 5 min
                    print(f"   💾 Checkpoint saved. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                    # On retry, dataset_samples_seen is preserved, so .skip() will resume
                else:
                    print(f"   ❌ Max retries reached. Exiting.")
                    pbar.close()
                    return

    # Flush remaining buffer
    if buf_pos > 0 and not _shutdown_requested:
        path = write_shard(buffer[:buf_pos], shard_id)
        shard_id += 1
        total_tokens += buf_pos
        pbar.update(buf_pos)

    pbar.close()

    # Final checkpoint
    save_checkpoint({
        "total_tokens": total_tokens,
        "shards": shard_id,
        "dataset_idx": 999,  # done
        "dataset_samples_seen": 0,
    })

    used_gb_after, _ = get_disk_usage_gb(DATA_DIR)
    print(f"\n✅ Done! Wrote {shard_id} shards, {total_tokens:,} tokens (~{total_tokens*2/(1024**3):.1f} GB)")
    print(f"💾 Disk usage: {used_gb_after:.1f} GB")

if __name__ == "__main__":
    main()
