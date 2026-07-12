#!/usr/bin/env python3
"""
LOCAL-PARQUET data preparation for SmolLM2-135M.
Downloads parquet files to disk, then processes with pyarrow + batch tokenization.
No streaming skip() overhead. Maxes out disk I/O and CPU.
"""

import os
import sys
import json
import time
import signal
import shutil
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pyarrow.parquet as pq
from tqdm import tqdm
from transformers import AutoTokenizer
from huggingface_hub import hf_hub_download, list_repo_files, HfApi

# ── Config ───────────────────────────────────────────────────────────────────
DATA_DIR = Path(os.environ.get("DATA_DIR", "/home/kenpeter/work/data"))
STAGING_DIR = DATA_DIR / "_staging"
CHECKPOINT_FILE = DATA_DIR / "prepare_checkpoint_local.json"
SHARD_SIZE = 1_024_000_000  # ~1GB uint16 tokens

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
    print("\n[CTRL+C] Saving checkpoint...")

signal.signal(signal.SIGINT, _sig_handler)
signal.signal(signal.SIGTERM, _sig_handler)

# ── Helpers ──────────────────────────────────────────────────────────────────

def load_checkpoint():
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE) as f:
            return json.load(f)
    return {"total_tokens": 0, "shards": 0, "dataset_idx": 0, "file_idx": 0}

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

def tokenize_batch(texts, tokenizer):
    if not texts:
        return []
    enc = tokenizer(
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
    parser.add_argument("--batch_size", type=int, default=1024)
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    STAGING_DIR.mkdir(parents=True, exist_ok=True)
    used_gb, total_gb = get_disk_usage_gb(DATA_DIR)
    avail_gb = total_gb - used_gb
    print(f"💾 Disk: {used_gb:.1f} GB used / {total_gb:.1f} GB total / {avail_gb:.1f} GB available")

    target_gb = args.max_gb or max(avail_gb - 20, 10)
    target_tokens = int(target_gb * (1024**3) / 2)
    print(f"🎯 Target: ~{target_gb:.1f} GB (~{target_tokens:,} tokens)")

    print("📥 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ckpt = load_checkpoint()
    existing_shards, existing_tokens = load_existing_shards()
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
    with ThreadPoolExecutor(max_workers=4) as executor:
        for ds_idx, (ds_name, ds_config, weight) in enumerate(DATASETS):
            if ds_idx < ckpt.get("dataset_idx", 0):
                continue
            if _shutdown_requested:
                break

            dataset_target = int(target_tokens * weight)
            dataset_tokens = 0
            file_idx = ckpt.get("file_idx", 0) if ds_idx == ckpt.get("dataset_idx", 0) else 0

            print(f"\n📚 Dataset: {ds_name}" + (f" ({ds_config})" if ds_config else ""))

            # List all parquet files for this dataset
            try:
                all_files = list(list_repo_files(ds_name, repo_type="dataset"))
            except Exception as e:
                print(f"   ⚠️  Failed to list files: {e}")
                continue

            parquet_files = sorted([f for f in all_files if f.endswith(".parquet")])
            if ds_config:
                # Filter to files containing the config name
                parquet_files = [f for f in parquet_files if ds_config.replace("-", "_").replace(".", "_") in f.replace("-", "_").replace(".", "_") or ds_config in f]
            print(f"   📁 {len(parquet_files)} parquet files found")

            for f_idx, repo_file in enumerate(parquet_files):
                if f_idx < file_idx:
                    continue
                if _shutdown_requested:
                    break

                # Download to staging
                local_path = None
                for attempt in range(5):
                    try:
                        local_path = hf_hub_download(
                            repo_id=ds_name,
                            filename=repo_file,
                            repo_type="dataset",
                            local_dir=STAGING_DIR,
                            local_dir_use_symlinks=False,
                        )
                        break
                    except Exception as e:
                        wait = min(2 ** attempt, 60)
                        print(f"   ⚠️  Download failed ({attempt+1}/5): {e}. Retrying in {wait}s...")
                        time.sleep(wait)

                if not local_path or not Path(local_path).exists():
                    print(f"   ❌ Failed to download {repo_file}, skipping...")
                    continue

                # Process with pyarrow (fast C++ reads)
                try:
                    pf = pq.ParquetFile(local_path)
                    for batch in pf.iter_batches(batch_size=args.batch_size, columns=["text"]):
                        if _shutdown_requested:
                            break

                        # Extract texts
                        texts = batch.column("text").to_pylist()
                        texts = [t for t in texts if t and isinstance(t, str)]
                        if not texts:
                            continue

                        # Tokenize in batch
                        token_ids_list = tokenize_batch(texts, tokenizer)

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
                                        "file_idx": f_idx,
                                    })

                                    if total_tokens >= target_tokens:
                                        break
                            if total_tokens >= target_tokens:
                                break
                except Exception as e:
                    print(f"   ⚠️  Error reading {repo_file}: {e}")

                # Delete downloaded file to save space
                try:
                    Path(local_path).unlink()
                    # Also clean empty parent dirs in staging
                    parent = Path(local_path).parent
                    while parent != STAGING_DIR and parent.exists() and not any(parent.iterdir()):
                        parent.rmdir()
                        parent = parent.parent
                except Exception as e:
                    pass

                if total_tokens >= target_tokens:
                    break
                if dataset_tokens >= dataset_target and dataset_target > 0:
                    print(f"   ✅ Reached dataset target ({dataset_tokens:,} tokens)")
                    break

            print(f"   → Got {dataset_tokens:,} tokens from {ds_name}")
            ckpt["dataset_idx"] = ds_idx + 1
            ckpt["file_idx"] = 0
            save_checkpoint(ckpt)

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
        "file_idx": 0,
    })

    # Cleanup staging
    try:
        shutil.rmtree(STAGING_DIR, ignore_errors=True)
    except:
        pass

    used_gb_after, _ = get_disk_usage_gb(DATA_DIR)
    print(f"\n✅ Done! Wrote {shard_id} shards, {total_tokens:,} tokens (~{total_tokens*2/(1024**3):.1f} GB)")
    print(f"💾 Disk usage: {used_gb_after:.1f} GB")

if __name__ == "__main__":
    main()
