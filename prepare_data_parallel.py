#!/usr/bin/env python3
"""
PARALLEL-DOWNLOAD data preparation for SmolLM2-135M.
Downloads 4 parquet files concurrently, processes sequentially.
Maximizes bandwidth usage + keeps resume robust.
"""

import os
import sys
import json
import time
import signal
import shutil
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pyarrow.parquet as pq
from tqdm import tqdm
from transformers import AutoTokenizer
from huggingface_hub import list_repo_files, hf_hub_download, HfApi

# ── Config ───────────────────────────────────────────────────────────────────
DATA_DIR = Path(os.environ.get("DATA_DIR", "/home/kenpeter/work/data"))
STAGING_DIR = DATA_DIR / "_staging"
CHECKPOINT_FILE = DATA_DIR / "prepare_checkpoint_parallel.json"
SHARD_SIZE = 1_024_000_000  # ~1GB uint16 tokens
CONCURRENT_DOWNLOADS = 4    # download this many files at once

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

def download_one_file(ds_name, repo_file, local_dir, max_retries=5):
    """Download a single parquet file with retry."""
    for attempt in range(max_retries):
        try:
            local_path = hf_hub_download(
                repo_id=ds_name,
                filename=repo_file,
                repo_type="dataset",
                local_dir=local_dir,
                local_dir_use_symlinks=False,
            )
            return local_path
        except Exception as e:
            wait = min(2 ** attempt, 60)
            print(f"   ⚠️  DL {repo_file} failed ({attempt+1}/{max_retries}): {e}. Retry {wait}s...")
            time.sleep(wait)
    return None

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

    for ds_idx, (ds_name, ds_config, weight) in enumerate(DATASETS):
        if ds_idx < ckpt.get("dataset_idx", 0):
            continue
        if _shutdown_requested:
            break

        dataset_target = int(target_tokens * weight)
        dataset_tokens = 0
        file_idx = ckpt.get("file_idx", 0) if ds_idx == ckpt.get("dataset_idx", 0) else 0

        print(f"\n📚 Dataset: {ds_name}" + (f" ({ds_config})" if ds_config else ""))

        try:
            all_files = list(list_repo_files(ds_name, repo_type="dataset"))
        except Exception as e:
            print(f"   ⚠️  Failed to list files: {e}")
            continue

        parquet_files = sorted([f for f in all_files if f.endswith(".parquet")])
        if ds_config:
            cfg_key = ds_config.replace("-", "_").replace(".", "_")
            parquet_files = [f for f in parquet_files if cfg_key in f.replace("-", "_").replace(".", "_") or ds_config in f]
        print(f"   📁 {len(parquet_files)} parquet files. Starting at file {file_idx}.")

        # Process in batches: download CONCURRENT_DOWNLOADS files in parallel, then process sequentially
        remaining_files = parquet_files[file_idx:]
        batch_size = CONCURRENT_DOWNLOADS

        for batch_start in range(0, len(remaining_files), batch_size):
            if _shutdown_requested:
                break
            batch = remaining_files[batch_start:batch_start + batch_size]
            actual_batch_idx = file_idx + batch_start

            print(f"   📥 Downloading batch {batch_start//batch_size + 1}: {len(batch)} files...")

            # Parallel download
            downloaded_paths = {}
            with ThreadPoolExecutor(max_workers=len(batch)) as dl_pool:
                futures = {
                    dl_pool.submit(download_one_file, ds_name, repo_file, STAGING_DIR): repo_file
                    for repo_file in batch
                }
                for future in as_completed(futures):
                    repo_file = futures[future]
                    local_path = future.result()
                    if local_path:
                        downloaded_paths[repo_file] = local_path
                        print(f"      ✅ {Path(repo_file).name} ({os.path.getsize(local_path)/(1024**2):.1f} MB)")
                    else:
                        print(f"      ❌ Failed: {repo_file}")

            # Process sequentially (in original order for deterministic resume)
            for repo_file in batch:
                if _shutdown_requested:
                    break
                if repo_file not in downloaded_paths:
                    file_idx += 1
                    continue

                local_path = downloaded_paths[repo_file]
                print(f"      🔧 Processing {Path(repo_file).name}...")

                try:
                    pf = pq.ParquetFile(local_path)
                    for arrow_batch in pf.iter_batches(batch_size=args.batch_size, columns=["text"]):
                        if _shutdown_requested:
                            break

                        texts = arrow_batch.column("text").to_pylist()
                        texts = [t for t in texts if t and isinstance(t, str)]
                        if not texts:
                            continue

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
                                        "file_idx": file_idx,
                                    })

                                    if total_tokens >= target_tokens:
                                        break
                            if total_tokens >= target_tokens:
                                break
                except Exception as e:
                    print(f"      ⚠️  Error reading {repo_file}: {e}")

                # Delete downloaded file to save space
                try:
                    Path(local_path).unlink()
                    # Clean empty dirs
                    parent = Path(local_path).parent
                    while parent != STAGING_DIR and parent.exists() and not any(parent.iterdir()):
                        parent.rmdir()
                        parent = parent.parent
                except Exception:
                    pass

                file_idx += 1

                if total_tokens >= target_tokens:
                    break
                if dataset_tokens >= dataset_target and dataset_target > 0:
                    print(f"   ✅ Reached dataset target ({dataset_tokens:,} tokens)")
                    break

            if total_tokens >= target_tokens:
                break
            if dataset_tokens >= dataset_target and dataset_target > 0:
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

    try:
        shutil.rmtree(STAGING_DIR, ignore_errors=True)
    except:
        pass

    used_gb_after, _ = get_disk_usage_gb(DATA_DIR)
    print(f"\n✅ Done! Wrote {shard_id} shards, {total_tokens:,} tokens (~{total_tokens*2/(1024**3):.1f} GB)")
    print(f"💾 Disk usage: {used_gb_after:.1f} GB")

if __name__ == "__main__":
    main()
