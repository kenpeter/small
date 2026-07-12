"""
Robust data prep with HuggingFace Xet high-performance mode.
Parallel download + sequential processing. Resumable.
"""
import os, sys, json, time, math, gc, glob
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import numpy as np
import pyarrow.parquet as pq
from transformers import AutoTokenizer
from huggingface_hub import hf_hub_download, list_repo_files

# ─── Config ──────────────────────────────────────────────────────
TOKENIZER_NAME = "HuggingFaceTB/SmolLM2-135M"
MAX_SEQ_LEN    = 8192
SHARD_SIZE     = int(1_073_741_824)     # 1 GiB tokens → ~2 GB on disk (uint16)
MAX_GB         = int(sys.argv[1]) if len(sys.argv) > 1 else 220
MAX_TOKENS     = int(MAX_GB * 1_073_741_824 // 2)

DATA_DIR       = Path("/home/kenpeter/work/data")
STAGING_DIR    = DATA_DIR / "_staging"
CHECKPOINT     = DATA_DIR / "prepare_checkpoint_xet.json"

# FineWeb-Edu (50%) + DCLM (20%) + Stack-Edu (10%) + FineMath (10%) + Infimm-WebMath (5%) + Cosmopedia (5%)
# For 220 GB tokenized, target per dataset in tokens:
DATASETS = [
    {"repo": "HuggingFaceFW/fineweb-edu",      "subset": "data/CC-MAIN-2013-20", "target_gb": 110, "pattern": "data/CC-MAIN-2013-20/*.parquet"},
    {"repo": "mlfoundations/dclm-baseline-1.0", "subset": "",                    "target_gb": 44,  "pattern": "*.parquet"},
    {"repo": "bigcode/the-stack-dedup",        "subset": "data/python",         "target_gb": 22,  "pattern": "data/python/*.parquet"},
    {"repo": "HuggingFaceTB/finemath",         "subset": "finemath-3plus",      "target_gb": 22,  "pattern": "finemath-3plus/*.parquet"},
    {"repo": "OpenCoder-LLM/InfIMMCorpus",     "subset": "webmath",             "target_gb": 11,  "pattern": "webmath/*.parquet"},
    {"repo": "HuggingFaceTB/cosmopedia",       "subset": "stanford",            "target_gb": 11,  "pattern": "stanford/*.parquet"},
]

# ─── Helpers ─────────────────────────────────────────────────────
def load_checkpoint():
    if CHECKPOINT.exists():
        with open(CHECKPOINT) as f:
            return json.load(f)
    # Start from existing 15 shards
    existing = sorted(DATA_DIR.glob("shard_*.bin"))
    total_tokens = len(existing) * SHARD_SIZE
    return {
        "total_tokens": total_tokens,
        "shards": len(existing),
        "dataset_idx": 0,
        "file_idx": 0,
    }

def save_checkpoint(state):
    tmp = CHECKPOINT.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(state, f)
    os.replace(tmp, CHECKPOINT)

def get_repo_files(repo_id, pattern):
    """List files in a repo matching a glob pattern."""
    files = []
    try:
        all_files = list_repo_files(repo_id, repo_type="dataset")
        prefix = pattern.replace("*.parquet", "").rstrip("/")
        files = [f for f in all_files if f.endswith(".parquet") and f.startswith(prefix)]
        files.sort()
    except Exception as e:
        print(f"⚠️ list_repo_files failed for {repo_id}: {e}")
    return files

def download_file(repo_id, filename, local_dir):
    """Download one file; returns local path or None on failure."""
    try:
        return hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="dataset",
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            resume_download=True,
        )
    except Exception as e:
        print(f"  ❌ Download failed: {repo_id}/{filename} → {e}")
        return None

# ─── Main ───────────────────────────────────────────────────────
def main():
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    state = load_checkpoint()
    buffer = np.empty(SHARD_SIZE, dtype=np.uint16)
    buf_pos = 0
    total_tokens = state["total_tokens"]
    shards = state["shards"]
    dataset_idx = state["dataset_idx"]
    file_idx = state["file_idx"]

    print(f"🔁 Resuming: dataset={dataset_idx}, file={file_idx}, shards={shards}, tokens={total_tokens:,}")
    print(f"🚀 Xet high-performance mode: {os.environ.get('HF_XET_HIGH_PERFORMANCE')}")

    try:
        while dataset_idx < len(DATASETS) and total_tokens < MAX_TOKENS:
            ds = DATASETS[dataset_idx]
            print(f"\n📦 Dataset {dataset_idx}: {ds['repo']}  (target {ds['target_gb']} GB)")

            # Get file list (once)
            if file_idx == 0:
                files = get_repo_files(ds["repo"], ds["pattern"])
                print(f"   Found {len(files)} parquet files")
            else:
                # On resume we already know files
                files = get_repo_files(ds["repo"], ds["pattern"])

            if not files:
                print("   ⚠️ No files found, skipping dataset")
                dataset_idx += 1
                file_idx = 0
                save_checkpoint({"total_tokens": total_tokens, "shards": shards, "dataset_idx": dataset_idx, "file_idx": file_idx})
                continue

            # Download in parallel, process as they land
            staging_sub = STAGING_DIR / ds["repo"].replace("/", "_")
            staging_sub.mkdir(parents=True, exist_ok=True)
            os.environ["HF_HOME"] = str(staging_sub / ".cache")
            os.environ["HF_HUB_CACHE"] = str(staging_sub / ".cache")

            active = {}
            with ThreadPoolExecutor(max_workers=3) as executor:
                # Seed first 3 downloads
                for _ in range(3):
                    if file_idx < len(files):
                        f = files[file_idx]
                        future = executor.submit(download_file, ds["repo"], f, str(staging_sub))
                        active[future] = (file_idx, f)
                        file_idx += 1

                while active or file_idx < len(files):
                    # Wait for at least one download to finish
                    done_futures = []
                    for future in as_completed(active):
                        done_futures.append(future)
                        break  # Just grab the first completed one

                    for future in done_futures:
                        idx, fname = active.pop(future)
                        local_path = future.result()

                        if local_path is None:
                            continue

                        # Tokenize this file
                        print(f"   ⚡ Processing {fname} ({idx+1}/{len(files)})")
                        try:
                            table = pq.read_table(local_path)
                            if "text" in table.column_names:
                                texts = table.column("text").to_pylist()
                            else:
                                texts = [str(r) for r in table.to_pandas().iloc[:, 0].tolist()]

                            for text in texts:
                                if not text:
                                    continue
                                ids = tokenizer.encode(text, add_special_tokens=False, truncation=False)
                                if not ids:
                                    continue
                                ids_arr = np.array(ids, dtype=np.uint16)
                                n = len(ids_arr)
                                if n == 0:
                                    continue

                                # Fill buffer
                                start = 0
                                while start < n:
                                    remaining = SHARD_SIZE - buf_pos
                                    chunk = ids_arr[start:start+remaining]
                                    buffer[buf_pos:buf_pos+len(chunk)] = chunk
                                    buf_pos += len(chunk)
                                    start += len(chunk)

                                    if buf_pos >= SHARD_SIZE:
                                        # Write shard
                                        shard_path = DATA_DIR / f"shard_{shards:06d}.bin"
                                        buffer.tofile(shard_path)
                                        shards += 1
                                        total_tokens += SHARD_SIZE
                                        buf_pos = 0
                                        save_checkpoint({"total_tokens": total_tokens, "shards": shards, "dataset_idx": dataset_idx, "file_idx": idx+1})
                                        if total_tokens >= MAX_TOKENS:
                                            print(f"\n✅ Target reached: {total_tokens:,} tokens ({shards} shards)")
                                            return

                        except Exception as e:
                            print(f"   ❌ Error processing {fname}: {e}")

                        # Delete processed file to save disk
                        try:
                            Path(local_path).unlink()
                        except Exception:
                            pass

                        # Queue next download
                        if file_idx < len(files):
                            f_next = files[file_idx]
                            future_next = executor.submit(download_file, ds["repo"], f_next, str(staging_sub))
                            active[future_next] = (file_idx, f_next)
                            file_idx += 1

            # Move to next dataset
            dataset_idx += 1
            file_idx = 0
            save_checkpoint({"total_tokens": total_tokens, "shards": shards, "dataset_idx": dataset_idx, "file_idx": file_idx})

    except KeyboardInterrupt:
        print("\n⏹ Interrupted by user")
    finally:
        # Flush any partial buffer
        if buf_pos > 0:
            shard_path = DATA_DIR / f"shard_{shards:06d}.bin"
            buffer[:buf_pos].tofile(shard_path)
            shards += 1
            total_tokens += buf_pos
            save_checkpoint({"total_tokens": total_tokens, "shards": shards, "dataset_idx": dataset_idx, "file_idx": file_idx})
            print(f"💾 Flushed partial shard ({buf_pos:,} tokens)")

    print(f"\n🏁 Done: {total_tokens:,} tokens written across {shards} shards")

if __name__ == "__main__":
    main()
