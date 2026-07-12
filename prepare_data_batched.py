"""
Fast data prep: wget download + batched tokenization.
Resumable, parallel downloads, much faster processing.
"""
import os, sys, json, time, gc, subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pyarrow.parquet as pq
from transformers import AutoTokenizer

# ─── Config ──────────────────────────────────────────────────────
TOKENIZER_NAME = "HuggingFaceTB/SmolLM2-135M"
MAX_SEQ_LEN    = 8192
SHARD_SIZE     = int(1_073_741_824)     # 1 GiB tokens → ~2 GB on disk (uint16)
MAX_GB         = int(sys.argv[1]) if len(sys.argv) > 1 else 220
MAX_TOKENS = int(MAX_GB * 1_073_741_824 // 2)

DATA_DIR       = Path("/home/kenpeter/work/data")
STAGING_DIR    = DATA_DIR / "_staging_batched"
CHECKPOINT     = DATA_DIR / "prepare_checkpoint_batched.json"

# FineWeb-Edu has 50 snapshots. Each has 15 files. Use first 8 snapshots = 120 files.
# We process until MAX_TOKENS is reached, so we may not need all 120.
FINEWEB_SNAPSHOTS = [
    "CC-MAIN-2013-20", "CC-MAIN-2013-48", "CC-MAIN-2014-10", "CC-MAIN-2014-15",
    "CC-MAIN-2014-23", "CC-MAIN-2014-35", "CC-MAIN-2014-41", "CC-MAIN-2014-42",
]

DATASETS = [
    {
        "repo": "HuggingFaceFW/fineweb-edu",
        "files": [f"data/{snap}/train-{i:05d}-of-00014.parquet" for snap in FINEWEB_SNAPSHOTS for i in range(15)],
        "target_gb": 110,
    },
    {
        "repo": "mlfoundations/dclm-baseline-1.0",
        "files": [f"train-{i:05d}-of-00041.parquet" for i in range(42)],
        "target_gb": 44,
    },
    {
        "repo": "bigcode/the-stack-dedup",
        "files": [f"data/python/train-{i:05d}-of-00198.parquet" for i in range(199)],
        "target_gb": 22,
    },
    {
        "repo": "HuggingFaceTB/finemath",
        "files": [f"finemath-3plus/train-{i:05d}-of-00009.parquet" for i in range(10)],
        "target_gb": 22,
    },
    {
        "repo": "OpenCoder-LLM/InfIMMCorpus",
        "files": [f"webmath/train-{i:05d}-of-00004.parquet" for i in range(5)],
        "target_gb": 11,
    },
    {
        "repo": "HuggingFaceTB/cosmopedia",
        "files": [f"stanford/train-{i:05d}-of-00007.parquet" for i in range(8)],
        "target_gb": 11,
    },
]

def hf_url(repo, filename):
    return f"https://huggingface.co/datasets/{repo}/resolve/main/{filename}"

def load_checkpoint():
    if CHECKPOINT.exists():
        with open(CHECKPOINT) as f:
            return json.load(f)
    existing = sorted(DATA_DIR.glob("shard_*.bin"))
    return {"total_tokens": len(existing) * SHARD_SIZE, "shards": len(existing), "dataset_idx": 0, "file_idx": 0}

def save_checkpoint(state):
    tmp = CHECKPOINT.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(state, f)
    os.replace(tmp, CHECKPOINT)

def download_wget(url, out_path):
    try:
        result = subprocess.run(
            ["wget", "-c", "-q", "-O", str(out_path), url],
            capture_output=True, text=True, timeout=1800
        )
        if result.returncode != 0:
            if out_path.exists():
                out_path.unlink()
            result = subprocess.run(
                ["wget", "-q", "-O", str(out_path), url],
                capture_output=True, text=True, timeout=1800
            )
        return result.returncode == 0
    except Exception as e:
        print(f"  ❌ wget error: {e}")
        return False

def flush_buffer(buffer, buf_pos, shards, total_tokens, dataset_idx, file_idx):
    if buf_pos > 0:
        shard_path = DATA_DIR / f"shard_{shards:06d}.bin"
        buffer[:buf_pos].tofile(shard_path)
        shards += 1
        total_tokens += buf_pos
        save_checkpoint({"total_tokens": total_tokens, "shards": shards, "dataset_idx": dataset_idx, "file_idx": file_idx})
    return shards, total_tokens, 0

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

    print(f"🔁 Resume: ds={dataset_idx} file={file_idx} shards={shards} tokens={total_tokens:,}")
    print(f"🚀 wget + batched tokenization")

    try:
        while dataset_idx < len(DATASETS) and total_tokens < MAX_TOKENS:
            ds = DATASETS[dataset_idx]
            print(f"\n📦 {ds['repo']}  (target {ds['target_gb']} GB)")

            files = ds["files"]
            print(f"   {len(files)} files available")

            staging_sub = STAGING_DIR / ds["repo"].replace("/", "_")
            staging_sub.mkdir(parents=True, exist_ok=True)

            if file_idx >= len(files):
                dataset_idx += 1
                file_idx = 0
                save_checkpoint({"total_tokens": total_tokens, "shards": shards, "dataset_idx": dataset_idx, "file_idx": file_idx})
                continue

            active = {}
            workers = 3
            with ThreadPoolExecutor(max_workers=workers) as executor:
                for _ in range(workers):
                    if file_idx < len(files):
                        fname = files[file_idx]
                        url = hf_url(ds["repo"], fname)
                        out = staging_sub / fname.replace("/", "_")
                        future = executor.submit(download_wget, url, out)
                        active[future] = (file_idx, fname, out)
                        file_idx += 1

                while active or file_idx < len(files):
                    for future in as_completed(active):
                        idx, fname, out = active.pop(future)
                        success = future.result()
                        break

                    if success and out.exists():
                        size_mb = out.stat().st_size / (1024*1024)
                        print(f"   ⚡ {fname} ({size_mb:.1f} MB) → processing")
                        try:
                            table = pq.read_table(str(out))
                            col = table.column("text") if "text" in table.column_names else table.column(0)
                            texts = [str(t) for t in col.to_pylist() if t]

                            # Batched tokenization
                            batch_size = 1000
                            for b_start in range(0, len(texts), batch_size):
                                batch = texts[b_start:b_start+batch_size]
                                encoded = tokenizer(batch, add_special_tokens=False, truncation=False, max_length=None)
                                for ids in encoded["input_ids"]:
                                    if not ids:
                                        continue
                                    ids_arr = np.array(ids, dtype=np.uint16)
                                    n = len(ids_arr)
                                    if n == 0:
                                        continue
                                    start = 0
                                    while start < n:
                                        remaining = SHARD_SIZE - buf_pos
                                        chunk = ids_arr[start:start+remaining]
                                        buffer[buf_pos:buf_pos+len(chunk)] = chunk
                                        buf_pos += len(chunk)
                                        start += len(chunk)
                                        if buf_pos >= SHARD_SIZE:
                                            shard_path = DATA_DIR / f"shard_{shards:06d}.bin"
                                            buffer.tofile(shard_path)
                                            shards += 1
                                            total_tokens += SHARD_SIZE
                                            buf_pos = 0
                                            save_checkpoint({"total_tokens": total_tokens, "shards": shards, "dataset_idx": dataset_idx, "file_idx": idx+1})
                                            if total_tokens >= MAX_TOKENS:
                                                print(f"\n✅ Target: {total_tokens:,} tokens, {shards} shards")
                                                return

                        except Exception as e:
                            print(f"   ❌ Process error {fname}: {e}")
                        finally:
                            try:
                                out.unlink()
                            except Exception:
                                pass
                    else:
                        print(f"   ⚠️ Failed {fname}")

                    if file_idx < len(files):
                        fname_next = files[file_idx]
                        url_next = hf_url(ds["repo"], fname_next)
                        out_next = staging_sub / fname_next.replace("/", "_")
                        f_next = executor.submit(download_wget, url_next, out_next)
                        active[f_next] = (file_idx, fname_next, out_next)
                        file_idx += 1

            dataset_idx += 1
            file_idx = 0
            save_checkpoint({"total_tokens": total_tokens, "shards": shards, "dataset_idx": dataset_idx, "file_idx": file_idx})

    except KeyboardInterrupt:
        print("\n⏹ Interrupted")
    finally:
        shards, total_tokens, buf_pos = flush_buffer(buffer, buf_pos, shards, total_tokens, dataset_idx, file_idx)
        print(f"💾 Flushed {buf_pos:,} tokens")

    print(f"\n🏁 Done: {total_tokens:,} tokens, {shards} shards")

if __name__ == "__main__":
    main()
