"""
Fast data prep using wget for direct HTTP download (8 MB/s vs 1.5 MB/s with requests).
Parallel download + immediate processing. Resumable.
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
MAX_TOKENS     = int(MAX_GB * 1_073_741_824 // 2)

DATA_DIR       = Path("/home/kenpeter/work/data")
STAGING_DIR    = DATA_DIR / "_staging_wget"
CHECKPOINT     = DATA_DIR / "prepare_checkpoint_wget.json"

# ─── Datasets ──────────────────────────────────────────────────────
DATASETS = [
    {"repo": "HuggingFaceFW/fineweb-edu",      "subdir": "data/CC-MAIN-2013-20", "pattern": "train-{i:05d}-of-00014.parquet", "count": 15, "target_gb": 110},
    {"repo": "mlfoundations/dclm-baseline-1.0", "subdir": "",                    "pattern": "train-{i:05d}-of-00041.parquet", "count": 42, "target_gb": 44},
    {"repo": "bigcode/the-stack-dedup",        "subdir": "data/python",         "pattern": "train-{i:05d}-of-00198.parquet", "count": 199, "target_gb": 22},
    {"repo": "HuggingFaceTB/finemath",         "subdir": "finemath-3plus",      "pattern": "train-{i:05d}-of-00009.parquet", "count": 10, "target_gb": 22},
    {"repo": "OpenCoder-LLM/InfIMMCorpus",     "subdir": "webmath",             "pattern": "train-{i:05d}-of-00004.parquet", "count": 5,  "target_gb": 11},
    {"repo": "HuggingFaceTB/cosmopedia",       "subdir": "stanford",            "pattern": "train-{i:05d}-of-00007.parquet", "count": 8,  "target_gb": 11},
]

def hf_url(repo, subdir, filename):
    base = f"https://huggingface.co/datasets/{repo}/resolve/main"
    if subdir:
        return f"{base}/{subdir}/{filename}"
    return f"{base}/{filename}"

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
    """Download with wget -c (resume). Returns True on success."""
    try:
        # wget -c resumes, -q quiet, -O output
        # Timeout after 30 min per file
        result = subprocess.run(
            ["wget", "-c", "-q", "-O", str(out_path), url],
            capture_output=True, text=True, timeout=1800
        )
        if result.returncode != 0:
            # If failed, try once more without resume
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
    print(f"🚀 wget direct download (much faster than Xet)")

    try:
        while dataset_idx < len(DATASETS) and total_tokens < MAX_TOKENS:
            ds = DATASETS[dataset_idx]
            print(f"\n📦 {ds['repo']}  (target {ds['target_gb']} GB tokenized)")

            files = [ds["pattern"].format(i=i) for i in range(ds["count"])]
            print(f"   {len(files)} files")

            staging_sub = STAGING_DIR / ds["repo"].replace("/", "_")
            staging_sub.mkdir(parents=True, exist_ok=True)

            if file_idx >= len(files):
                dataset_idx += 1
                file_idx = 0
                save_checkpoint({"total_tokens": total_tokens, "shards": shards, "dataset_idx": dataset_idx, "file_idx": file_idx})
                continue

            active = {}
            workers = 3  # 3 parallel wget downloads
            with ThreadPoolExecutor(max_workers=workers) as executor:
                for _ in range(workers):
                    if file_idx < len(files):
                        fname = files[file_idx]
                        url = hf_url(ds["repo"], ds["subdir"], fname)
                        out = staging_sub / fname
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
                        print(f"   ⚠️ Failed {fname}, skipping")

                    if file_idx < len(files):
                        fname_next = files[file_idx]
                        url_next = hf_url(ds["repo"], ds["subdir"], fname_next)
                        out_next = staging_sub / fname_next
                        f_next = executor.submit(download_wget, url_next, out_next)
                        active[f_next] = (file_idx, fname_next, out_next)
                        file_idx += 1

            dataset_idx += 1
            file_idx = 0
            save_checkpoint({"total_tokens": total_tokens, "shards": shards, "dataset_idx": dataset_idx, "file_idx": file_idx})

    except KeyboardInterrupt:
        print("\n⏹ Interrupted")
    finally:
        if buf_pos > 0:
            shard_path = DATA_DIR / f"shard_{shards:06d}.bin"
            buffer[:buf_pos].tofile(shard_path)
            shards += 1
            total_tokens += buf_pos
            save_checkpoint({"total_tokens": total_tokens, "shards": shards, "dataset_idx": dataset_idx, "file_idx": file_idx})
            print(f"💾 Flushed {buf_pos:,} tokens")

    print(f"\n🏁 Done: {total_tokens:,} tokens, {shards} shards")

if __name__ == "__main__":
    main()
