"""
Fast parallel bulk download of HuggingFaceFW/fineweb-edu parquet files.
Uses wget for speed. Resumable.
"""
import os, sys, json, subprocess, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import huggingface_hub

REPO = "HuggingFaceFW/fineweb-edu"
STAGING = Path("/home/kenpeter/work/data/_staging_v2/HuggingFaceFW_fineweb-edu")
CHECKPOINT = STAGING / "download_checkpoint.json"
WORKERS = 4

def get_actual_files():
    files = huggingface_hub.list_repo_files(REPO, repo_type='dataset')
    parquet = sorted([f for f in files if f.endswith('.parquet') and 'sample/' not in f])
    return parquet

def load_checkpoint():
    if CHECKPOINT.exists():
        with open(CHECKPOINT) as f:
            return json.load(f)
    return {"downloaded": [], "failed": []}

def save_checkpoint(state):
    tmp = CHECKPOINT.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(state, f)
    os.replace(tmp, CHECKPOINT)

def download_one(url, out_path, log_path):
    try:
        result = subprocess.run(
            ["wget", "-c", "-q", "--show-progress", "-O", str(out_path), url],
            capture_output=True, text=True, timeout=1800
        )
        if result.returncode != 0:
            if out_path.exists():
                out_path.unlink()
            result = subprocess.run(
                ["wget", "-q", "--show-progress", "-O", str(out_path), url],
                capture_output=True, text=True, timeout=1800
            )
        if result.returncode == 0 and out_path.exists() and out_path.stat().st_size > 1024:
            with open(log_path, "a") as f:
                f.write(f"OK {url}\n")
            return True
        else:
            with open(log_path, "a") as f:
                f.write(f"FAIL {url}\n")
            return False
    except Exception as e:
        with open(log_path, "a") as f:
            f.write(f"ERR {url} {e}\n")
        return False

def main():
    STAGING.mkdir(parents=True, exist_ok=True)
    state = load_checkpoint()
    downloaded = set(state.get("downloaded", []))
    failed = set(state.get("failed", []))

    all_files = get_actual_files()
    print(f"Total files on HF: {len(all_files)}")

    # Filter out already downloaded
    todo = [f for f in all_files if f not in downloaded and f not in failed]
    print(f"Already downloaded: {len(downloaded)}")
    print(f"Already failed: {len(failed)}")
    print(f"Remaining to download: {len(todo)}")

    # Check existing files in staging
    existing_on_disk = set()
    for f in STAGING.glob("*.parquet"):
        # Map back to HF path: data_CC-MAIN-..._train-...of....parquet -> data/.../train-...of....parquet
        name = f.name
        # Remove data_ prefix and replace first _ with /
        if name.startswith("data_"):
            name = name[5:]
            parts = name.split("_train-")
            if len(parts) == 2:
                hf_path = f"data/{parts[0]}/train-{parts[1]}"
                existing_on_disk.add(hf_path)

    print(f"Already on disk: {len(existing_on_disk)}")
    todo = [f for f in todo if f not in existing_on_disk]
    print(f"Actual remaining: {len(todo)}")

    if not todo:
        print("Nothing to download!")
        return

    log_path = STAGING / "download.log"
    total = len(todo)
    completed = 0
    t0 = time.time()

    print(f"\n🚀 Starting bulk download with {WORKERS} workers...")
    print(f"   Target: {total} files")

    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        futures = {}
        idx = 0
        # Submit first batch
        while idx < min(WORKERS * 2, total):
            fname = todo[idx]
            url = f"https://huggingface.co/datasets/{REPO}/resolve/main/{fname}"
            out = STAGING / fname.replace("/", "_")
            if not out.exists() or out.stat().st_size < 1024:
                future = executor.submit(download_one, url, out, log_path)
                futures[future] = fname
            else:
                downloaded.add(fname)
                completed += 1
            idx += 1

        while futures:
            for future in as_completed(futures):
                fname = futures.pop(future)
                success = future.result()
                completed += 1
                if success:
                    downloaded.add(fname)
                else:
                    failed.add(fname)

                if completed % 10 == 0 or completed == total:
                    elapsed = time.time() - t0
                    rate = completed / elapsed if elapsed > 0 else 0
                    print(f"   [{completed}/{total}] {rate:.2f} files/s | downloaded={len(downloaded)} failed={len(failed)}")
                    save_checkpoint({"downloaded": list(downloaded), "failed": list(failed)})
                break

            # Submit next
            if idx < total:
                fname = todo[idx]
                url = f"https://huggingface.co/datasets/{REPO}/resolve/main/{fname}"
                out = STAGING / fname.replace("/", "_")
                if not out.exists() or out.stat().st_size < 1024:
                    future = executor.submit(download_one, url, out, log_path)
                    futures[future] = fname
                else:
                    downloaded.add(fname)
                    completed += 1
                idx += 1

    save_checkpoint({"downloaded": list(downloaded), "failed": list(failed)})
    elapsed = time.time() - t0
    print(f"\n✅ Done: {len(downloaded)} downloaded, {len(failed)} failed in {elapsed:.0f}s")

if __name__ == "__main__":
    main()
