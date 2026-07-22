#!/usr/bin/env python3
"""Parallel download with 3 workers using direct HTTP/curl."""
import os, time, json, subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

HF_TOKEN = os.environ.get("HF_TOKEN", "")
STAGING = Path("/home/kenpeter/work/data/_staging_multi")
STAGING.mkdir(parents=True, exist_ok=True)

DATASETS = [
    ("fineweb-edu", "HuggingFaceFW/fineweb-edu", "data/CC-MAIN-*/train-*.parquet", 200),
    ("finemath-3plus", "HuggingFaceTB/finemath", "finemath-3plus/train-*.parquet", 200),
    ("cosmopedia", "HuggingFaceTB/cosmopedia", "data/stanford/train-*.parquet", 200),
    ("open-web-math", "open-web-math/open-web-math", "data/train-*.parquet", 200),
]

def list_files(repo, pattern, max_files=200):
    """List files from HF API."""
    import urllib.request
    url = f"https://huggingface.co/api/datasets/{repo}/tree/main?recursive=true"
    headers = {}
    if HF_TOKEN:
        headers["Authorization"] = f"Bearer {HF_TOKEN}"
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read())
    files = [item["path"] for item in data
             if item.get("type") == "file"
             and item["path"].endswith(".parquet")
             and "train-" in item["path"]]
    prefix = pattern.split("*")[0]
    matched = [f for f in files if f.startswith(prefix)]
    return matched[:max_files]

def download_file(repo, remote_path, local_path):
    """Download using curl with 3 retries."""
    if local_path.exists() and local_path.stat().st_size > 100000:
        return "skip"
    url = f"https://huggingface.co/datasets/{repo}/resolve/main/{remote_path}"
    cmd = ["curl", "-sL", "--max-time", "300", "-o", str(local_path)]
    if HF_TOKEN:
        cmd.extend(["-H", f"Authorization: Bearer {HF_TOKEN}"])
    cmd.append(url)
    for attempt in range(3):
        try:
            result = subprocess.run(cmd, capture_output=True, timeout=310)
            if result.returncode == 0 and local_path.exists() and local_path.stat().st_size > 100000:
                return "ok"
            if attempt < 2:
                time.sleep(5)
        except subprocess.TimeoutExpired:
            if attempt < 2:
                time.sleep(5)
    return f"err: curl exit after 3 retries"

def download_dataset(name, repo, pattern, max_files):
    print(f"\n[{name}] Listing files...", flush=True)
    files = list_files(repo, pattern, max_files)
    total = len(files)
    if total == 0:
        print(f"[{name}] No files found", flush=True)
        return
    print(f"[{name}] {total} files to download (3 workers, queue depth 8, 1.5s stagger)", flush=True)

    out_dir = STAGING / name
    out_dir.mkdir(parents=True, exist_ok=True)

    done, ok, skip, fail = 0, 0, 0, 0
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {}
        idx = 0

        # Seed initial queue
        while len(futures) < 8 and idx < total:
            f = files[idx]
            local = out_dir / f.split("/")[-1]
            if local.exists() and local.stat().st_size > 100000:
                skip += 1; done += 1; idx += 1; continue
            futures[pool.submit(download_file, repo, f, local)] = f
            idx += 1
            time.sleep(1.5)  # stagger to keep network usable

        while futures:
            for future in as_completed(futures):
                f = futures.pop(future)
                result = future.result()
                done += 1
                if result == "ok": ok += 1
                elif result == "skip": skip += 1
                else:
                    fail += 1
                    print(f"  {result}", flush=True)
                if done % 5 == 0 or done == total:
                    elapsed = time.time() - t0
                    print(f"  [{name}] {done}/{total} | ok={ok} skip={skip} fail={fail} | {elapsed:.0f}s", flush=True)
                break

            # Refill queue
            while len(futures) < 8 and idx < total:
                f = files[idx]
                local = out_dir / f.split("/")[-1]
                if local.exists() and local.stat().st_size > 100000:
                    skip += 1; done += 1; idx += 1; continue
                futures[pool.submit(download_file, repo, f, local)] = f
                idx += 1
                time.sleep(1.5)

    elapsed = time.time() - t0
    print(f"[{name}] Done: {ok} ok, {skip} skip, {fail} fail in {elapsed:.0f}s", flush=True)

def main():
    for name, repo, pattern, max_files in DATASETS:
        download_dataset(name, repo, pattern, max_files)

if __name__ == "__main__":
    main()
