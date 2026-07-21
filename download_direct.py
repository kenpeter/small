#!/usr/bin/env python3
"""Direct HTTP download using wget/curl - bypasses slow hf_hub_download."""
import os, time, fnmatch, subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

HF_TOKEN = os.environ.get("HF_TOKEN", "")
STAGING = Path("/home/kenpeter/work/data/_staging_multi")
STAGING.mkdir(parents=True, exist_ok=True)

# Only download fineweb-edu for now
DATASETS = [
    ("fineweb-edu", "HuggingFaceFW/fineweb-edu", "data/CC-MAIN-*/train-*.parquet", 200),
]

def list_files_api(repo, pattern, max_files=None):
    """List files using HF API directly."""
    import requests
    url = f"https://huggingface.co/api/datasets/{repo}/tree/main?recursive=true"
    headers = {}
    if HF_TOKEN:
        headers["Authorization"] = f"Bearer {HF_TOKEN}"
    
    try:
        resp = requests.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        files = [item["path"] for item in data if item.get("type") == "file" and item["path"].endswith(".parquet")]
        matched = [f for f in files if fnmatch.fnmatch(f, pattern)]
        if max_files:
            matched = matched[:max_files]
        return matched
    except Exception as e:
        print(f"API error: {e}")
        return []

def download_file(repo, remote_path, local_path):
    """Download using curl with resume support."""
    if local_path.exists() and local_path.stat().st_size > 100000:
        return "skip"
    
    url = f"https://huggingface.co/datasets/{repo}/resolve/main/{remote_path}"
    cmd = ["curl", "-sL", "--max-time", "120", "-o", str(local_path), url]
    if HF_TOKEN:
        cmd.extend(["-H", f"Authorization: Bearer {HF_TOKEN}"])
    
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=130)
        if result.returncode == 0 and local_path.exists() and local_path.stat().st_size > 100000:
            return "ok"
        return f"err: curl exit {result.returncode}"
    except subprocess.TimeoutExpired:
        return "err: timeout"
    except Exception as e:
        return f"err: {str(e)[:60]}"

def download_dataset(name, repo, pattern, max_files):
    print(f"\n[{name}] Listing files...", flush=True)
    files = list_files_api(repo, pattern, max_files)
    if not files:
        print(f"[{name}] No files found!")
        return
    
    out_dir = STAGING / name
    out_dir.mkdir(parents=True, exist_ok=True)
    total = len(files)
    
    print(f"[{name}] {total} files to download (3 workers, staggered)", flush=True)
    done, ok, skip = 0, 0, 0
    t0 = time.time()
    
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {}
        idx = 0
        
        while len(futures) < 6 and idx < total:
            f = files[idx]
            local = out_dir / f.split('/')[-1]
            if local.exists() and local.stat().st_size > 100000:
                skip += 1; done += 1; idx += 1; continue
            futures[pool.submit(download_file, repo, f, local)] = f
            idx += 1
            time.sleep(2)
        
        while futures:
            for future in as_completed(futures):
                f = futures.pop(future)
                result = future.result()
                done += 1
                if result == "ok": ok += 1
                elif result == "skip": skip += 1
                else: skip += 1
                
                if done % 5 == 0 or done == total:
                    elapsed = time.time() - t0
                    print(f"  [{name}] {done}/{total} | ok={ok} skip={skip} | {elapsed:.0f}s", flush=True)
                break
            
            while len(futures) < 6 and idx < total:
                f = files[idx]
                local = out_dir / f.split('/')[-1]
                if local.exists() and local.stat().st_size > 100000:
                    skip += 1; done += 1; idx += 1; continue
                futures[pool.submit(download_file, repo, f, local)] = f
                idx += 1
                time.sleep(2)
    
    elapsed = time.time() - t0
    print(f"[{name}] Done: {ok} ok, {skip} skip in {elapsed:.0f}s", flush=True)

def main():
    for name, repo, pattern, max_files in DATASETS:
        download_dataset(name, repo, pattern, max_files)

if __name__ == "__main__":
    main()
