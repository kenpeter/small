#!/usr/bin/env python3
"""Fast sequential download using direct HTTP."""
import os, time, json, subprocess
from pathlib import Path

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
    matched = [f for f in files if pattern in f or f.startswith(pattern.split('*')[0])]
    return matched[:max_files]

def download_file(repo, remote_path, local_path):
    """Download using curl."""
    if local_path.exists() and local_path.stat().st_size > 100000:
        return "skip"
    
    url = f"https://huggingface.co/datasets/{repo}/resolve/main/{remote_path}"
    cmd = ["curl", "-sL", "--max-time", "300", "-o", str(local_path)]
    if HF_TOKEN:
        cmd.extend(["-H", f"Authorization: Bearer {HF_TOKEN}"])
    cmd.append(url)
    
    # Retry up to 3 times
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
    return f"err: curl exit {result.returncode} after 3 retries"

def download_dataset(name, repo, pattern, max_files):
    print(f"\n[{name}] Listing files...", flush=True)
    files = list_files(repo, pattern, max_files)
    total = len(files)
    if total == 0:
        print(f"[{name}] No files found", flush=True)
        return
    print(f"[{name}] {total} files to download", flush=True)
    
    out_dir = STAGING / name
    out_dir.mkdir(parents=True, exist_ok=True)
    
    done, ok, skip, fail = 0, 0, 0, 0
    t0 = time.time()
    
    for i, remote_path in enumerate(files):
        filename = remote_path.split("/")[-1]
        local_path = out_dir / filename
        
        result = download_file(repo, remote_path, local_path)
        done += 1
        if result == "ok":
            ok += 1
        elif result == "skip":
            skip += 1
        else:
            fail += 1
            print(f"  {result}", flush=True)
        
        if done % 5 == 0 or done == total:
            elapsed = time.time() - t0
            print(f"  [{name}] {done}/{total} | ok={ok} skip={skip} fail={fail} | {elapsed:.0f}s", flush=True)
        
        time.sleep(2)  # stagger to avoid router overload
    
    elapsed = time.time() - t0
    print(f"[{name}] Done: {ok} ok, {skip} skip, {fail} fail in {elapsed:.0f}s", flush=True)

def main():
    for name, repo, pattern, max_files in DATASETS:
        download_dataset(name, repo, pattern, max_files)

if __name__ == "__main__":
    main()
