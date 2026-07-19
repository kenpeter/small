"""
Low-impact download of raw parquet files for relaxed pretraining pipeline.
One dataset at a time, 3 workers each. Saves to _staging_multi/.
"""
import os, sys, time, json, subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

os.environ["HF_TOKEN"] = os.environ.get("HF_TOKEN", "")
STAGING = Path("/home/kenpeter/work/data/_staging_multi")
STAGING.mkdir(parents=True, exist_ok=True)

CONFIGS = [
    # fineweb-edu: 2410 files total (~180 GB). First 200 files = early dumps ~15 GB
    ("fineweb-edu", "HuggingFaceFW/fineweb-edu", "data/CC-MAIN-*/train-*.parquet", 200),
    ("finemath-3plus", "HuggingFaceTB/finemath", "finemath-3plus/train-*.parquet", None),
    ("cosmopedia", "HuggingFaceTB/cosmopedia", "data/stanford/train-*.parquet", None),
    ("open-web-math", "open-web-math/open-web-math", "data/train-*.parquet", None),
]

def get_files(repo, pattern, max_files=None):
    from huggingface_hub import list_repo_files
    all_files = list_repo_files(repo, repo_type='dataset')
    parquet = sorted([f for f in all_files if f.endswith('.parquet') and 'sample/' not in f])
    import fnmatch
    matched = [f for f in parquet if fnmatch.fnmatch(f, pattern)]
    if max_files and len(matched) > max_files:
        matched = matched[:max_files]
    return matched

def download_file(repo, remote_path, local_path):
    if local_path.exists() and local_path.stat().st_size > 100000:
        return "skip"
    url = f"https://huggingface.co/datasets/{repo}/resolve/main/{remote_path}"
    cmd = ["curl", "-sL", "--max-time", "600",
           "-H", f"Authorization: Bearer {os.environ['HF_TOKEN']}",
           "-o", str(local_path), url]
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=600)
        if r.returncode == 0 and local_path.exists() and local_path.stat().st_size > 100000:
            return "ok"
        if local_path.exists():
            local_path.unlink()
        return "fail"
    except:
        if local_path.exists():
            local_path.unlink()
        return "err"

def main():
    for name, repo, pattern, max_files in CONFIGS:
        print(f"\n[{name}] Listing files...", flush=True)
        files = get_files(repo, pattern, max_files)
        if not files:
            print(f"[{name}] No files found!", flush=True)
            continue
        
        out_dir = STAGING / name
        out_dir.mkdir(parents=True, exist_ok=True)
        
        total = len(files)
        done, ok, skip = 0, 0, 0
        t0 = time.time()
        
        print(f"[{name}] {total} files to download", flush=True)
        
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = {}
            idx = 0
            while idx < min(6, total):
                f = files[idx]
                local = out_dir / f.split('/')[-1]
                if local.exists() and local.stat().st_size > 100000:
                    skip += 1; done += 1; idx += 1
                    if done % 10 == 0:
                        elapsed = time.time() - t0
                        print(f"  [{name}] {done}/{total} | skip={skip} | {elapsed:.0f}s", flush=True)
                    continue
                futures[pool.submit(download_file, repo, f, local)] = f
                idx += 1
            
            while futures:
                for future in as_completed(futures):
                    f = futures.pop(future)
                    result = future.result()
                    done += 1
                    if result == "ok": ok += 1
                    else: skip += 1
                    if done % 10 == 0 or done == total:
                        elapsed = time.time() - t0
                        rate = done / elapsed if elapsed > 0 else 0
                        print(f"  [{name}] {done}/{total} | ok={ok} skip={skip} | {rate:.2f} file/s | {elapsed:.0f}s", flush=True)
                    break
                
                if idx < total:
                    f = files[idx]
                    local = out_dir / f.split('/')[-1]
                    if local.exists() and local.stat().st_size > 100000:
                        skip += 1; done += 1; idx += 1
                        continue
                    futures[pool.submit(download_file, repo, f, local)] = f
                    idx += 1
        
        elapsed = time.time() - t0
        print(f"[{name}] Done: {ok} ok, {skip} skip in {elapsed:.0f}s", flush=True)

if __name__ == "__main__":
    main()
