"""
Low-impact download of raw parquet files for relaxed pretraining pipeline.
One dataset at a time, 4 workers each. Saves to _staging_multi/.
"""
import os, sys, time, fnmatch
os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# Read token from environment (avoids hardcoding secrets in source)
# Set before running: export HF_TOKEN=hf_xxxxx
HF_TOKEN = os.environ.get("HF_TOKEN", "")
# If no token provided, run unauthenticated (public datasets only, slower)
STAGING = Path("/home/kenpeter/work/data/_staging_multi")
STAGING.mkdir(parents=True, exist_ok=True)

from huggingface_hub import list_repo_files, hf_hub_download

CONFIGS = [
    ("fineweb-edu", "HuggingFaceFW/fineweb-edu", "data/CC-MAIN-*/train-*.parquet", 200),
    ("finemath-3plus", "HuggingFaceTB/finemath", "finemath-3plus/train-*.parquet", None),
    ("cosmopedia", "HuggingFaceTB/cosmopedia", "data/stanford/train-*.parquet", None),
    ("open-web-math", "open-web-math/open-web-math", "data/train-*.parquet", None),
]

def get_files(repo, pattern, max_files=None):
    all_files = list_repo_files(repo, repo_type='dataset', token=HF_TOKEN)
    parquet = sorted([f for f in all_files if f.endswith('.parquet') and 'sample/' not in f])
    matched = [f for f in parquet if fnmatch.fnmatch(f, pattern)]
    if max_files and len(matched) > max_files:
        matched = matched[:max_files]
    return matched

def download_file(repo, remote_path, local_path):
    if local_path.exists() and local_path.stat().st_size > 100000:
        return "skip"
    try:
        import shutil
        cached = hf_hub_download(
            repo_id=repo,
            filename=remote_path,
            repo_type='dataset',
            token=HF_TOKEN,
        )
        if cached and Path(cached).exists() and Path(cached).stat().st_size > 100000:
            shutil.copy2(cached, str(local_path))
            return "ok"
        return "fail"
    except Exception as e:
        return f"err:{str(e)[:60]}"

def download_dataset(name, repo, pattern, max_files):
    print(f"\n[{name}] Listing files...", flush=True)
    files = get_files(repo, pattern, max_files)
    if not files:
        print(f"[{name}] No files found!", flush=True)
        return

    out_dir = STAGING / name
    out_dir.mkdir(parents=True, exist_ok=True)

    total = len(files)
    done, ok, skip = 0, 0, 0
    t0 = time.time()
    print(f"[{name}] {total} files to download (3 workers, staggered)", flush=True)

    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {}
        idx = 0

        while len(futures) < 8 and idx < total:
            f = files[idx]; local = out_dir / f.split('/')[-1]
            if local.exists() and local.stat().st_size > 100000:
                skip += 1; done += 1; idx += 1; continue
            futures[pool.submit(download_file, repo, f, local)] = f
            idx += 1
            time.sleep(1.5)  # stagger starts to keep network usable

        while futures:
            for future in as_completed(futures):
                f = futures.pop(future); result = future.result()
                done += 1
                if result == "ok": ok += 1
                else: skip += 1
                if done % 5 == 0 or done == total:
                    elapsed = time.time() - t0
                    print(f"  [{name}] {done}/{total} | ok={ok} skip={skip} | {elapsed:.0f}s", flush=True)
                break

            while len(futures) < 8 and idx < total:
                f = files[idx]; local = out_dir / f.split('/')[-1]
                if local.exists() and local.stat().st_size > 100000:
                    skip += 1; done += 1; idx += 1; continue
                futures[pool.submit(download_file, repo, f, local)] = f
                idx += 1
                time.sleep(1.5)

    elapsed = time.time() - t0
    print(f"[{name}] Done: {ok} ok, {skip} skip in {elapsed:.0f}s", flush=True)

def main():
    for name, repo, pattern, max_files in CONFIGS:
        download_dataset(name, repo, pattern, max_files)

if __name__ == "__main__":
    main()
