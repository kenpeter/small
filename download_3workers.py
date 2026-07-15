"""
3-worker parallel download for 4 datasets.
Worker 1: Stack Python (remaining 83 files)
Worker 2: FineMath (128 files)
Worker 3: Cosmopedia + OpenWebMath (13 + 114 = 127 files)
"""
import os
import subprocess
import concurrent.futures
from pathlib import Path
import time

DATA_DIR = Path("/home/kenpeter/work/data/_staging_multi")
HF_TOKEN = os.environ.get("HF_TOKEN", "")

DATASETS = {
    "stack-python": {
        "url_template": "https://huggingface.co/datasets/bigcode/the-stack-dedup/resolve/main/data/python/data-{i:05d}-of-00144.parquet",
        "start": 61,
        "end": 144,
        "dir": DATA_DIR / "stack-python"
    },
    "finemath": {
        "url_template": "https://huggingface.co/datasets/HuggingFaceTB/finemath/resolve/main/finemath-3plus/train-{i:05d}-of-00128.parquet",
        "start": 0,
        "end": 128,
        "dir": DATA_DIR / "finemath-3plus"
    },
    "cosmopedia": {
        "url_template": "https://huggingface.co/datasets/HuggingFaceTB/cosmopedia/resolve/main/data/stanford/train-{i:05d}-of-00013.parquet",
        "start": 0,
        "end": 13,
        "dir": DATA_DIR / "cosmopedia"
    },
    "open-web-math": {
        "url_template": "https://huggingface.co/datasets/open-web-math/open-web-math/resolve/main/data/train-{i:05d}-of-00114.parquet",
        "start": 0,
        "end": 114,
        "dir": DATA_DIR / "open-web-math"
    }
}

def download_file(url, out_path, token):
    """Download single file with curl."""
    if out_path.exists() and out_path.stat().st_size > 1000000:
        return f"SKIP: {out_path.name}"
    
    cmd = [
        "curl", "-sL", "--max-time", "300",
        "-H", f"Authorization: Bearer {token}",
        "-o", str(out_path), url
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=300)
        if result.returncode == 0 and out_path.exists():
            size = out_path.stat().st_size
            if size > 1000000:
                return f"OK: {out_path.name} ({size//1024//1024}MB)"
            else:
                out_path.unlink()
                return f"FAIL: {out_path.name} (too small)"
        else:
            if out_path.exists():
                out_path.unlink()
            return f"FAIL: {out_path.name}"
    except Exception as e:
        if out_path.exists():
            out_path.unlink()
        return f"ERROR: {out_path.name} - {e}"

def worker_stack():
    """Worker 1: Stack Python remaining files."""
    ds = DATASETS["stack-python"]
    ds["dir"].mkdir(parents=True, exist_ok=True)
    results = []
    for i in range(ds["start"], ds["end"]):
        url = ds["url_template"].format(i=i)
        out = ds["dir"] / f"data-{i:05d}-of-00144.parquet"
        results.append(download_file(url, out, HF_TOKEN))
        if i % 10 == 0:
            print(f"[Stack] {i}/{ds['end']}: {results[-1]}")
    return results

def worker_finemath():
    """Worker 2: FineMath."""
    ds = DATASETS["finemath"]
    ds["dir"].mkdir(parents=True, exist_ok=True)
    results = []
    for i in range(ds["start"], ds["end"]):
        url = ds["url_template"].format(i=i)
        out = ds["dir"] / f"train-{i:05d}-of-00128.parquet"
        results.append(download_file(url, out, HF_TOKEN))
        if i % 10 == 0:
            print(f"[FineMath] {i}/{ds['end']}: {results[-1]}")
    return results

def worker_cosmo_owm():
    """Worker 3: Cosmopedia + OpenWebMath."""
    results = []
    
    # Cosmopedia
    ds = DATASETS["cosmopedia"]
    ds["dir"].mkdir(parents=True, exist_ok=True)
    for i in range(ds["start"], ds["end"]):
        url = ds["url_template"].format(i=i)
        out = ds["dir"] / f"train-{i:05d}-of-00013.parquet"
        results.append(download_file(url, out, HF_TOKEN))
    print(f"[Cosmopedia] Done: {len([r for r in results if r.startswith('OK')])}/{ds['end']}")
    
    # OpenWebMath
    ds = DATASETS["open-web-math"]
    ds["dir"].mkdir(parents=True, exist_ok=True)
    for i in range(ds["start"], ds["end"]):
        url = ds["url_template"].format(i=i)
        out = ds["dir"] / f"train-{i:05d}-of-00114.parquet"
        results.append(download_file(url, out, HF_TOKEN))
        if i % 10 == 0:
            print(f"[OWM] {i}/{ds['end']}: {results[-1]}")
    
    return results

def main():
    print("3-Worker Parallel Download")
    print("=" * 50)
    print(f"Worker 1: Stack Python (files 61-144)")
    print(f"Worker 2: FineMath (128 files)")
    print(f"Worker 3: Cosmopedia (13) + OpenWebMath (114)")
    print()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        f1 = executor.submit(worker_stack)
        f2 = executor.submit(worker_finemath)
        f3 = executor.submit(worker_cosmo_owm)
        
        for future in concurrent.futures.as_completed([f1, f2, f3]):
            results = future.result()
            ok = len([r for r in results if r.startswith('OK')])
            print(f"Worker done: {ok}/{len(results)} files downloaded")
    
    print("\nAll workers complete!")

if __name__ == "__main__":
    main()
