"""
Download without auth (public datasets).
Uses wget which is more reliable than curl.
"""
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

DATA_DIR = Path("/home/kenpeter/work/data/_staging_multi")

URLS = {
    "finemath": [
        (f"https://huggingface.co/datasets/HuggingFaceTB/finemath/resolve/main/finemath-3plus/train-{i:05d}-of-00128.parquet", 
         DATA_DIR / "finemath-3plus" / f"train-{i:05d}-of-00128.parquet")
        for i in range(4, 128)  # Continue from file 4
    ],
    "open-web-math": [
        (f"https://huggingface.co/datasets/open-web-math/open-web-math/resolve/main/data/train-{i:05d}-of-00114.parquet",
         DATA_DIR / "open-web-math" / f"train-{i:05d}-of-00114.parquet")
        for i in range(114)
    ],
}

def download(url, out_path):
    """Download with wget."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    if out_path.exists() and out_path.stat().st_size > 1000000:
        return f"SKIP {out_path.name}"
    
    tmp_path = out_path.with_suffix('.tmp')
    
    cmd = ["wget", "-q", "--timeout=300", "-O", str(tmp_path), url]
    
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=300)
        if result.returncode == 0 and tmp_path.exists() and tmp_path.stat().st_size > 1000000:
            tmp_path.rename(out_path)
            return f"OK {out_path.name} ({out_path.stat().st_size//1024//1024}MB)"
        else:
            if tmp_path.exists():
                tmp_path.unlink()
            return f"FAIL {out_path.name}"
    except Exception as e:
        if tmp_path.exists():
            tmp_path.unlink()
        return f"ERR {out_path.name}: {e}"

def main():
    print("Public Download (FineMath + OpenWebMath)")
    print("=" * 50)
    
    # Flatten all URLs
    all_downloads = []
    for dataset, urls in URLS.items():
        all_downloads.extend(urls)
        print(f"{dataset}: {len(urls)} files")
    
    print(f"\nTotal: {len(all_downloads)} files")
    print("Starting 10 parallel workers...\n")
    
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(download, url, path): (url, path) for url, path in all_downloads}
        
        completed = 0
        for future in as_completed(futures):
            result = future.result()
            completed += 1
            if completed % 10 == 0 or "FAIL" in result or "ERR" in result:
                print(f"[{completed}/{len(all_downloads)}] {result}")
    
    print("\nDone!")

if __name__ == "__main__":
    main()
