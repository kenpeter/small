"""
True parallel download - 10 concurrent per dataset.
Uses ThreadPoolExecutor with 10 workers each.
"""
import os, subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

DATA_DIR = Path("/home/kenpeter/work/data/_staging_multi")
HF_TOKEN = os.environ.get("HF_TOKEN", "")

# Dataset configs
CONFIGS = [
    ("stack-python", "bigcode/the-stack-dedup", "data/python/data-{i:05d}-of-00144.parquet", 61, 144),
    ("finemath", "HuggingFaceTB/finemath", "finemath-3plus/train-{i:05d}-of-00128.parquet", 0, 128),
    ("cosmopedia", "HuggingFaceTB/cosmopedia", "data/stanford/train-{i:05d}-of-00013.parquet", 0, 13),
    ("open-web-math", "open-web-math/open-web-math", "data/train-{i:05d}-of-00114.parquet", 0, 114),
]

def download_one(name, repo, template, i, out_dir):
    """Download single file."""
    filename = template.format(i=i)
    url = f"https://huggingface.co/datasets/{repo}/resolve/main/{filename}"
    out_path = out_dir / filename.split('/')[-1]
    
    if out_path.exists() and out_path.stat().st_size > 100000:
        return f"SKIP {name}/{i}"
    
    cmd = [
        "curl", "-fsL", "--max-time", "300",
        "-H", f"Authorization: Bearer {HF_TOKEN}",
        "-o", str(out_path), url
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=300)
        if result.returncode == 0 and out_path.exists() and out_path.stat().st_size > 100000:
            return f"OK {name}/{i} ({out_path.stat().st_size//1024//1024}MB)"
        else:
            if out_path.exists():
                out_path.unlink()
            return f"FAIL {name}/{i}"
    except Exception as e:
        if out_path.exists():
            out_path.unlink()
        return f"ERR {name}/{i}: {e}"

def download_dataset(name, repo, template, start, end, max_workers=10):
    """Download one dataset with 10 parallel workers."""
    out_dir = DATA_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[{name}] Starting {start}-{end} with {max_workers} workers...")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(download_one, name, repo, template, i, out_dir): i 
            for i in range(start, end)
        }
        
        completed = 0
        for future in as_completed(futures):
            result = future.result()
            completed += 1
            if completed % 5 == 0 or "ERR" in result or "FAIL" in result:
                print(f"[{name}] {completed}/{end-start}: {result}")
    
    print(f"[{name}] DONE")

def main():
    print("10-Worker Parallel Download (10 per dataset)")
    print("=" * 50)
    
    # Run all 4 datasets in parallel with 10 workers each
    with ThreadPoolExecutor(max_workers=4) as executor:
        for name, repo, template, start, end in CONFIGS:
            executor.submit(download_dataset, name, repo, template, start, end, 10)
    
    print("\nAll downloads complete!")

if __name__ == "__main__":
    main()
