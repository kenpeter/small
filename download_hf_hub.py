"""
Download using huggingface_hub library (reliable, handles auth).
"""
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import huggingface_hub

DATA_DIR = Path("/home/kenpeter/work/data/_staging_multi")
HF_TOKEN = os.environ.get("HF_TOKEN", "")

# Ensure authenticated
huggingface_hub.login(token=HF_TOKEN)

DATASETS = [
    ("stack-python", "bigcode/the-stack-dedup", "data/python/data-{i:05d}-of-00144.parquet", 61, 144),
    ("finemath", "HuggingFaceTB/finemath", "finemath-3plus/train-{i:05d}-of-00128.parquet", 0, 128),
    ("cosmopedia", "HuggingFaceTB/cosmopedia", "data/stanford/train-{i:05d}-of-00013.parquet", 0, 13),
    ("open-web-math", "open-web-math/open-web-math", "data/train-{i:05d}-of-00114.parquet", 0, 114),
]

def download_file(name, repo_id, filename_template, i, out_dir):
    """Download single file using hf_hub."""
    filename = filename_template.format(i=i)
    out_path = out_dir / filename.split('/')[-1]
    
    if out_path.exists() and out_path.stat().st_size > 100000:
        return f"SKIP {name}/{i}"
    
    try:
        # Download with hf_hub
        downloaded = huggingface_hub.hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="dataset",
            local_dir=str(out_dir),
            local_dir_use_symlinks=False,
            resume_download=True,
        )
        
        file_size = Path(downloaded).stat().st_size
        return f"OK {name}/{i} ({file_size//1024//1024}MB)"
    except Exception as e:
        return f"FAIL {name}/{i}: {str(e)[:50]}"

def download_dataset(name, repo_id, template, start, end, max_workers=5):
    """Download one dataset with parallel workers."""
    out_dir = DATA_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[{name}] Starting files {start}-{end}...")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(download_file, name, repo_id, template, i, out_dir): i 
            for i in range(start, end)
        }
        
        completed = 0
        for future in ascompleted(futures):
            result = future.result()
            completed += 1
            if completed % 5 == 0 or "FAIL" in result:
                print(f"[{name}] {completed}/{end-start}: {result}")
    
    print(f"[{name}] Complete!")

def main():
    print("HF Hub Parallel Download")
    print("=" * 50)
    
    # Run each dataset with 5 workers
    for name, repo_id, template, start, end in DATASETS:
        download_dataset(name, repo_id, template, start, end, max_workers=5)
    
    print("\nAll done!")

if __name__ == "__main__":
    main()
