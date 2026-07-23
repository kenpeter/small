#!/usr/bin/env python3
"""Super-robust parallel download with resume, size validation, and retry.

Fixed: handles large files (>2GB) with wget resume, longer timeouts,
       and skips datasets already present on disk.
"""
import os, time, json, subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from statistics import median

HF_TOKEN = os.environ.get("HF_TOKEN", "")
STAGING = Path("/home/kenpeter/work/data/_raw_original")
STAGING.mkdir(parents=True, exist_ok=True)

# Dataset config: (name, repo, pattern, max_files, skip_if_present_threshold)
# skip_if_present_threshold = min files on disk to consider "done"
DATASETS = [
    # fineweb-edu already have 42 sample files (~1.8GB) - skip full 2GB CC-MAIN files
    ("fineweb-edu", "HuggingFaceFW/fineweb-edu", "data/CC-MAIN-*/train-*.parquet", 0, 30),
    ("finemath-3plus", "HuggingFaceTB/finemath", "finemath-3plus/train-*.parquet", 200, 0),
    ("cosmopedia", "HuggingFaceTB/cosmopedia", "data/stanford/train-*.parquet", 200, 0),
    ("open-web-math", "open-web-math/open-web-math", "data/train-*.parquet", 200, 0),
]

LOG = Path("/home/kenpeter/work/small/download.log")


def log(msg):
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    # Note: stdout is redirected to download.log by the launcher


def list_files(repo, pattern, max_files=200):
    import urllib.request
    url = f"https://huggingface.co/api/datasets/{repo}/tree/main?recursive=true"
    headers = {}
    if HF_TOKEN:
        headers["Authorization"] = f"Bearer {HF_TOKEN}"
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
    except Exception as e:
        log(f"[list_files] API error for {repo}: {e}")
        return []
    files = [item["path"] for item in data
             if item.get("type") == "file"
             and item["path"].endswith(".parquet")
             and "train-" in item["path"]]
    prefix = pattern.split("*")[0]
    matched = [f for f in files if f.startswith(prefix)]
    return matched[:max_files]


def download_file(repo, remote_path, local_path, expected_size=None):
    """Download using wget with resume. Handles files up to 5GB+."""
    local_path = Path(local_path)

    # Check if already complete
    if local_path.exists():
        actual = local_path.stat().st_size
        if expected_size and actual >= expected_size * 0.98:
            return "skip"
        if not expected_size and actual > 10_000_000:  # >10MB heuristic
            return "skip"
        # Partial — delete and retry
        log(f"  Removing partial {local_path.name} ({actual/1e6:.1f} MB)")
        local_path.unlink()

    url = f"https://huggingface.co/datasets/{repo}/resolve/main/{remote_path}"

    # wget: resume, 20 retries, 20 min timeout, continue on slow networks
    cmd = [
        "wget", "-q",
        "-c",           # resume
        "-t", "20",     # 20 retries
        "--timeout=1200",   # 20 min per attempt
        "--read-timeout=600",
        "-O", str(local_path),
    ]
    if HF_TOKEN:
        cmd.extend(["--header", f"Authorization: Bearer {HF_TOKEN}"])
    cmd.append(url)

    for attempt in range(3):
        try:
            result = subprocess.run(cmd, capture_output=True, timeout=7200)  # 2h total
            if result.returncode == 0 and local_path.exists():
                actual = local_path.stat().st_size
                if expected_size and actual >= expected_size * 0.98:
                    return "ok"
                if actual > 10_000_000:
                    return "ok"
                # Too small — probably truncated
                log(f"  {local_path.name} too small ({actual/1e6:.1f} MB), retrying")
                local_path.unlink()
            else:
                stderr = result.stderr.decode() if result.stderr else ""
                log(f"  wget err (attempt {attempt+1}): rc={result.returncode} {stderr[:120]}")
        except subprocess.TimeoutExpired:
            log(f"  wget TIMEOUT attempt {attempt+1} for {local_path.name}")
            if local_path.exists():
                local_path.unlink()
        except Exception as e:
            log(f"  wget exception: {e}")

        time.sleep(15 * (attempt + 1))

    return "err: wget failed after 3 rounds"


def download_dataset(name, repo, pattern, max_files, skip_threshold):
    out_dir = STAGING / name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Check how many files already exist
    existing = list(out_dir.rglob("*.parquet"))
    if len(existing) >= skip_threshold and skip_threshold > 0:
        log(f"[{name}] Already have {len(existing)} files, skipping (threshold {skip_threshold})")
        return

    log(f"[{name}] Listing files...")
    files = list_files(repo, pattern, max_files)
    total = len(files)
    if total == 0:
        log(f"[{name}] No files found")
        return
    log(f"[{name}] {total} files to download")

    # Sample expected sizes from first 3 files via HEAD
    sample_sizes = []
    for f in files[:3]:
        import urllib.request
        url = f"https://huggingface.co/datasets/{repo}/resolve/main/{f}"
        headers = {}
        if HF_TOKEN:
            headers["Authorization"] = f"Bearer {HF_TOKEN}"
        req = urllib.request.Request(url, headers=headers, method="HEAD")
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                sz = resp.headers.get("Content-Length")
                if sz:
                    sample_sizes.append(int(sz))
        except Exception:
            pass
    expected_size = int(median(sample_sizes)) if sample_sizes else None
    if expected_size:
        log(f"[{name}] Expected file size ~{expected_size/1e6:.1f} MB")

    done, ok, skip, fail = 0, 0, 0, 0
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=4) as pool:  # 4 workers for faster downloads
        futures = {}
        idx = 0

        # Seed initial queue (depth 4 for large files)
        while len(futures) < 4 and idx < total:
            f = files[idx]
            local = out_dir / f.split("/")[-1]
            if local.exists():
                actual = local.stat().st_size
                if expected_size and actual >= expected_size * 0.98:
                    skip += 1; done += 1; idx += 1
                    continue
                if actual > 10_000_000 and not expected_size:
                    skip += 1; done += 1; idx += 1
                    continue
                # Partial — will be removed by download_file
            futures[pool.submit(download_file, repo, f, local, expected_size)] = f
            idx += 1
            time.sleep(3.0)  # slower stagger for large files

        while futures:
            for future in as_completed(futures):
                f = futures.pop(future)
                result = future.result()
                done += 1
                if result == "ok":
                    ok += 1
                elif result == "skip":
                    skip += 1
                else:
                    fail += 1
                    log(f"  {result}")

                if done % 3 == 0 or done == total:
                    elapsed = time.time() - t0
                    rate = done / (elapsed / 60) if elapsed > 0 else 0
                    log(f"[{name}] {done}/{total} | ok={ok} skip={skip} fail={fail} | {elapsed:.0f}s | {rate:.1f} files/min")
                break

            # Refill queue
            while len(futures) < 4 and idx < total:
                f = files[idx]
                local = out_dir / f.split("/")[-1]
                if local.exists():
                    actual = local.stat().st_size
                    if expected_size and actual >= expected_size * 0.98:
                        skip += 1; done += 1; idx += 1
                        continue
                    if actual > 10_000_000 and not expected_size:
                        skip += 1; done += 1; idx += 1
                        continue
                futures[pool.submit(download_file, repo, f, local, expected_size)] = f
                idx += 1
                time.sleep(3.0)

    elapsed = time.time() - t0
    log(f"[{name}] Done: {ok} ok, {skip} skip, {fail} fail in {elapsed:.0f}s")


def main():
    for name, repo, pattern, max_files, skip_threshold in DATASETS:
        download_dataset(name, repo, pattern, max_files, skip_threshold)

if __name__ == "__main__":
    main()
