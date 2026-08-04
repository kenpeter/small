#!/usr/bin/env python3
"""QuRatedPajama adaptive parallel downloader — FIXED monitor.
- Throughput measured by sampling actual staging dir size (captures in-flight
  wget writes, not just completed attempts).
- Hysteresis: 2 consecutive low windows required before dropping a worker.
- wget -c resume per file; shared queue; no double-downloads.
"""
import os, time, subprocess, json, urllib.request, threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

HF_TOKEN = os.environ.get("HF_TOKEN", "").strip()
if not HF_TOKEN:
    _tok = Path.home() / ".huggingface" / "token"
    if _tok.exists():
        HF_TOKEN = _tok.read_text().strip()
REPO = "princeton-nlp/QuRatedPajama-260B"
SPLIT = os.environ.get("SPLIT", "")
STAGING = Path("/mnt/file_drive/_qrp_staging")
LOG_FILE = Path("/mnt/file_drive/qrp_download.log")
MAX_FILES = 120
MAX_WORKERS = 8
MIN_WORKERS = 6
MAX_ATTEMPTS = 60
RETRY_BACKOFF = 10
CHECK_INTERVAL = 90
LOW_THRESHOLD = 0.15     # MB/s — only drop if nearly dead (2x in a row)
HIGH_THRESHOLD = 1.2     # MB/s — above this = ramp a worker back up

STAGING.mkdir(parents=True, exist_ok=True)
_log_lock = threading.Lock()

_current_workers = MAX_WORKERS
_low_streak = 0

def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    with _log_lock:
        print(line, flush=True)
        with open(LOG_FILE, "a") as f:
            f.write(line + "\n")

def dir_size():
    return sum(x.stat().st_size for x in STAGING.glob("*.parquet"))

def monitor(prev_size, prev_time):
    """Called every CHECK_INTERVAL from monitor thread. Returns (new_size, new_time)."""
    global _current_workers, _low_streak
    now = time.time()
    cur = dir_size()
    dt = now - prev_time
    speed = (cur - prev_size) / 1e6 / dt if dt > 0 else 0
    if speed < LOW_THRESHOLD:
        _low_streak += 1
        if _low_streak >= 2 and _current_workers > MIN_WORKERS:
            _current_workers -= 1
            log(f"⚠️  {speed:.2f} MB/s for 2 windows — dropping to {_current_workers} workers")
            _low_streak = 0
        else:
            log(f"📉 {speed:.2f} MB/s (streak {_low_streak}) — holding at {_current_workers}")
    else:
        _low_streak = 0
        if speed > HIGH_THRESHOLD and _current_workers < MAX_WORKERS:
            _current_workers += 1
            log(f"✅ {speed:.2f} MB/s — ramping to {_current_workers} workers")
        else:
            log(f"📊 {speed:.2f} MB/s — steady at {_current_workers} workers")
    return cur, now

def list_files():
    """Return list of (path, size) from API — no HEAD requests needed."""
    url = f"https://huggingface.co/api/datasets/{REPO}/tree/main/data?recursive=false&limit=1000"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {HF_TOKEN}"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read())
    files = [(item["path"], item.get("size", 0)) for item in data
             if item.get("type") == "file" and item["path"].endswith(".parquet")]
    return sorted(files, key=lambda x: x[0])[:MAX_FILES]

def download_one(fentry):
    remote_path, want = fentry
    fname = Path(remote_path).name
    local = STAGING / fname
    if local.exists() and want > 0 and local.stat().st_size >= want * 0.99:
        return "skip"
    url = f"https://huggingface.co/datasets/{REPO}/resolve/main/{remote_path}"
    for attempt in range(1, MAX_ATTEMPTS + 1):
        if local.exists() and want > 0 and local.stat().st_size >= want * 0.99:
            return "ok"
        cmd = ["wget", "-c", "-q", "--tries=1", "--timeout=60",
               "-O", str(local), "--header", f"Authorization: Bearer {HF_TOKEN}", url]
        try:
            subprocess.run(cmd, timeout=180, capture_output=True)
        except subprocess.TimeoutExpired:
            pass  # partial kept, resume next round
        got = local.stat().st_size if local.exists() else 0
        if want > 0 and got >= want * 0.99:
            log(f"✅ {fname} complete ({got/1e6:.0f}MB)")
            return "ok"
        if attempt % 5 == 0:
            log(f"⏳ {fname}: {got/1e6:.0f}/{want/1e6:.0f}MB (attempt {attempt})")
        time.sleep(RETRY_BACKOFF)
    log(f"❌ {fname}: gave up after {MAX_ATTEMPTS} attempts ({got/1e6:.0f}/{want/1e6:.0f}MB)")
    return "fail"

if __name__ == "__main__":
    log(f"Listing files (cap {MAX_FILES})...")
    files = list_files()
    pending = []
    for fpath, fsize in files:
        local = STAGING / Path(fpath).name
        if local.exists() and fsize > 0 and local.stat().st_size >= fsize * 0.99:
            continue
        pending.append((fpath, fsize))
    log(f"Found {len(files)} parquet, {len(pending)} pending. Adaptive workers (start {MAX_WORKERS}).")
    if SPLIT == "A":
        pending = pending[0::2]
    elif SPLIT == "B":
        pending = pending[1::2]
    if SPLIT:
        log(f"SPLIT={SPLIT}: assigned {len(pending)} files")

    # monitor thread: sample disk size every CHECK_INTERVAL
    state = {"prev": dir_size(), "time": time.time()}
    def monitor_loop():
        while True:
            time.sleep(CHECK_INTERVAL)
            state["prev"], state["time"] = monitor(state["prev"], state["time"])
    threading.Thread(target=monitor_loop, daemon=True).start()

    t0 = time.time()
    results = {"ok": 0, "skip": 0, "fail": 0}
    done = 0
    while pending:
        batch = pending[:_current_workers]
        pending = pending[_current_workers:]
        with ThreadPoolExecutor(max_workers=_current_workers) as ex:
            for status in ex.map(download_one, batch):
                results[status] = results.get(status, 0) + 1
                done += 1
                if done % 10 == 0:
                    n = len(list(STAGING.glob("*.parquet")))
                    sz = sum(x.stat().st_size for x in STAGING.glob("*.parquet")) / 1e9
                    log(f"  progress: {n}/{len(files)} files, {sz:.1f} GB, {(time.time()-t0)/60:.1f} min")
    n = len(list(STAGING.glob("*.parquet")))
    sz = sum(x.stat().st_size for x in STAGING.glob("*.parquet")) / 1e9
    log(f"DONE: {n}/{len(files)} files, {sz:.1f} GB in {(time.time()-t0)/60:.1f} min | {results}")
