"""
Download + tokenize DCLM (jsonl.zst format).
Streams zstd decompression to avoid disk bloat.
Resumable. Deletes raw .zst after tokenization.
"""
import os, sys, json, time, gc, subprocess, glob
from pathlib import Path

import numpy as np
import zstandard
from transformers import AutoTokenizer

# ─── Config ──────────────────────────────────────────────────────────────────
TOKENIZER_NAME = "HuggingFaceTB/SmolLM2-135M"
MAX_TEXT_CHARS = 50000
SHARD_SIZE = int(1_073_741_824)  # 1 GiB tokens

DATA_DIR = Path("/home/kenpeter/work/data")
STAGING_DIR = DATA_DIR / "_staging_dclm"
CHECKPOINT = DATA_DIR / "dclm_checkpoint.json"
LOG_FILE = DATA_DIR / "dclm_pipeline.log"

HF_TOKEN = os.environ.get("HF_TOKEN", "")

# DCLM is ~7TB total. We target ~44GB tokenized = ~80-100GB raw text.
# Each .zst ~200MB → decompresses to ~600-800MB text.
# Need ~120-150 files ≈ one local shard (279 files = ~720GB compressed = way too much).
# We'll take 150 files from global-shard_01/local-shard_0_of_10.
DCLM_REPO = "mlfoundations/dclm-baseline-1.0"
DCLM_PATH_PREFIX = "global-shard_01_of_10/local-shard_0_of_10"
MAX_FILES = 150  # Adjust up/down based on disk/speed

# ─── Helpers ─────────────────────────────────────────────────────────────────
def log(msg):
    t = time.strftime("%H:%M:%S")
    line = f"[{t}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")

def load_checkpoint():
    if CHECKPOINT.exists():
        with open(CHECKPOINT) as f:
            return json.load(f)
    return {"done": [], "shard_idx": 49, "buf_pos": 0}  # continue from existing shards

def save_checkpoint(state):
    tmp = CHECKPOINT.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(state, f)
    os.replace(tmp, CHECKPOINT)

def get_dclm_files():
    from huggingface_hub import list_repo_files
    all_files = list_repo_files(DCLM_REPO, repo_type="dataset", token=HF_TOKEN)
    zst_files = [f for f in all_files
                 if f.startswith(DCLM_PATH_PREFIX) and f.endswith(".jsonl.zst")]
    zst_files.sort()
    return zst_files[:MAX_FILES]

def hf_url(filename):
    return f"https://huggingface.co/datasets/{DCLM_REPO}/resolve/main/{filename}"

def download_file(filename, out_path):
    url = hf_url(filename)
    os.makedirs(out_path.parent, exist_ok=True)
    cmd = [
        "curl", "-s", "-L", "-o", str(out_path),
        "-H", f"Authorization: Bearer {HF_TOKEN}",
        "--retry", "2", "--max-time", "300",
        url,
    ]
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0:
        log(f"  curl failed: {r.stderr.decode()[:200]}")
        return False
    if out_path.stat().st_size < 1024 * 1024:  # < 1MB = probably HTML error
        log(f"  File too small ({out_path.stat().st_size} bytes), likely HTML redirect")
        out_path.unlink(missing_ok=True)
        return False
    return True

def stream_zstd_jsonl(zst_path, text_buffer, tokenizer, shard_idx, buf_pos, checkpoint):
    """Decompress zstd stream, read jsonl lines, tokenize text, append to shards."""
    dctx = zstandard.ZstdDecompressor()
    tokens_batch = []
    bytes_read = 0

    with open(zst_path, "rb") as fh:
        with dctx.stream_reader(fh) as reader:
            # Read line by line from decompressed stream
            buf = b""
            while True:
                chunk = reader.read(65536)
                if not chunk:
                    break
                buf += chunk
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    if not line:
                        continue
                    try:
                        obj = json.loads(line.decode("utf-8", errors="ignore"))
                        text = obj.get("text", "")
                        if not text:
                            continue
                        if len(text) > MAX_TEXT_CHARS:
                            text = text[:MAX_TEXT_CHARS]
                        tokens = tokenizer.encode(text, add_special_tokens=False)
                        tokens_batch.extend(tokens)
                    except Exception:
                        pass

                    # Flush to shard when batch is large enough
                    if len(tokens_batch) >= 100_000:
                        shard_idx, buf_pos = flush_tokens(tokens_batch, shard_idx, buf_pos, checkpoint)
                        tokens_batch = []

    # Flush remaining
    if tokens_batch:
        shard_idx, buf_pos = flush_tokens(tokens_batch, shard_idx, buf_pos, checkpoint)

    return shard_idx, buf_pos

def flush_tokens(tokens, shard_idx, buf_pos, checkpoint):
    arr = np.array(tokens, dtype=np.uint16)
    needed = min(len(arr), SHARD_SIZE - buf_pos)
    if needed > 0:
        shard_path = DATA_DIR / f"shard_{shard_idx:06d}.bin"
        if not shard_path.exists():
            # Pre-create empty shard
            np.zeros(SHARD_SIZE, dtype=np.uint16).tofile(shard_path)
        # mmap write
        mm = np.memmap(shard_path, dtype=np.uint16, mode="r+")
        mm[buf_pos:buf_pos + needed] = arr[:needed]
        mm.flush()
        del mm
        buf_pos += needed

    if buf_pos >= SHARD_SIZE:
        log(f"  ✓ Shard {shard_idx} full")
        checkpoint["shard_idx"] = shard_idx
        checkpoint["buf_pos"] = 0
        save_checkpoint(checkpoint)
        shard_idx += 1
        buf_pos = 0
        remaining = arr[needed:]
        if len(remaining) > 0:
            return flush_tokens(remaining.tolist(), shard_idx, buf_pos, checkpoint)

    checkpoint["shard_idx"] = shard_idx
    checkpoint["buf_pos"] = buf_pos
    save_checkpoint(checkpoint)
    return shard_idx, buf_pos

def main():
    os.makedirs(STAGING_DIR, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    checkpoint = load_checkpoint()
    done = set(checkpoint.get("done", []))
    shard_idx = checkpoint.get("shard_idx", 49)
    buf_pos = checkpoint.get("buf_pos", 0)

    log(f"Resuming DCLM: shard={shard_idx}, buf_pos={buf_pos}, done={len(done)} files")

    files = get_dclm_files()
    log(f"DCLM files to process: {len(files)}")

    for i, f in enumerate(files):
        if f in done:
            continue
        fname = f.replace("/", "__")
        zst_path = STAGING_DIR / fname

        log(f"[{i+1}/{len(files)}] Downloading {f}...")
        ok = download_file(f, zst_path)
        if not ok:
            log(f"  Skipping {f} (download failed)")
            continue
        log(f"  Downloaded {zst_path.stat().st_size / 1e6:.1f} MB")

        log(f"  Tokenizing...")
        try:
            shard_idx, buf_pos = stream_zstd_jsonl(zst_path, None, tokenizer, shard_idx, buf_pos, checkpoint)
            done.add(f)
            checkpoint["done"] = sorted(done)
            save_checkpoint(checkpoint)
        except Exception as e:
            log(f"  Tokenize error: {e}")
            continue
        finally:
            # Always delete raw to save disk
            zst_path.unlink(missing_ok=True)
            gc.collect()

        log(f"  ✓ Done. Current shard: {shard_idx}, pos: {buf_pos}")

    log("=== DCLM pipeline complete ===")

if __name__ == "__main__":
    main()
