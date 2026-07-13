"""
Parallel tokenization of downloaded fineweb-edu parquet files into .bin shards.
Workers tokenize individual files; main process stitches tokens into 1GiB shards.
"""
import os, sys, json, time, gc
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pyarrow.parquet as pq
from transformers import AutoTokenizer

# ─── Config ──────────────────────────────────────────────────────
TOKENIZER_NAME = "HuggingFaceTB/SmolLM2-135M"
MAX_TEXT_CHARS = 50000
SHARD_SIZE = int(1_073_741_824)  # 1 GiB tokens → ~2 GB on disk (uint16)

DATA_DIR = Path("/home/kenpeter/work/data")
STAGING_DIR = DATA_DIR / "_staging_v2" / "HuggingFaceFW_fineweb-edu"
CHECKPOINT = DATA_DIR / "tokenize_checkpoint.json"

BATCH_SIZE = 5000
WORKERS = 4  # parallel tokenization workers

def _tokenize_one(parquet_path: str):
    """Tokenize a single parquet file, return all token IDs as np.uint16 array."""
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    try:
        table = pq.read_table(parquet_path)
        col = table.column("text") if "text" in table.column_names else table.column(0)
        texts = [str(t)[:MAX_TEXT_CHARS] for t in col.to_pylist() if t]
        del table, col
        gc.collect()

        all_ids = []
        for b_start in range(0, len(texts), BATCH_SIZE):
            batch = texts[b_start:b_start + BATCH_SIZE]
            encoded = tokenizer(batch, add_special_tokens=False, truncation=False, max_length=None)
            del batch
            for ids in encoded["input_ids"]:
                if ids:
                    all_ids.extend(ids)
            del encoded
            gc.collect()

        del texts
        gc.collect()

        tokens = np.array(all_ids, dtype=np.uint16)
        return {"success": True, "tokens": tokens, "path": parquet_path}
    except Exception as e:
        return {"success": False, "error": str(e), "path": parquet_path}


def main():
    # Load checkpoint
    if CHECKPOINT.exists():
        with open(CHECKPOINT) as f:
            state = json.load(f)
        processed = set(state.get("processed", []))
        shard_start = state.get("shards", 0)
        buf_pos = state.get("buf_pos", 0)
    else:
        processed = set()
        existing = sorted(DATA_DIR.glob("shard_*.bin"))
        shard_start = len(existing)
        buf_pos = 0

    # Find unprocessed parquet files
    parquet_files = sorted(STAGING_DIR.glob("data_CC-MAIN-*.parquet"))
    todo = [str(f) for f in parquet_files if str(f) not in processed]

    print(f"Already processed: {len(processed)}")
    print(f"Total parquet files: {len(parquet_files)}")
    print(f"Remaining to tokenize: {len(todo)}")
    print(f"Starting shard index: {shard_start}")

    if not todo:
        print("Nothing to tokenize!")
        return

    # Shared shard-building state
    buffer = np.empty(SHARD_SIZE, dtype=np.uint16)
    buf_pos = buf_pos
    shard_idx = shard_start
    total_tokens = 0
    all_processed = list(processed)

    def _flush():
        nonlocal buf_pos, shard_idx
        if buf_pos > 0:
            shard_path = DATA_DIR / f"shard_{shard_idx:06d}.bin"
            buffer[:buf_pos].tofile(shard_path)
            shard_idx += 1
            total_tokens_flushed = buf_pos
            buf_pos = 0
            return total_tokens_flushed
        return 0

    t0 = time.time()
    completed = 0

    with ProcessPoolExecutor(max_workers=WORKERS) as executor:
        futures = {}
        idx = 0

        # Seed first batch
        while idx < len(todo) and len(futures) < WORKERS * 2:
            fpath = todo[idx]
            future = executor.submit(_tokenize_one, fpath)
            futures[future] = fpath
            idx += 1

        while futures:
            for future in as_completed(futures):
                fpath = futures.pop(future)
                result = future.result()
                completed += 1

                if result["success"]:
                    tokens = result["tokens"]
                    n = len(tokens)
                    start = 0
                    while start < n:
                        remaining = SHARD_SIZE - buf_pos
                        chunk = tokens[start:start + remaining]
                        buffer[buf_pos:buf_pos + len(chunk)] = chunk
                        buf_pos += len(chunk)
                        start += len(chunk)
                        if buf_pos >= SHARD_SIZE:
                            _flush()
                    total_tokens += n
                    all_processed.append(fpath)
                    print(f"  ✅ {Path(fpath).name}: {n:,} tokens | shards={shard_idx} buf={buf_pos:,}")
                else:
                    print(f"  ❌ {Path(fpath).name}: {result['error']}")
                break

            # Checkpoint every 10 files
            if completed % 10 == 0:
                with open(CHECKPOINT, "w") as f:
                    json.dump({
                        "processed": all_processed,
                        "shards": shard_idx,
                        "buf_pos": buf_pos,
                    }, f)
                print(f"  💾 Checkpoint: {shard_idx} shards, {len(all_processed)} files")

            # Submit next
            if idx < len(todo):
                fpath = todo[idx]
                future = executor.submit(_tokenize_one, fpath)
                futures[future] = fpath
                idx += 1

    # Final flush + checkpoint
    flushed = _flush()
    total_tokens += flushed
    with open(CHECKPOINT, "w") as f:
        json.dump({
            "processed": all_processed,
            "shards": shard_idx,
            "buf_pos": buf_pos,
        }, f)

    elapsed = time.time() - t0
    print(f"\n🏁 Done: {len(all_processed)} files → {shard_idx} shards ({total_tokens:,} tokens) in {elapsed:.0f}s")


if __name__ == "__main__":
    main()
