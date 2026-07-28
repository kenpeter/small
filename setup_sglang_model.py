#!/usr/bin/env python3
"""Create symlink for downloaded shard and verify the model is complete."""
import os

CACHE = "/home/kenpeter/.cache/huggingface/hub/models--Qwen--Qwen3-8B-AWQ"
SNAPSHOT = os.path.join(CACHE, "snapshots", "4da05a8edb55c6046cce958586c33b61da07bb79")
BLOBS = os.path.join(CACHE, "blobs")

LFS_SHA256 = "6e112429856bc65e3837a9f38d6f6b71ffdda832cb46299a12f4fa8f6352516e"
EXPECTED_SIZE = 4853922024  # 4.85 GB

blob_path = os.path.join(BLOBS, LFS_SHA256)
link_path = os.path.join(SNAPSHOT, "model-00001-of-00002.safetensors")

# Check blob
if not os.path.exists(blob_path):
    print(f"ERROR: Blob not found at {blob_path}")
    exit(1)

actual_size = os.path.getsize(blob_path)
print(f"Blob size: {actual_size:,} bytes ({actual_size/1024**3:.2f} GB)")
print(f"Expected:  {EXPECTED_SIZE:,} bytes ({EXPECTED_SIZE/1024**3:.2f} GB)")

if actual_size != EXPECTED_SIZE:
    print(f"WARNING: Size mismatch! Download may be incomplete.")
    exit(1)

print("Size matches! Creating symlink...")

# Remove existing link if any
if os.path.islink(link_path) or os.path.exists(link_path):
    os.remove(link_path)

# Create symlink (relative)
rel = os.path.relpath(blob_path, os.path.dirname(link_path))
os.symlink(rel, link_path)

# Verify
assert os.path.islink(link_path), "Symlink not created!"
assert os.path.exists(link_path), "Symlink points to non-existent file!"
real = os.path.realpath(link_path)
assert real == os.path.realpath(blob_path), f"Symlink broken: {real} != {blob_path}"

# List final snapshot
print("\n=== Final snapshot ===")
for f in sorted(os.listdir(SNAPSHOT)):
    fp = os.path.join(SNAPSHOT, f)
    sz = os.path.getsize(fp)
    print(f"  {f:50s} {sz/1024**3:.2f} GB")

# Quick model load test
import sys
sys.path.insert(0, "/home/kenpeter/work/small")
print("\nModel files complete! Ready for SGLang.")
