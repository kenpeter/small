#!/usr/bin/env python3
"""Launch all 8 reformat chunks with 1 worker each."""
import subprocess, sys, time, os

venv_python = "/home/kenpeter/work/small/venv/bin/python3"
script = "/home/kenpeter/work/small/reformat_data.py"
model = "/home/kenpeter/.cache/huggingface/hub/models--Qwen--Qwen3-8B-AWQ/snapshots/4da05a8edb55c6046cce958586c33b61da07bb79"
base = "http://localhost:8000/v1"

procs = []
for i in range(8):
    inp = f"/home/kenpeter/work/data/_reformatted/_input_chunk_{i}.jsonl"
    out = f"/home/kenpeter/work/data/_reformatted_chunk_{i}"
    log = open(f"/tmp/chunk_{i}.log", "w")
    p = subprocess.Popen([
        venv_python, script,
        "--input", inp,
        "--output", out,
        "--formats", "textbook,qa",
        "--max-docs", "800",
        "--max-tokens", "2048",
        "--temperature", "0.2",
        "--workers", "1",
        "--api-base", base,
        "--model", model,
        "--no-resume",
    ], stdout=log, stderr=subprocess.STDOUT)
    procs.append(p)
    print(f"Chunk {i}: PID {p.pid}")
    time.sleep(2)

print(f"\nAll {len(procs)} launched. Waiting for completion...")
for i, p in enumerate(procs):
    p.wait()
    print(f"Chunk {i}: exited with code {p.returncode}")
print("Done")
