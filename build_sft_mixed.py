#!/usr/bin/env python3
"""Build a MIXED SFT set: verified LeetCode code + general instruction.

Merges already-tokenized .pt shards into one balanced SFT set so the model
gains correct code generation (verified) while keeping general/chat abilities
(avoids catastrophic forgetting on a code-only SFT).

Composition:
  - code    : 513 exec-verified problem->Solution  (from _sft_codegold_shards)
  - general : N sub-sampled instruction examples  (from _sft_final_shards:
              openhermes / openorca — balanced general chat)

Output: flat list of {input_ids,labels,dataset} saved as .pt shards (same format
sft_gpu.py'SFTDataset expects), written to _sft_mixed_shards.
"""
import argparse, glob, time
from pathlib import Path
import torch

GENERAL_DIR = "/mnt/file_drive/data/_sft_final_shards"
CODE_DIR = "/mnt/file_drive/data/_sft_codegold_shards"

def load_examples(dirpath):
    ex = []
    for f in sorted(glob.glob(str(Path(dirpath) / "*.pt"))):
        d = torch.load(f, map_location="cpu", weights_only=False)
        ex.extend(d)
    return ex

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--code-dir", default=CODE_DIR)
    ap.add_argument("--general-dir", default=GENERAL_DIR)
    ap.add_argument("--general-n", type=int, default=2500,
                    help="number of general instruction examples to include")
    ap.add_argument("--out-dir", default="/mnt/file_drive/data/_sft_mixed_shards")
    ap.add_argument("--per-shard", type=int, default=1000)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading code examples...", flush=True)
    code = load_examples(args.code_dir)
    print(f"  code: {len(code)}", flush=True)

    # Sub-sample general from openhermes + openorca heads (already balanced/tokenized)
    general = []
    # take from openhermes then openorca in a round-robin to keep balance
    for prefix in ["openhermes", "openorca"]:
        files = sorted(glob.glob(str(Path(args.general_dir) / f"shard_{prefix}_*.pt")))
        want = args.general_n // 2
        have = 0
        for f in files:
            if have >= want:
                break
            d = torch.load(f, map_location="cpu", weights_only=False)
            take = min(want - have, len(d))
            general.extend(d[:take])
            have += take
            print(f"  {Path(f).name}: +{take} (total general {len(general)})", flush=True)
    print(f"  general total: {len(general)}", flush=True)

    # Build balance: code + general, shuffle for mixing
    all_ex = code + general
    # interleave so batches mix domains
    mixed = []
    ci, gi = 0, 0
    while ci < len(code) or gi < len(general):
        if ci < len(code):
            mixed.append(code[ci]); ci += 1
        if gi < len(general):
            mixed.append(general[gi]); gi += 1

    n = 0
    shards = []
    for e in mixed:
        shards.append(e)
        if len(shards) >= args.per_shard:
            n += 1
            torch.save(shards, out_dir / f"mix_{n:05d}.pt")
            shards = []
    if shards:
        n += 1
        torch.save(shards, out_dir / f"mix_{n:05d}.pt")
    print(f"DONE: mixed={len(mixed)} (code {len(code)} + general {len(general)}) -> {n} shards in {out_dir}", flush=True)

if __name__ == "__main__":
    main()
