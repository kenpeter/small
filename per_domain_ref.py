#!/usr/bin/env python3
"""Per-domain reference losses for DoReMi-lite (arXiv 2305.10429).

Computes the 'before restart' baseline: mean CE loss of a checkpoint on
sample sequences from each domain shard dir. The resulting refs.json is
passed to pretrain_gpu.py --doremi-ref — domains whose CURRENT loss exceeds
their reference get upweighted at each curriculum re-glide.

Usage (run while training is STOPPED, it needs the GPU):
    ./venv/bin/python per_domain_ref.py --ckpt checkpoints/megatrain_latest.pt \
        --samples 200 --out refs.json

No training — inference only. ~2-5 min on GPU, ~30-60 min on CPU.
"""
import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from pretrain_megatrain import SHARD_DIRS, _load_shard_list
from pretrain_gpu import build_model, chunked_ce

SEQ_LEN = 2048


def eval_domain(model, shard_path, seq_len, n_samples, device):
    """Mean next-token CE on n_samples random sequences from one shard list."""
    entries, total = _load_shard_list(shard_path, seq_len)
    if total == 0:
        return None
    # deterministic sample: evenly spaced indices across the domain
    step = max(1, total // n_samples)
    losses = []
    with torch.no_grad():
        for i in range(0, min(total, step * n_samples), step):
            # walk the shard list to find the file holding seq i
            for sp, n_seqs, start in entries:
                if i < start + n_seqs:
                    local = i - start
                    mm = __import__("numpy").memmap(str(sp), dtype="uint16", mode="r",
                                                    offset=local * seq_len * 2,
                                                    shape=(seq_len,))
                    toks = torch.from_numpy(mm.copy().astype("int64"))
                    del mm
                    break
            toks = toks.to(device).unsqueeze(0)
            hidden = model.model.norm(model.model(toks).last_hidden_state)
            logits = F.linear(hidden, model.lm_head.weight)  # bf16 [1, S, V]
            loss = chunked_ce(logits, toks)  # CPU-safe; identical to fused CE
            losses.append(loss.item())
    if not losses:
        return None
    return sum(losses) / len(losses)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/home/kenpeter/work/checkpoints/megatrain_latest.pt")
    ap.add_argument("--samples", type=int, default=200,
                    help="sequences sampled per domain (default 200, evenly spaced)")
    ap.add_argument("--out", default="/home/kenpeter/work/checkpoints/refs.json")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {args.ckpt} (device={device})...")
    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    sd = state["model_state_dict"]
    if any(k.startswith("_orig_mod.") for k in sd):
        sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    model = build_model(torch.bfloat16).to(device).eval()
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"  WARNING missing keys: {missing[:5]} ({len(missing)})")
    del state, sd
    torch.cuda.empty_cache()

    refs = {}
    for domain, dpath in sorted(SHARD_DIRS.items()):
        if not Path(dpath).exists():
            print(f"  skip {domain}: dir missing")
            continue
        loss = eval_domain(model, Path(dpath), SEQ_LEN, args.samples, device)
        if loss is None:
            print(f"  skip {domain}: no sequences")
            continue
        refs[domain] = round(loss, 4)
        print(f"  {domain}: ref loss {loss:.4f}")

    with open(args.out, "w") as f:
        json.dump(refs, f, indent=2, sort_keys=True)
    print(f"Saved {len(refs)} reference domains → {args.out}")
    print("Enable with: --doremi-lite --doremi-ref", args.out)


if __name__ == "__main__":
    main()
