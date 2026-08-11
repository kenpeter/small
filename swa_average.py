#!/usr/bin/env python3
"""SWA: average the lean swa_tail snapshots into one flat-basin checkpoint.

Usage: python3 swa_average.py [--last N] [--swa-dir DIR] [--out PATH]
Run at the END of a training run (or any time) over the snapshots collected
by --swa-window in pretrain_gpu.py. Output ckpt matches the megatrain format
(model_state_dict, bf16) → loadable by quick_eval_pretrain.py / apply_resume.
"""
import argparse
import os
import torch
from pathlib import Path


def average_swa(swa_dir, out_path, last=0):
    """Average model weights of the (last `last`) swa snapshots → out_path.

    Returns the list of averaged step numbers. Exact arithmetic mean in fp32,
    saved back in bf16 (same dtype as training) — lands in the flat basin a
    sharp-minimum finder like SAM would hunt for, at ~zero training cost.
    """
    paths = sorted(Path(swa_dir).glob("swa_*.pt"))
    if last and last > 0:
        paths = paths[-last:]
    if not paths:
        raise SystemExit(f"no swa snapshots found in {swa_dir}")
    acc, steps = None, []
    for p in paths:
        ck = torch.load(p, map_location="cpu", weights_only=False)
        sd = ck["model_state_dict"]
        if acc is None:
            acc = {k: v.clone().float() for k, v in sd.items()}
        else:
            for k, v in sd.items():
                acc[k] += v.float()
        steps.append(int(ck["step"]))
    n = len(paths)
    for k in acc:
        acc[k] = (acc[k] / n).to(torch.bfloat16)
    torch.save({
        "step": steps[-1],
        "loss": None,
        "best_loss": None,
        "model_state_dict": acc,
        "config": {"swa": f"{n} snapshots (steps {steps[0]}..{steps[-1]})"},
    }, out_path)
    print(f"SWA: averaged {n} snapshots (steps {steps[0]}..{steps[-1]}) → {out_path}")
    return steps


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--swa-dir", default="/home/kenpeter/work/checkpoints/swa_tail")
    ap.add_argument("--out", default="/home/kenpeter/work/checkpoints/megatrain_swa.pt")
    ap.add_argument("--last", type=int, default=0, help="use only the last N snapshots (0 = all)")
    args = ap.parse_args()
    average_swa(args.swa_dir, args.out, args.last)
