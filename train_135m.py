#!/usr/bin/env python3
"""
Train a SmolLM2-135M replica from scratch on a single GPU.
Simple, efficient PyTorch training loop with gradient accumulation.
"""

import os
import sys
import math
import json
import time
import random
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Iterator

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import LlamaConfig, LlamaForCausalLM, AutoTokenizer
from tqdm import tqdm

# ── Config ───────────────────────────────────────────────────────────────────

@dataclass
class TrainConfig:
    # Model
    vocab_size: int = 49152
    hidden_size: int = 576
    intermediate_size: int = 1536
    num_layers: int = 30
    num_attention_heads: int = 9
    num_key_value_heads: int = 3
    max_position_embeddings: int = 2048  # start at 2k, extend later
    rms_norm_eps: float = 1e-5
    rope_theta: float = 100000.0
    tie_word_embeddings: bool = True

    # Training
    batch_size: int = 8
    grad_accum_steps: int = 4      # effective batch = 8 * 4 * 2048 = 65k tokens
    max_steps: int = 100_000
    learning_rate: float = 3e-3
    min_lr: float = 0.0
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    warmup_steps: int = 2000
    decay_ratio: float = 0.20      # WSD: decay last 20% of steps

    # Data
    data_dir: str = "/home/kenpeter/work/data"
    seq_len: int = 2048

    # System
    device: str = "cuda"
    dtype: str = "bfloat16"
    compile: bool = True
    eval_interval: int = 1000
    save_interval: int = 5000
    out_dir: str = "/home/kenpeter/work/small/out"

    # Logging
    wandb: bool = False


# ── Dataset ───────────────────────────────────────────────────────────────────

class TokenizedDataset(Dataset):
    """Memory-mapped dataset of uint16 token shards."""

    def __init__(self, data_dir: str, seq_len: int):
        self.seq_len = seq_len
        self.shards = sorted(Path(data_dir).glob("shard_*.bin"))
        if not self.shards:
            raise ValueError(f"No shards found in {data_dir}")

        self.mmaps = []
        self.cum_lengths = []
        total = 0
        for s in self.shards:
            tokens = np.memmap(str(s), dtype=np.uint16, mode="r")
            n = len(tokens) // seq_len
            self.mmaps.append(tokens)
            total += n
            self.cum_lengths.append(total)
        self.total_samples = total
        print(f"📖 Loaded {len(self.shards)} shards, {self.total_samples:,} samples")

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx: int):
        # Find which shard
        shard_idx = 0
        for i, cum in enumerate(self.cum_lengths):
            if idx < cum:
                shard_idx = i
                break
        if shard_idx > 0:
            local_idx = idx - self.cum_lengths[shard_idx - 1]
        else:
            local_idx = idx

        offset = local_idx * self.seq_len
        tokens = self.mmaps[shard_idx][offset:offset + self.seq_len]
        x = torch.from_numpy(tokens.astype(np.int64))
        return x


# ── LR Schedule ──────────────────────────────────────────────────────────────────

def get_lr(step: int, cfg: TrainConfig) -> float:
    """WSD (Warmup-Stable-Decay) scheduler."""
    if step < cfg.warmup_steps:
        return cfg.learning_rate * step / cfg.warmup_steps

    decay_start = int(cfg.max_steps * (1 - cfg.decay_ratio))
    if step < decay_start:
        return cfg.learning_rate

    # Linear decay
    decay_steps = cfg.max_steps - decay_start
    ratio = (step - decay_start) / decay_steps
    return cfg.learning_rate * (1 - ratio) + cfg.min_lr * ratio


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--data_dir", default="/home/kenpeter/work/data")
    parser.add_argument("--out_dir", default="/home/kenpeter/work/small/out")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--max_steps", type=int, default=100_000)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--compile", action="store_true", default=True)
    parser.add_argument("--wandb", action="store_true", default=False)
    args = parser.parse_args()

    cfg = TrainConfig(
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        data_dir=args.data_dir,
        out_dir=args.out_dir,
        compile=args.compile,
        wandb=args.wandb,
    )

    # Setup
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if cfg.dtype == "bfloat16" and torch.cuda.is_bf16_supported() else torch.float16
    Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)

    print(f"🚀 Device: {device}, dtype: {dtype}")
    print(f"📊 Effective batch size: {cfg.batch_size * cfg.grad_accum_steps} sequences = "
          f"{cfg.batch_size * cfg.grad_accum_steps * cfg.seq_len:,} tokens/step")

    # Load tokenizer (for saving model later)
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")

    # Build model
    llama_cfg = LlamaConfig(
        vocab_size=cfg.vocab_size,
        hidden_size=cfg.hidden_size,
        intermediate_size=cfg.intermediate_size,
        num_hidden_layers=cfg.num_layers,
        num_attention_heads=cfg.num_attention_heads,
        num_key_value_heads=cfg.num_key_value_heads,
        max_position_embeddings=cfg.max_position_embeddings,
        rms_norm_eps=cfg.rms_norm_eps,
        rope_theta=cfg.rope_theta,
        tie_word_embeddings=cfg.tie_word_embeddings,
        torch_dtype=cfg.dtype,
    )
    model = LlamaForCausalLM(llama_cfg)
    model = model.to(device=device, dtype=dtype)

    if cfg.compile and hasattr(torch, "compile"):
        print("🔧 Compiling model...")
        model = torch.compile(model)

    print(f"🧪 Model params: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer
    decay_params = [p for n, p in model.named_parameters() if "bias" not in n and "embed" not in n]
    nodecay_params = [p for n, p in model.named_parameters() if "bias" in n or "embed" in n]
    optim_groups = [
        {"params": decay_params, "weight_decay": cfg.weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(optim_groups, lr=cfg.learning_rate, betas=(0.9, 0.95))

    # Resume
    step = 0
    if args.resume:
        ckpts = sorted(Path(cfg.out_dir).glob("ckpt_step_*.pt"))
        if ckpts:
            latest = ckpts[-1]
            state = torch.load(latest, map_location=device)
            model.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            step = state["step"]
            print(f"🔄 Resumed from {latest.name} at step {step}")

    # DataLoader
    dataset = TokenizedDataset(cfg.data_dir, cfg.seq_len)
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )

    # Training loop
    model.train()
    optimizer.zero_grad()
    t0 = time.time()
    running_loss = 0.0
    loss_steps = 0

    pbar = tqdm(initial=step, total=cfg.max_steps, desc="Training")
    data_iter = iter(loader)

    while step < cfg.max_steps:
        for accum in range(cfg.grad_accum_steps):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                batch = next(data_iter)

            batch = batch.to(device)
            inputs = batch[:, :-1]
            targets = batch[:, 1:]

            with torch.autocast(device_type=device.type, dtype=dtype):
                logits = model(inputs).logits
                loss = nn.functional.cross_entropy(
                    logits.view(-1, cfg.vocab_size),
                    targets.reshape(-1),
                    ignore_index=-100,
                )
                loss = loss / cfg.grad_accum_steps

            loss.backward()
            running_loss += loss.item() * cfg.grad_accum_steps
            loss_steps += 1

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

        # LR update
        lr = get_lr(step, cfg)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        optimizer.step()
        optimizer.zero_grad()
        step += 1
        pbar.update(1)

        # Logging
        if step % 10 == 0:
            avg_loss = running_loss / loss_steps
            tok_per_sec = (cfg.batch_size * cfg.grad_accum_steps * cfg.seq_len * 10) / (time.time() - t0)
            pbar.set_postfix({"loss": f"{avg_loss:.3f}", "lr": f"{lr:.2e}", "tok/s": f"{tok_per_sec:,.0f}"})
            t0 = time.time()
            running_loss = 0.0
            loss_steps = 0

        # Evaluation / checkpointing
        if step % cfg.save_interval == 0:
            ckpt_path = Path(cfg.out_dir) / f"ckpt_step_{step:06d}.pt"
            torch.save({
                "step": step,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "config": llama_cfg.to_dict(),
            }, ckpt_path)
            print(f"\n💾 Saved checkpoint: {ckpt_path}")

    pbar.close()

    # Save final model
    final_dir = Path(cfg.out_dir) / "final"
    final_dir.mkdir(exist_ok=True)
    model.config.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    # Save weights in safetensors format if possible
    try:
        from safetensors.torch import save_file
        state_dict = model.state_dict()
        save_file(state_dict, final_dir / "model.safetensors")
        print(f"✅ Saved final model to {final_dir}")
    except ImportError:
        torch.save(model.state_dict(), final_dir / "pytorch_model.bin")
        print(f"✅ Saved final model (bin format) to {final_dir}")

    print("🎉 Training complete!")


if __name__ == "__main__":
    main()
