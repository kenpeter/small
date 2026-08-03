#!/usr/bin/env python3
"""
Pure-GPU pretraining for the 1B model — replaces CPUMaster offload.

Removes the serial CPU<->GPU layer-streaming bottleneck (measured: 1 hot
CPU thread, ~2.3K tok/s). Model + AdamW moments live on GPU in bf16
(12GB VRAM budget: 2.1GB params + 4.1GB moments + 2.1GB grads + ~1GB
activations with gradient checkpointing). Effective batch identical to
the CPUMaster run: batch 4 × accum 8 × seq 2048 = 65K tokens/step.

Fresh start only (optimizer family differs from CPUMaster fp32 AdamW).
"""
import argparse, logging, os, time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaConfig

from pretrain_megatrain import (
    FARM_DIR,
    FlatFarmDataset,
    collate_pretrain,
    get_lr,
    save_checkpoint_robust,
)

logger = logging.getLogger("pretrain_gpu")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

SEQ_LEN = 2048
OUTPUT_DIR = Path("/home/kenpeter/work/checkpoints")
LATEST = OUTPUT_DIR / "megatrain_latest.pt"
BEST = OUTPUT_DIR / "megatrain_best.pt"


def build_model(dtype: torch.dtype) -> torch.nn.Module:
    hf_config = LlamaConfig(
        vocab_size=49152,
        hidden_size=1536,
        intermediate_size=4608,
        num_hidden_layers=32,
        num_attention_heads=12,
        num_key_value_heads=4,
        max_position_embeddings=8192,
        rope_theta=10000.0,
        rms_norm_eps=1e-5,
        hidden_act="silu",
        tie_word_embeddings=False,
        attention_bias=False,
        mlp_bias=False,
        initializer_range=0.02,
        torch_dtype=dtype,
        head_dim=128,
        architectures=["LlamaForCausalLM"],
    )
    model = AutoModelForCausalLM.from_config(
        hf_config, dtype=dtype, trust_remote_code=True, attn_implementation="sdpa"
    )
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    n = sum(p.numel() for p in model.parameters())
    logger.info(f"Model: {n:,} params ({n/1e9:.2f}B), dtype={dtype}")
    return model


def make_optimizer(model, base_lr: float):
    params = list(model.parameters())
    vocab_embed_numel = 49152 * 1536
    g2d = [p for p in params if p.ndim >= 2 and p.numel() != vocab_embed_numel]
    geh = [p for p in params if p.ndim >= 2 and p.numel() == vocab_embed_numel]
    gsc = [p for p in params if p.ndim < 2]
    groups = [
        dict(params=g2d, lr=base_lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1),
        dict(params=geh, lr=base_lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1),
        dict(params=gsc, lr=base_lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0),
    ]
    return torch.optim.AdamW(groups)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--num-steps", type=int, default=60000)
    parser.add_argument("--max-seq-len", type=int, default=SEQ_LEN)
    parser.add_argument("--log-interval", type=int, default=400)
    parser.add_argument("--save-interval", type=int, default=3000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--init-from", type=str, default=None,
                        help="Optional checkpoint .pt to load MODEL WEIGHTS from "
                             "(warm start; optimizer starts fresh — bf16 family)")
    args = parser.parse_args()

    torch.manual_seed(42)
    device = "cuda"

    model = build_model(torch.bfloat16).to(device)
    if args.init_from and os.path.exists(args.init_from):
        ckpt = torch.load(args.init_from, map_location="cpu", weights_only=False)
        sd = ckpt["model_state_dict"]
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing:
            logger.warning(f"init-from: missing keys ({len(missing)}): {missing[:5]}")
        if unexpected:
            logger.warning(f"init-from: unexpected keys ({len(unexpected)}): {unexpected[:5]}")
        logger.info(f"🔥 Warm-started from {args.init_from} (step {ckpt.get('step', '?')})")
        del ckpt, sd
        torch.cuda.empty_cache()
    optimizer = make_optimizer(model, args.lr)

    ds = FlatFarmDataset(FARM_DIR, seq_len=args.max_seq_len, val_frac=0.01, is_val=False)
    dl = DataLoader(ds, batch_size=args.batch_size, collate_fn=collate_pretrain,
                    shuffle=False, num_workers=0, pin_memory=True)
    data_iter = iter(dl)

    logger.info("Starting pure-GPU pretraining from scratch (bf16 AdamW, grad checkpointing)...")
    best_loss = float("inf")
    global_step = 0
    running = 0.0
    t0 = time.time()

    for step in range(args.num_steps):
        lr = get_lr(step + 1, args.warmup_steps, args.num_steps, args.lr, args.min_lr)
        for g in optimizer.param_groups:
            g["lr"] = lr

        acc_loss = 0.0
        for _ in range(args.grad_accum):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dl)
                batch = next(data_iter)
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attn = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            out = model(input_ids=input_ids, attention_mask=attn, labels=labels)
            loss = out.loss / args.grad_accum
            acc_loss += loss.item()
            loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        global_step += 1
        running += acc_loss

        if (step + 1) % args.log_interval == 0:
            dt = time.time() - t0
            t0 = time.time()
            avg = running / args.log_interval
            running = 0.0
            tps = args.batch_size * args.max_seq_len * args.grad_accum * args.log_interval / dt
            mem = torch.cuda.max_memory_allocated(device) / 1024**3
            logger.info(
                f"Step {step+1}/{args.num_steps} | Loss {avg:.4f} | LR {lr:.2e} | "
                f"{dt/args.log_interval:.2f}s/step | {tps:.0f} tok/s | GPU {mem:.2f}GB"
            )

        if (step + 1) % args.save_interval == 0 or step == args.num_steps - 1:
            is_best = acc_loss < best_loss
            if is_best:
                best_loss = acc_loss
            state = {
                "step": step + 1,
                "loss": acc_loss,
                "best_loss": best_loss,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": args.__dict__,
            }
            save_checkpoint_robust(state, OUTPUT_DIR, is_best, logger)

    logger.info("✅ Pure-GPU pretraining complete!")


if __name__ == "__main__":
    main()
