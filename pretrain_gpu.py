#!/usr/bin/env python3
"""
Pure-GPU pretraining for the 1B model — replaces CPUMaster offload.

Removes the serial CPU<->GPU layer-streaming bottleneck (measured: 1 hot
CPU thread, ~2.3K tok/s). Model + AdamW moments live on GPU in bf16
(12GB VRAM budget: 2.1GB params + 4.1GB moments + 2.1GB grads + ~1GB
activations with gradient checkpointing). Effective batch: 2 × accum 16
× seq 2048 = 65K tokens/step.

Loss is computed with chunked fp32 cross-entropy (512-token slices) to
avoid HF's full-logits fp32 spike — peak GPU 10.2GB, safely under 11.59GB.

Supports --init-from warm-start (checkpoint loaded on CPU, freed after
load_state_dict — the OOM fix).
"""
import argparse, logging, os, time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaConfig

from pretrain_megatrain import (
    FARM_DIR,
    SHARD_DIRS,
    FlatFarmDataset,
    StratifiedShardDataset,
    collate_pretrain,
    get_lr,
    get_curriculum_ratios,
    CURRICULUM_UPDATE_INTERVAL,
    save_checkpoint_robust,
    should_save_checkpoint,
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
    parser.add_argument("--save-every-minutes", type=int, default=0,
                        help="Also save every N wall-clock minutes (0 = step-based only). "
                             "Keeps worst-case loss window small regardless of step speed.")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--init-from", type=str, default=None,
                        help="Optional checkpoint .pt to load MODEL WEIGHTS from "
                             "(warm start; optimizer starts fresh — bf16 family)")
    parser.add_argument("--curriculum", action="store_true",
                        help="Enable G1-G4 curriculum (StratifiedShardDataset + "
                             "get_curriculum_ratios). Default: flat farm.")
    parser.add_argument("--compile", action="store_true",
                        help="Wrap model in torch.compile (fused kernels — first steps "
                             "slower due to JIT warmup, then faster).")
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
    if args.compile:
        logger.info("⚡ Compiling model with torch.compile (JIT warmup on first steps)...")
        model = torch.compile(model)
    optimizer = make_optimizer(model, args.lr)

    if args.curriculum:
        ratios0 = get_curriculum_ratios(0, args.num_steps)
        logger.info(f"📚 G1-G4 curriculum ON — start ratios: {ratios0}")
        ds = StratifiedShardDataset(SHARD_DIRS, seq_len=args.max_seq_len,
                                    ratios=ratios0, dedup=False)
    else:
        logger.info(f"Flat farm: {FARM_DIR}")
        ds = FlatFarmDataset(FARM_DIR, seq_len=args.max_seq_len, val_frac=0.01, is_val=False)
    dl = DataLoader(ds, batch_size=args.batch_size, collate_fn=collate_pretrain,
                    shuffle=False, num_workers=0, pin_memory=True)
    data_iter = iter(dl)

    logger.info("Starting pure-GPU pretraining from scratch (bf16 AdamW, grad checkpointing)...")
    best_loss = float("inf")
    global_step = 0
    running = 0.0
    t0 = time.time()
    last_save_time = time.time()

    for step in range(args.num_steps):
        # ── G1-G4 curriculum: rebuild tier ratios every CURRICULUM_UPDATE_INTERVAL ──
        # G1+G3: smooth easy→hard weights at t=step/total (no cliffs)
        # G2: cosine easy-review boost + renormalize (inside get_curriculum_ratios)
        # G4: windowed JIT shuffle (inside _build_stratified_order)
        if args.curriculum and step > 0 and step % CURRICULUM_UPDATE_INTERVAL == 0:
            new_ratios = get_curriculum_ratios(step, args.num_steps)
            ds.ratios = new_ratios
            ds._build_stratified_order()
            data_iter = iter(dl)
            logger.info(f"📚 G1-G4 curriculum @ step {step}: {new_ratios}")
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

            out = model(input_ids=input_ids, attention_mask=attn)
            logits = out.logits  # bf16 [B, S, V] — no labels → HF skips its fp32 loss
            # Chunked fp32 cross-entropy: identical math to HF's loss, but the
            # fp32 logits conversion happens in 512-token slices (~100MB) instead
            # of one 768MB block — keeps the run safely under the 11.59GB ceiling.
            loss = None
            B, S, V = logits.shape
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            n_tok = shift_labels.numel()
            for i in range(0, S - 1, 512):
                ce = torch.nn.functional.cross_entropy(
                    shift_logits[:, i : i + 512, :].float().reshape(-1, V),
                    shift_labels[:, i : i + 512].reshape(-1),
                    reduction="sum",
                    ignore_index=-100,
                )
                loss = ce if loss is None else loss + ce
            loss = loss / n_tok / args.grad_accum
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

        if should_save_checkpoint(step + 1, args.save_interval, last_save_time,
                                  time.time(), args.save_every_minutes) or step + 1 == args.num_steps:
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
            last_save_time = time.time()

    logger.info("✅ Pure-GPU pretraining complete!")


if __name__ == "__main__":
    main()
