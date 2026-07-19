"""
SmolLM2-0.85B pretraining with DeepSpeed ZeRO-3 + CPU offload.
Fits 12GB VRAM with batch=4+ at seq=2048.
"""
import os, sys, json, time, math, glob, gc, hashlib
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from torch.utils.data import IterableDataset, DataLoader
from torch.utils.checkpoint import checkpoint
from transformers import AutoTokenizer

import deepspeed
from deepspeed.ops.adam import DeepSpeedCPUAdam

# ─── Config ───────────────────────────────────────────────────
@dataclass
class TrainConfig:
    vocab_size: int = 49152
    dim: int = 1536
    n_layers: int = 28
    n_heads: int = 12
    n_kv_heads: int = 4
    intermediate_size: int = 4096
    max_seq_len: int = 8192
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-5
    dropout: float = 0.0

    batch_size: int = 4
    gradient_accumulation_steps: int = 6
    max_steps: int = 100_000
    learning_rate: float = 4e-4
    min_lr: float = 1e-4
    warmup_steps: int = 2000
    weight_decay: float = 0.1
    max_grad_norm: float = 1.0
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    save_every_n_steps: int = 2000
    log_every_n_steps: int = 10
    val_every_n_steps: int = 500
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    use_gradient_checkpointing: bool = True
    compile: bool = False
    seq_len: int = 2048
    val_frac: float = 0.01

    checkpoint_dir: Path = Path("/home/kenpeter/work/checkpoints")
    data_dir: Path = Path("/home/kenpeter/work/data/_shards_final")


# ─── Architecture (same as train.py) ──────────────────────────
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 8192, theta: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        t = torch.arange(max_seq_len, device=inv_freq.device)
        freqs = torch.outer(t, inv_freq)
        self.register_buffer("cos", torch.cos(freqs), persistent=False)
        self.register_buffer("sin", torch.sin(freqs), persistent=False)

    def forward(self, x, seq_len: int):
        cos = self.cos[:seq_len, :].repeat_interleave(2, dim=-1)
        sin = self.sin[:seq_len, :].repeat_interleave(2, dim=-1)
        x1, x2 = x[..., ::2], x[..., 1::2]
        rotated = torch.stack([-x2, x1], dim=-1).flatten(-2)
        return x * cos + rotated * sin


def repeat_kv(x, n_rep: int):
    b, n_kv, h, d = x.shape
    if n_rep == 1:
        return x
    return x.unsqueeze(2).expand(b, n_kv, n_rep, h, d).reshape(b, n_kv * n_rep, h, d)


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: TrainConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.n_rep = cfg.n_heads // cfg.n_kv_heads
        self.head_dim = cfg.dim // cfg.n_heads

        self.q_proj = nn.Linear(cfg.dim, cfg.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.dim, cfg.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.dim, cfg.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(cfg.n_heads * self.head_dim, cfg.dim, bias=False)
        self.rope = RotaryEmbedding(self.head_dim, cfg.max_seq_len, cfg.rope_theta)

    def forward(self, x):
        B, T, C = x.size()
        q = self.q_proj(x).reshape(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).reshape(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).reshape(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        q = self.rope(q, T)
        k = self.rope(k, T)

        k = repeat_kv(k, self.n_rep)
        v = repeat_kv(v, self.n_rep)

        y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.o_proj(y)


class SwiGLUMLP(nn.Module):
    def __init__(self, cfg: TrainConfig):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.dim, cfg.intermediate_size, bias=False)
        self.up_proj = nn.Linear(cfg.dim, cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.dim, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    def __init__(self, cfg: TrainConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.dim, cfg.rms_norm_eps)
        self.attn = CausalSelfAttention(cfg)
        self.mlp_norm = RMSNorm(cfg.dim, cfg.rms_norm_eps)
        self.mlp = SwiGLUMLP(cfg)
        self.use_checkpoint = getattr(cfg, "use_gradient_checkpointing", False)

    def forward(self, x):
        if self.use_checkpoint and self.training:
            x = x + torch.utils.checkpoint.checkpoint(self.attn, self.attn_norm(x), use_reentrant=False)
            x = x + torch.utils.checkpoint.checkpoint(self.mlp, self.mlp_norm(x), use_reentrant=False)
        else:
            x = x + self.attn(self.attn_norm(x))
            x = x + self.mlp(self.mlp_norm(x))
        return x


class SmolLM2(nn.Module):
    def __init__(self, cfg: TrainConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_embeddings = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.layers = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.dim, cfg.rms_norm_eps)
        self.lm_head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        x = self.tok_embeddings(idx)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100)
        return logits, loss

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())


# ─── Data ─────────────────────────────────────────────────────
class BinShardDataset(IterableDataset):
    def __init__(self, data_dir: Path, seq_len: int, val_frac: float = 0.0, is_val: bool = False):
        self.seq_len = seq_len
        self.shards = sorted(data_dir.glob("shard_*.bin"))
        if not self.shards:
            raise RuntimeError(f"No .bin shards found in {data_dir}")

        self.val_frac = val_frac
        self.is_val = is_val

        sorted_shards = sorted(self.shards, key=lambda p: hashlib.md5(str(p).encode()).hexdigest())
        n_val = max(1, int(len(sorted_shards) * val_frac))
        if is_val:
            self.shards = sorted_shards[-n_val:]
        else:
            self.shards = sorted_shards[:-n_val] if n_val > 0 else sorted_shards

        self.token_dtype = np.uint16
        self.tokens_per_shard = self.shards[0].stat().st_size // np.dtype(self.token_dtype).itemsize

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            shards = self.shards
        else:
            per_worker = len(self.shards) // worker_info.num_workers
            start = worker_info.id * per_worker
            end = start + per_worker if worker_info.id < worker_info.num_workers - 1 else len(self.shards)
            shards = self.shards[start:end]

        for shard_path in shards:
            tokens = np.memmap(str(shard_path), dtype=self.token_dtype, mode="r")
            n = len(tokens) - self.seq_len
            offset = (hash(str(shard_path)) % max(1, n)) if n > 0 else 0
            for i in range(offset, n, self.seq_len):
                chunk = tokens[i : i + self.seq_len + 1]
                if len(chunk) < self.seq_len + 1:
                    continue
                x = torch.from_numpy(chunk[:-1].astype(np.int64))
                y = torch.from_numpy(chunk[1:].astype(np.int64))
                yield x, y
            del tokens
            gc.collect()


def get_dataloader(cfg: TrainConfig, is_val: bool = False):
    ds = BinShardDataset(cfg.data_dir, cfg.seq_len, cfg.val_frac, is_val)
    return DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0, pin_memory=True)


# ─── DeepSpeed Config ─────────────────────────────────────────
def make_ds_config(cfg: TrainConfig):
    """Build DeepSpeed ZeRO-3 + CPU offload config."""
    return {
        "train_batch_size": cfg.batch_size * cfg.gradient_accumulation_steps,
        "gradient_accumulation_steps": cfg.gradient_accumulation_steps,
        "fp16": {
            "enabled": True,
            "loss_scale": 0,
            "loss_scale_window": 1000,
            "initial_scale_power": 16,
            "hysteresis": 2,
            "min_loss_scale": 1,
        },
        "bf16": {
            "enabled": False,
        },
        "zero_optimization": {
            "stage": 3,
            "offload_optimizer": {
                "device": "cpu",
                "pin_memory": True,
            },
            "offload_param": {
                "device": "cpu",
                "pin_memory": True,
            },
            "overlap_comm": True,
            "contiguous_gradients": True,
            "reduce_bucket_size": 5e7,
            "stage3_prefetch_bucket_size": 5e7,
            "stage3_param_persistence_threshold": 1e5,
            "sub_group_size": 1e9,
            "reduce_scatter": True,
        },
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": cfg.learning_rate,
                "betas": [cfg.beta1, cfg.beta2],
                "eps": cfg.eps,
                "weight_decay": cfg.weight_decay,
            },
        },
        "scheduler": {
            "type": "WarmupCosineLR",
            "params": {
                "warmup_min_lr": 0,
                "warmup_max_lr": cfg.learning_rate,
                "warmup_num_steps": cfg.warmup_steps,
                "total_num_steps": cfg.max_steps,
            },
        },
        "gradient_clipping": cfg.max_grad_norm,
        "steps_per_print": cfg.log_every_n_steps,
        "wall_clock_breakdown": False,
    }


# ─── Eval ─────────────────────────────────────────────────────
@torch.no_grad()
def generate_sample(model, tokenizer, prompt: str, max_new: int = 20, device: str = "cuda"):
    model.eval()
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    for _ in range(max_new):
        logits, _ = model(input_ids)
        next_tok = logits[0, -1, :].argmax(dim=-1, keepdim=True)
        input_ids = torch.cat([input_ids, next_tok.unsqueeze(0)], dim=1)
    out = tokenizer.decode(input_ids[0].tolist())
    model.train()
    return out


REAL_PROMPTS = [
    "The capital of France is",
    "1 + 1 =",
    "The sky is",
    "Water boils at",
    "Once upon a time",
]


@torch.no_grad()
def estimate_loss(model, val_loader, max_batches: int = 50):
    model.eval()
    losses = []
    for i, (x, y) in enumerate(val_loader):
        if i >= max_batches:
            break
        x, y = x.cuda(), y.cuda()
        _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses) if losses else float("inf")


# ─── Training ─────────────────────────────────────────────────
def train(cfg: TrainConfig):
    print(f"Device: {cfg.device}")
    print(f"Data dir: {cfg.data_dir}")
    print(f"Checkpoint dir: {cfg.checkpoint_dir}")

    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M", trust_remote_code=True)
    print(f"Tokenizer vocab size: {len(tokenizer)}")

    # Create model on CPU (DeepSpeed will offload to GPU as needed)
    model = SmolLM2(cfg)
    n_params = model.count_parameters()
    print(f"Model params: {n_params:,} ({n_params/1e9:.3f}B)")

    # DeepSpeed config
    ds_config = make_ds_config(cfg)
    print(f"DeepSpeed ZeRO-3 + CPU offload")
    print(f"  batch_size={cfg.batch_size}, grad_accum={cfg.gradient_accumulation_steps}")
    print(f"  effective batch={cfg.batch_size * cfg.gradient_accumulation_steps * cfg.seq_len:,} tok/step")

    # Initialize DeepSpeed engine
    engine = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        config_params=ds_config,
    )[0]
    print(f"  Initialized engine")

    # Data
    train_loader = get_dataloader(cfg, is_val=False)
    val_loader = get_dataloader(cfg, is_val=True)
    train_iter = iter(train_loader)

    print(f"Train shards: {len(train_loader.dataset.shards)}")
    print(f"Val shards:   {len(val_loader.dataset.shards)}")

    # Training loop
    t0 = time.time()
    running_loss = 0.0
    step = 0
    val_interval = cfg.val_every_n_steps
    best_loss = float("inf")
    stall_count = 0

    while step < cfg.max_steps:
        step += 1

        # Gradient accumulation micro-batches
        accumulated_loss = 0.0
        for micro_step in range(cfg.gradient_accumulation_steps):
            try:
                x, y = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                x, y = next(train_iter)

            x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)

            output = engine(x, y)
            loss = output[1] if isinstance(output, tuple) else output
            engine.backward(loss)

            accumulated_loss += loss.item()

        engine.step()

        running_loss += accumulated_loss

        # Logging
        if step % cfg.log_every_n_steps == 0:
            dt = time.time() - t0
            t0 = time.time()
            tokens_per_sec = (cfg.batch_size * cfg.seq_len * cfg.gradient_accumulation_steps * cfg.log_every_n_steps) / dt
            avg_loss = running_loss / cfg.log_every_n_steps
            print(f"step {step:6d} | loss {avg_loss:.4f} | {dt:.1f}s | {tokens_per_sec:,.0f} tok/s")
            running_loss = 0.0

            # Memory info
            if torch.cuda.is_available():
                alloc = torch.cuda.memory_allocated() / 1024**3
                reserved = torch.cuda.memory_reserved() / 1024**3
                print(f"  │ GPU: {alloc:.2f}GB alloc / {reserved:.2f}GB reserved")

        # Validation
        if step % val_interval == 0:
            internal_model = engine.module if hasattr(engine, 'module') else engine
            val_loss = estimate_loss(internal_model, val_loader)
            print(f"  📊 Val loss: {val_loss:.4f} | best: {best_loss:.4f}")
            test_prompt = REAL_PROMPTS[step % len(REAL_PROMPTS)]
            gen = generate_sample(internal_model, tokenizer, test_prompt, max_new=15, device=cfg.device)
            print(f"  💬 Gen: {gen[:80]}")
            if val_loss < best_loss:
                best_loss = val_loss
                stall_count = 0
            else:
                stall_count += 1
                if stall_count >= 3 and val_interval > 50:
                    val_interval = max(50, val_interval // 2)
                    print(f"  ⚠ Plateau → eval every {val_interval} steps")

        # Checkpoint
        if step % cfg.save_every_n_steps == 0:
            ckpt_dir = str(cfg.checkpoint_dir)
            engine.save_checkpoint(ckpt_dir, tag=f"step_{step}")
            print(f"  💾 Saved checkpoint (step {step})")

    # Final save
    ckpt_dir = str(cfg.checkpoint_dir)
    engine.save_checkpoint(ckpt_dir, tag=f"step_{step}")
    print(f"\n✅ Training complete.")


if __name__ == "__main__":
    cfg = TrainConfig()
    train(cfg)
