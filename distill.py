"""
Distillation: Train student (our model) to mimic SmolLM2-135M-Instruct teacher.
KL divergence on logits — response tokens only.
"""
import os, sys, json, time, math, gc
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM

# ─── Student architecture (same as dpo.py) ─────────────────────────
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
    if n_rep == 1: return x
    return x.unsqueeze(2).expand(b, n_kv, n_rep, h, d).reshape(b, n_kv * n_rep, h, d)

class CausalSelfAttention(nn.Module):
    def __init__(self, cfg):
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
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        q = self.rope(q, T); k = self.rope(k, T)
        k = repeat_kv(k, self.n_rep); v = repeat_kv(v, self.n_rep)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.o_proj(y)

class SwiGLUMLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.dim, cfg.intermediate_size, bias=False)
        self.up_proj = nn.Linear(cfg.dim, cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.dim, bias=False)
    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.dim, cfg.rms_norm_eps)
        self.attn = CausalSelfAttention(cfg)
        self.mlp_norm = RMSNorm(cfg.dim, cfg.rms_norm_eps)
        self.mlp = SwiGLUMLP(cfg)
    def forward(self, x):
        x = x + self.attn(self.attn_norm(x))
        x = x + self.mlp(self.mlp_norm(x))
        return x

class SmolLM2(nn.Module):
    def __init__(self, cfg):
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

# ─── Config ─────────────────────────────────────────────────────
@dataclass
class DistillConfig:
    vocab_size: int = 49152
    dim: int = 576
    n_layers: int = 30
    n_heads: int = 9
    n_kv_heads: int = 3
    intermediate_size: int = 1536
    max_seq_len: int = 8192
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-5
    dropout: float = 0.0

    student_checkpoint: Path = Path("/home/kenpeter/work/checkpoints/checkpoint_best.pt")
    sft_data_dir: Path = Path("/home/kenpeter/work/data/_sft_staging")
    output_dir: Path = Path("/home/kenpeter/work/checkpoints")
    seq_len: int = 1024
    batch_size: int = 2
    gradient_accumulation_steps: int = 4
    max_steps: int = 20000
    learning_rate: float = 2e-5
    min_lr: float = 1e-6
    warmup_steps: int = 100
    weight_decay: float = 0.1
    max_grad_norm: float = 1.0
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    distill_temp: float = 2.0
    save_every_n_steps: int = 1000
    log_every_n_steps: int = 10
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

# ─── Load SFT data (chat format) ────────────────────────────────
def load_sft_data(data_dir: Path, max_samples: int = 50000):
    samples = []
    for subdir in sorted(data_dir.iterdir()):
        if not subdir.is_dir(): continue
        for jfile in sorted(subdir.glob("shard_*.jsonl")):
            with open(jfile, "r", encoding="utf-8") as f:
                for line in f:
                    try: ex = json.loads(line)
                    except: continue
                    samples.append(ex)
                    if len(samples) >= max_samples: break
            if len(samples) >= max_samples: break
        if len(samples) >= max_samples: break
    print(f"Loaded {len(samples)} SFT samples")
    return samples

def normalize_sample(ex):
    """Convert any SFT format to standard messages list."""
    # Format 1: messages (standard)
    if "messages" in ex:
        return ex["messages"]
    # Format 2: conversations (OpenHermes)
    if "conversations" in ex:
        msgs = []
        for c in ex["conversations"]:
            role_map = {"human": "user", "gpt": "assistant", "system": "system"}
            msgs.append({"role": role_map.get(c["from"], "user"), "content": c["value"]})
        return msgs
    # Format 3: instruction/input/output (Alpaca, Code-Alpaca)
    if "instruction" in ex:
        inp = ex.get("input", "").strip()
        inst = ex["instruction"].strip()
        user_text = f"{inst}\n{inp}" if inp else inst
        return [
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": ex.get("output", "").strip()},
        ]
    # Format 4: system_prompt/question/response (OpenOrca)
    if "question" in ex:
        msgs = []
        if ex.get("system_prompt"):
            msgs.append({"role": "system", "content": ex["system_prompt"]})
        msgs.append({"role": "user", "content": ex["question"]})
        msgs.append({"role": "assistant", "content": ex.get("response", "")})
        return msgs
    # Format 5: data as list of alternating strings (UltraChat)
    if "data" in ex and isinstance(ex["data"], list) and len(ex["data"]) > 1:
        msgs = []
        for i, turn in enumerate(ex["data"]):
            role = "assistant" if i % 2 == 1 else "user"
            msgs.append({"role": role, "content": str(turn)})
        return msgs
    return None

class DistillDataset(Dataset):
    def __init__(self, samples: list, tokenizer, max_len: int = 1024):
        self.input_ids_list = []
        self.mask_list = []
        for ex in samples:
            msgs = normalize_sample(ex)
            if not msgs: continue
            text, mask = self._format_with_mask(msgs, tokenizer, max_len)
            self.input_ids_list.append(text)
            self.mask_list.append(mask)
        print(f"  → {len(self.input_ids_list)} usable samples")

    def _format_with_mask(self, msgs, tokenizer, max_len):
        """Format chat, return (token_ids, mask) where mask=1 for assistant tokens."""
        parts = []
        mask = []
        for msg in msgs:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                t = f"<|im_start|>system\n{content}<|im_end|>\n"
                ids = tokenizer(t, add_special_tokens=False)["input_ids"]
                parts.extend(ids); mask.extend([0]*len(ids))
            elif role == "user":
                t = f"<|im_start|>user\n{content}<|im_end|>\n"
                ids = tokenizer(t, add_special_tokens=False)["input_ids"]
                parts.extend(ids); mask.extend([0]*len(ids))
            elif role == "assistant":
                t = f"<|im_start|>assistant\n{content}<|im_end|>"
                ids = tokenizer(t, add_special_tokens=False)["input_ids"]
                parts.extend(ids); mask.extend([1]*len(ids))
        if len(parts) > max_len:
            parts = parts[:max_len]; mask = mask[:max_len]
        else:
            pad = tokenizer.pad_token_id or 0
            parts += [pad]*(max_len - len(parts))
            mask += [0]*(max_len - len(mask))
        return torch.tensor(parts, dtype=torch.long), torch.tensor(mask, dtype=torch.bool)

    def __len__(self):
        return len(self.input_ids_list)

    def __getitem__(self, idx):
        return {"input_ids": self.input_ids_list[idx], "mask": self.mask_list[idx]}

# ─── Distillation loss (KL divergence) ────────────────────────────
def distill_loss(student_logits, teacher_logits, mask, temp=2.0):
    """KL divergence between teacher and student distributions over response tokens."""
    T = temp
    s_logits = student_logits / T
    t_logits = teacher_logits / T
    s_probs = F.log_softmax(s_logits, dim=-1)
    t_probs = F.softmax(t_logits, dim=-1)
    # Element-wise KL: sum(t_probs * (log(t_probs) - log(s_probs)))
    kl = F.kl_div(s_probs, t_probs, reduction="none", log_target=False)
    kl = kl.sum(dim=-1)  # (B, T)
    # Apply mask (only response tokens)
    kl = kl * mask.float()
    loss = kl.sum() / mask.float().sum().clamp(min=1)
    return loss * (T ** 2)  # Scale by T^2 as per standard distillation

# ─── Training ─────────────────────────────────────────────────────
def train_distill(cfg: DistillConfig):
    print(f"Device: {cfg.device}")
    print(f"Student checkpoint: {cfg.student_checkpoint}")

    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M", trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load teacher (transformers)
    print("Loading teacher (SmolLM2-135M-Instruct)...")
    teacher = AutoModelForCausalLM.from_pretrained(
        "HuggingFaceTB/SmolLM2-135M-Instruct",
        torch_dtype=torch.bfloat16,
    ).to(cfg.device)
    for p in teacher.parameters():
        p.requires_grad = False
    teacher.eval()
    print(f"Teacher params: {sum(p.numel() for p in teacher.parameters()):,}")

    # Load student (our architecture)
    print("Loading student...")
    student = SmolLM2(cfg).to(cfg.device)
    if cfg.student_checkpoint.exists():
        state = torch.load(cfg.student_checkpoint, map_location=cfg.device, weights_only=False)
        sd = state["model_state_dict"]
        if any(k.startswith("_orig_mod.") for k in sd.keys()):
            sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
        student.load_state_dict(sd)
        print(f"  Loaded step {state.get('step', 'unknown')} | loss {state.get('loss', 'unknown')}")
    else:
        print("  ⚠ No checkpoint — starting from scratch")

    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=cfg.learning_rate, betas=(cfg.beta1, cfg.beta2),
        eps=cfg.eps, weight_decay=cfg.weight_decay,
    )
    print(f"Student params: {sum(p.numel() for p in student.parameters()):,}")

    # Data
    samples = load_sft_data(cfg.sft_data_dir)
    if not samples:
        print("No SFT data. Exiting.")
        return
    dataset = DistillDataset(samples, tokenizer, cfg.seq_len)
    dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=0)
    data_iter = iter(dataloader)
    print(f"Batches: ~{len(dataset) // cfg.batch_size}")

    use_amp = cfg.device == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    def get_lr(step):
        if step < cfg.warmup_steps:
            return cfg.learning_rate * (step + 1) / cfg.warmup_steps
        if step >= cfg.max_steps:
            return cfg.min_lr
        decay_ratio = (step - cfg.warmup_steps) / (cfg.max_steps - cfg.warmup_steps)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return cfg.min_lr + coeff * (cfg.learning_rate - cfg.min_lr)

    t0 = time.time()
    running_loss = 0.0
    step = 0
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")

    while step < cfg.max_steps:
        step += 1
        lr = get_lr(step)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        accumulated_loss = 0.0
        for micro_step in range(cfg.gradient_accumulation_steps):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)

            input_ids = batch["input_ids"].to(cfg.device)
            mask = batch["mask"].to(cfg.device)

            with torch.autocast(device_type=cfg.device, dtype=torch.bfloat16 if use_amp else torch.float32):
                # Student forward
                s_logits, _ = student(input_ids)
                # Teacher forward (no grad)
                with torch.no_grad():
                    t_out = teacher(input_ids)
                    t_logits = t_out.logits

                loss = distill_loss(s_logits, t_logits, mask, cfg.distill_temp)
                loss = loss / cfg.gradient_accumulation_steps

            accumulated_loss += loss.item()
            scaler.scale(loss).backward()

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(student.parameters(), cfg.max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        running_loss += accumulated_loss

        if step % cfg.log_every_n_steps == 0:
            dt = time.time() - t0
            t0 = time.time()
            avg_loss = running_loss / cfg.log_every_n_steps
            print(f"step {step:5d} | distill_loss {avg_loss:.4f} | lr {lr:.2e} | {dt:.1f}s")
            running_loss = 0.0

        if step % cfg.save_every_n_steps == 0:
            val_loss = avg_loss
            is_best = val_loss < best_loss
            if is_best: best_loss = val_loss
            state = {
                "step": step, "loss": val_loss, "best_loss": best_loss,
                "model_state_dict": student.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": cfg.__dict__,
            }
            tmp = cfg.output_dir / "distill_latest.tmp"
            torch.save(state, tmp)
            os.replace(tmp, cfg.output_dir / "distill_latest.pt")
            if is_best:
                os.replace(cfg.output_dir / "distill_latest.pt", cfg.output_dir / "distill_best.pt")
                print(f"  ⭐ Saved best distill (loss {val_loss:.4f})")
            else:
                print(f"  💾 Saved latest distill (loss {val_loss:.4f})")

    state = {
        "step": step, "loss": running_loss, "best_loss": best_loss,
        "model_state_dict": student.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": cfg.__dict__,
    }
    torch.save(state, cfg.output_dir / "distill_final.pt")
    print(f"\n✅ Distillation complete. distill_final.pt | best={best_loss:.4f}")

if __name__ == "__main__":
    cfg = DistillConfig()
    train_distill(cfg)
