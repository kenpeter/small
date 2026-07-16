"""
DPO (Direct Preference Optimization) for SmolLM2-135M.
Loads SFT checkpoint, aligns on preference pairs {prompt, chosen, rejected}.
No separate reward model needed — DPO directly optimizes the policy.
Uses pre-downloaded DPO data from _dpo_staging/.
"""
import os, sys, json, time, math, gc
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

# ─── Reuse architecture ─────────────────────────────────────────
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
class DPOConfig:
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

    sft_checkpoint: Path = Path("/home/kenpeter/work/checkpoints/sft_best.pt")
    dpo_data_dir: Path = Path("/home/kenpeter/work/data/_dpo_staging")
    output_dir: Path = Path("/home/kenpeter/work/checkpoints")
    seq_len: int = 1024
    batch_size: int = 1  # Pairs are large; keep small
    gradient_accumulation_steps: int = 4
    max_steps: int = 5_000
    learning_rate: float = 5e-7  # Very low — DPO is sensitive
    beta: float = 0.1  # KL divergence penalty weight
    min_lr: float = 1e-7
    warmup_steps: int = 100
    weight_decay: float = 0.1
    max_grad_norm: float = 1.0
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    save_every_n_steps: int = 1000
    log_every_n_steps: int = 10
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

# ─── Load DPO preference pairs ────────────────────────────────────
def load_dpo_pairs(data_dir: Path, max_pairs: int = 50_000):
    """Load {prompt, chosen, rejected} from JSONL shards."""
    pairs = []
    for subdir in sorted(data_dir.iterdir()):
        if not subdir.is_dir():
            continue
        shard_files = sorted(subdir.glob("shard_*.jsonl"))
        print(f"[{subdir.name}] {len(shard_files)} files")
        for jfile in shard_files:
            with open(jfile, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        ex = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    # Support multiple DPO formats
                    prompt = ex.get("prompt", ex.get("question", ""))
                    chosen = ex.get("chosen", ex.get("chatgpt_answer", ex.get("response_a", "")))
                    rejected = ex.get("rejected", ex.get("llama2-13b_answer", ex.get("response_b", "")))
                    if prompt and chosen and rejected:
                        pairs.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})
                    if len(pairs) >= max_pairs:
                        break
            if len(pairs) >= max_pairs:
                break
        if len(pairs) >= max_pairs:
            break
    print(f"  ✓ Total DPO pairs: {len(pairs)}")
    return pairs

# ─── Tokenize with chat template ──────────────────────────────────
def format_dpo(tokenizer, prompt: str, response: str, max_len: int = 2048):
    """Format prompt+response with chat template, return token IDs."""
    text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n{response}<|im_end|>"
    ids = tokenizer(text, add_special_tokens=False, truncation=True, max_length=max_len)["input_ids"]
    return torch.tensor(ids, dtype=torch.long)

class DPODataset(Dataset):
    def __init__(self, pairs: list, tokenizer, max_len: int = 2048):
        self.samples = []
        for p in pairs:
            prompt = p["prompt"]
            chosen = p["chosen"]
            rejected = p["rejected"]
            self.samples.append({
                "prompt_ids": tokenizer(f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n",
                                        add_special_tokens=False, truncation=True, max_length=max_len)["input_ids"],
                "chosen_ids": tokenizer(f"{chosen}<|im_end|>", add_special_tokens=False, truncation=True, max_length=max_len)["input_ids"],
                "rejected_ids": tokenizer(f"{rejected}<|im_end|>", add_special_tokens=False, truncation=True, max_length=max_len)["input_ids"],
            })
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        # Concatenate prompt + response for each branch
        chosen_full = s["prompt_ids"] + s["chosen_ids"]
        rejected_full = s["prompt_ids"] + s["rejected_ids"]
        # Pad to max_len
        def pad(ids):
            if len(ids) > self.max_len:
                return ids[:self.max_len]
            return ids + [self.tokenizer.pad_token_id or 0] * (self.max_len - len(ids))
        return {
            "chosen": torch.tensor(pad(chosen_full), dtype=torch.long),
            "rejected": torch.tensor(pad(rejected_full), dtype=torch.long),
            "prompt_len": len(s["prompt_ids"]),
        }

# ─── DPO helpers ──────────────────────────────────────────────────
def get_batch_logps(logits, tokens, mask):
    """Gather per-token log-probs, sum over mask positions, return per-sample avg."""
    logps = torch.log_softmax(logits, dim=-1)
    # Gather log-prob of actual token at each position
    B, T, V = logps.shape
    gathered = logps.gather(dim=-1, index=tokens.unsqueeze(-1)).squeeze(-1)  # (B, T)
    # Zero out padded / non-response positions
    gathered = gathered * mask.float()
    # Sum and divide by number of valid positions
    sum_logps = gathered.sum(dim=1)
    n_valid = mask.sum(dim=1).clamp(min=1)
    return sum_logps / n_valid

# ─── DPO Loss ─────────────────────────────────────────────────────
def dpo_loss(policy_chosen_logps, policy_rejected_logps,
             ref_chosen_logps, ref_rejected_logps, beta=0.1):
    """
    Standard DPO loss: -logsigmoid(beta * ((policy_chosen - ref_chosen) -
                                            (policy_rejected - ref_rejected)))
    All logps are per-sample averages over response tokens.
    """
    policy_ratio = policy_chosen_logps - policy_rejected_logps
    ref_ratio = ref_chosen_logps - ref_rejected_logps
    loss = -F.logsigmoid(beta * (policy_ratio - ref_ratio)).mean()
    return loss

# ─── Training ─────────────────────────────────────────────────────
def train_dpo(cfg: DPOConfig):
    print(f"Device: {cfg.device}")
    print(f"SFT checkpoint: {cfg.sft_checkpoint}")
    print(f"Beta: {cfg.beta} | LR: {cfg.learning_rate}")

    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M", trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    pad_id = tokenizer.pad_token_id or 0

    # Policy model (trainable)
    policy = SmolLM2(cfg).to(cfg.device)
    # Reference model (frozen)
    reference = SmolLM2(cfg).to(cfg.device)

    # Load SFT weights into both, or resume DPO
    latest_dpo = cfg.output_dir / "dpo_latest.pt"
    resume_step = 0
    best_loss = float("inf")
    if latest_dpo.exists():
        state = torch.load(latest_dpo, map_location=cfg.device, weights_only=False)
        sd = state["model_state_dict"]
        if any(k.startswith("_orig_mod.") for k in sd.keys()):
            sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
        policy.load_state_dict(sd)
        reference.load_state_dict(sd)
        optimizer.load_state_dict(state["optimizer_state_dict"])
        resume_step = state.get("step", 0)
        best_loss = state.get("best_loss", float("inf"))
        if "rng_state" in state:
            torch.set_rng_state(state["rng_state"].cpu())
        if "cuda_rng_state" in state and cfg.device == "cuda":
            torch.cuda.set_rng_state_all([s.cpu() if s.is_cuda else s for s in state["cuda_rng_state"]])
        print(f"  🔄 Resumed DPO from step {resume_step} (best_loss={best_loss:.4f})")
    elif cfg.sft_checkpoint.exists():
        state = torch.load(cfg.sft_checkpoint, map_location=cfg.device, weights_only=False)
        sd = state["model_state_dict"]
        # Strip torch.compile _orig_mod prefix if present
        if any(k.startswith("_orig_mod.") for k in sd.keys()):
            sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
        policy.load_state_dict(sd)
        reference.load_state_dict(sd)
        print(f"  🔄 Loaded SFT checkpoint step {state.get('step', 'unknown')}")
    else:
        print("  ⚠ No checkpoint found. Starting from scratch.")

    # Freeze reference
    for p in reference.parameters():
        p.requires_grad = False
    reference.eval()

    print(f"Policy params: {sum(p.numel() for p in policy.parameters()):,}")
    print(f"Reference params: {sum(p.numel() for p in reference.parameters()):,}")

    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=cfg.learning_rate,
        betas=(cfg.beta1, cfg.beta2),
        eps=cfg.eps,
        weight_decay=cfg.weight_decay,
    )

    # Data
    pairs = load_dpo_pairs(cfg.dpo_data_dir)
    if not pairs:
        print("No DPO data. Exiting.")
        return

    dataset = DPODataset(pairs, tokenizer, cfg.seq_len)
    dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=0)
    data_iter = iter(dataloader)
    print(f"Batches: ~{len(dataset) // cfg.batch_size}")

    use_amp = cfg.device == "cuda" and torch.cuda.is_bf16_supported()
    # bf16 does not need loss scaling; GradScaler can cause NaN
    scaler = None if use_amp else None

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
    step = resume_step
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

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

            chosen = batch["chosen"].to(cfg.device)
            rejected = batch["rejected"].to(cfg.device)
            prompt_len = batch["prompt_len"]

            # Build masks: only response tokens (after prompt)
            B, T = chosen.shape
            chosen_mask = torch.zeros(B, T, dtype=torch.bool, device=cfg.device)
            rejected_mask = torch.zeros(B, T, dtype=torch.bool, device=cfg.device)
            for b in range(B):
                pl = min(prompt_len[b].item(), T)
                chosen_mask[b, pl:] = True
                rejected_mask[b, pl:] = True

            ctx = torch.autocast(device_type=cfg.device, dtype=torch.bfloat16 if use_amp else torch.float32)
            with ctx:
                # Policy forward on both branches
                policy_chosen_logits, _ = policy(chosen)
                policy_rejected_logits, _ = policy(rejected)

                # Reference forward (no grad)
                with torch.no_grad():
                    ref_chosen_logits, _ = reference(chosen)
                    ref_rejected_logits, _ = reference(rejected)

                # Compute per-sample log-probs over response tokens
                policy_chosen_logps = get_batch_logps(policy_chosen_logits, chosen, chosen_mask)
                policy_rejected_logps = get_batch_logps(policy_rejected_logits, rejected, rejected_mask)
                ref_chosen_logps = get_batch_logps(ref_chosen_logits, chosen, chosen_mask)
                ref_rejected_logps = get_batch_logps(ref_rejected_logits, rejected, rejected_mask)

                loss = dpo_loss(
                    policy_chosen_logps, policy_rejected_logps,
                    ref_chosen_logps, ref_rejected_logps, cfg.beta,
                ) / cfg.gradient_accumulation_steps

            accumulated_loss += loss.item()

            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

        if scaler:
            scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.max_grad_norm)
        if scaler:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        running_loss += accumulated_loss

        if step % cfg.log_every_n_steps == 0:
            dt = time.time() - t0
            t0 = time.time()
            avg_loss = running_loss / cfg.log_every_n_steps
            print(f"step {step:5d} | loss {avg_loss:.4f} | lr {lr:.2e} | {dt:.1f}s")
            running_loss = 0.0

        if step % cfg.save_every_n_steps == 0:
            val_loss = avg_loss
            is_best = val_loss < best_loss
            if is_best:
                best_loss = val_loss
            state = {
                "step": step,
                "loss": val_loss,
                "best_loss": best_loss,
                "model_state_dict": policy.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": cfg.__dict__,
                "rng_state": torch.get_rng_state(),
                "cuda_rng_state": torch.cuda.get_rng_state_all() if cfg.device == "cuda" else None,
            }
            tmp = cfg.output_dir / "dpo_latest.tmp"
            torch.save(state, tmp)
            os.replace(tmp, cfg.output_dir / "dpo_latest.pt")
            if is_best:
                os.replace(cfg.output_dir / "dpo_latest.pt", cfg.output_dir / "dpo_best.pt")
                print(f"  ⭐ Saved best DPO (loss {val_loss:.4f})")
            else:
                print(f"  💾 Saved latest DPO (loss {val_loss:.4f})")

    state = {
        "step": step,
        "loss": running_loss,
        "best_loss": best_loss,
        "model_state_dict": policy.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": cfg.__dict__,
        "rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all() if cfg.device == "cuda" else None,
    }
    torch.save(state, cfg.output_dir / "dpo_final.pt")
    print(f"\n✅ DPO complete. dpo_final.pt | best={best_loss:.4f}")

if __name__ == "__main__":
    cfg = DPOConfig()
    train_dpo(cfg)
