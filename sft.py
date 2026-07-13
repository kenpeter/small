"""
SFT (Supervised Fine-Tuning) for SmolLM2-135M.
Loads pretrained checkpoint, fine-tunes on instruction-response pairs.
Only assistant tokens contribute to loss (system/user masked with -100).
"""
import os, sys, json, time, math, gc
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset, concatenate_datasets

# ─── Reuse architecture from train.py ─────────────────────────────
# (paste the same classes to avoid import coupling)

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

# ─── Config ──────────────────────────────────────────────────────
@dataclass
class SFTConfig:
    # Model (same as pretraining)
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

    # Training
    base_checkpoint: Path = Path("/home/kenpeter/work/checkpoints/checkpoint_best.pt")
    output_dir: Path = Path("/home/kenpeter/work/checkpoints")
    seq_len: int = 2048
    batch_size: int = 2
    gradient_accumulation_steps: int = 4
    max_steps: int = 20_000
    learning_rate: float = 2e-5
    min_lr: float = 2e-6
    warmup_steps: int = 100
    weight_decay: float = 0.1
    max_grad_norm: float = 1.0
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    save_every_n_steps: int = 1000
    log_every_n_steps: int = 10
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

# ─── Chat formatting ─────────────────────────────────────────────
CHAT_TEMPLATE = """<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
{instruction}<|im_end|>
<|im_start|>assistant
{response}<|im_end|>"""

def format_openhermes(example: dict) -> dict:
    """OpenHermes 2.5 format: conversations list."""
    conversations = example.get("conversations", [])
    text = ""
    for turn in conversations:
        role = turn.get("from", turn.get("role", ""))
        content = turn.get("value", turn.get("content", ""))
        if role in ("human", "user"):
            text += f"<|im_start|>user\n{content}<|im_end|>\n"
        elif role in ("gpt", "assistant"):
            text += f"<|im_start|>assistant\n{content}<|im_end|>\n"
        elif role == "system":
            text += f"<|im_start|>system\n{content}<|im_end|>\n"
    return {"text": text}

def format_openorca(example: dict) -> dict:
    """OpenOrca format: system_prompt, question, response."""
    system = example.get("system_prompt", "You are a helpful assistant.")
    question = example.get("question", example.get("instruction", ""))
    response = example.get("response", example.get("answer", ""))
    text = f"<|im_start|>system\n{system}<|im_end|>\n"
    text += f"<|im_start|>user\n{question}<|im_end|>\n"
    text += f"<|im_start|>assistant\n{response}<|im_end|>\n"
    return {"text": text}

def format_alpaca(example: dict) -> dict:
    """Alpaca format: instruction, input, output."""
    instruction = example.get("instruction", "")
    inp = example.get("input", "")
    output = example.get("output", "")
    if inp:
        instruction += f"\n{inp}"
    text = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
    text += f"<|im_start|>user\n{instruction}<|im_end|>\n"
    text += f"<|im_start|>assistant\n{output}<|im_end|>\n"
    return {"text": text}

def format_ultrachat(example: dict) -> dict:
    """Ultrachat format: data list of turns."""
    data = example.get("data", [])
    text = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
    for i, turn in enumerate(data):
        role = "user" if i % 2 == 0 else "assistant"
        text += f"<|im_start|>{role}\n{turn}<|im_end|>\n"
    return {"text": text}

# ─── Tokenize with loss masking ──────────────────────────────────
class SFTDataset(Dataset):
    def __init__(self, texts: List[str], tokenizer, seq_len: int = 2048):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.samples = []
        for text in texts:
            sample = self._tokenize_with_mask(text)
            if sample:
                self.samples.append(sample)

    def _tokenize_with_mask(self, text: str):
        """Tokenize and mask non-assistant tokens with -100."""
        # Simple heuristic: split by role markers and mask accordingly
        parts = text.split("<|im_start|>")
        all_ids = []
        all_labels = []
        for part in parts:
            if not part.strip():
                continue
            # Determine role
            if part.startswith("assistant"):
                # Train on this part
                content = part[len("assistant"):].split("<|im_end|>")[0].strip()
                if not content:
                    continue
                ids = self.tokenizer.encode(content, add_special_tokens=False)
                all_ids.extend(ids)
                all_labels.extend(ids)
            elif part.startswith("user") or part.startswith("system"):
                # Mask this part
                content = part.split("\n", 1)[1].split("<|im_end|>")[0] if "\n" in part else ""
                if not content:
                    continue
                ids = self.tokenizer.encode(content, add_special_tokens=False)
                all_ids.extend(ids)
                all_labels.extend([-100] * len(ids))
            else:
                # Raw text (shouldn't happen with proper formatting)
                ids = self.tokenizer.encode(part, add_special_tokens=False)
                all_ids.extend(ids)
                all_labels.extend([-100] * len(ids))

        if len(all_ids) < 2:
            return None

        # Truncate or pad to seq_len
        if len(all_ids) > self.seq_len:
            all_ids = all_ids[:self.seq_len]
            all_labels = all_labels[:self.seq_len]
        else:
            # Pad
            pad_len = self.seq_len - len(all_ids)
            all_ids.extend([self.tokenizer.pad_token_id] * pad_len)
            all_labels.extend([-100] * pad_len)

        return {
            "input_ids": torch.tensor(all_ids, dtype=torch.long),
            "labels": torch.tensor(all_labels, dtype=torch.long),
        }

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]["input_ids"], self.samples[idx]["labels"]

# ─── Load datasets ───────────────────────────────────────────────
def load_sft_datasets(max_samples_per_ds: int = 50_000):
    """Load and combine SFT datasets."""
    all_texts = []

    # OpenHermes 2.5
    try:
        print("Loading OpenHermes 2.5...")
        ds = load_dataset("teknium/OpenHermes-2.5", split="train", streaming=True)
        for i, ex in enumerate(ds):
            if i >= max_samples_per_ds:
                break
            all_texts.append(format_openhermes(ex)["text"])
        print(f"  ✓ OpenHermes: {len(all_texts)} samples so far")
    except Exception as e:
        print(f"  ⚠ OpenHermes failed: {e}")

    # OpenOrca
    try:
        print("Loading OpenOrca...")
        ds = load_dataset("Open-Orca/OpenOrca", split="train", streaming=True)
        count = 0
        for ex in ds:
            if count >= max_samples_per_ds:
                break
            all_texts.append(format_openorca(ex)["text"])
            count += 1
        print(f"  ✓ OpenOrca: +{count} samples")
    except Exception as e:
        print(f"  ⚠ OpenOrca failed: {e}")

    # Alpaca-GPT4
    try:
        print("Loading Alpaca-GPT4...")
        ds = load_dataset("vicgalle/alpaca-gpt4", split="train")
        count = min(len(ds), max_samples_per_ds)
        for ex in ds.select(range(count)):
            all_texts.append(format_alpaca(ex)["text"])
        print(f"  ✓ Alpaca-GPT4: +{count} samples")
    except Exception as e:
        print(f"  ⚠ Alpaca-GPT4 failed: {e}")

    # Code-Alpaca
    try:
        print("Loading CodeAlpaca...")
        ds = load_dataset("sahil2801/CodeAlpaca-20k", split="train")
        count = min(len(ds), 20_000)
        for ex in ds.select(range(count)):
            all_texts.append(format_alpaca(ex)["text"])
        print(f"  ✓ CodeAlpaca: +{count} samples")
    except Exception as e:
        print(f"  ⚠ CodeAlpaca failed: {e}")

    # Ultrachat
    try:
        print("Loading Ultrachat...")
        ds = load_dataset("stingning/ultrachat", split="train", streaming=True)
        count = 0
        for ex in ds:
            if count >= max_samples_per_ds:
                break
            all_texts.append(format_ultrachat(ex)["text"])
            count += 1
        print(f"  ✓ Ultrachat: +{count} samples")
    except Exception as e:
        print(f"  ⚠ Ultrachat failed: {e}")

    print(f"\n📊 Total SFT samples: {len(all_texts)}")
    return all_texts

# ─── LR Scheduler ─────────────────────────────────────────────────
def get_lr(step: int, cfg: SFTConfig):
    if step < cfg.warmup_steps:
        return cfg.learning_rate * (step + 1) / cfg.warmup_steps
    if step >= cfg.max_steps:
        return cfg.min_lr
    decay_ratio = (step - cfg.warmup_steps) / (cfg.max_steps - cfg.warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return cfg.min_lr + coeff * (cfg.learning_rate - cfg.min_lr)

# ─── Training ──────────────────────────────────────────────────────
def train_sft(cfg: SFTConfig):
    print(f"Device: {cfg.device}")
    print(f"Base checkpoint: {cfg.base_checkpoint}")

    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M", trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"Tokenizer vocab: {len(tokenizer)}")

    # Model
    model = SmolLM2(cfg).to(cfg.device)
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    # Load pretrained weights
    if cfg.base_checkpoint.exists():
        state = torch.load(cfg.base_checkpoint, map_location=cfg.device, weights_only=False)
        model.load_state_dict(state["model_state_dict"])
        print(f"  🔄 Loaded pretrained from step {state.get('step', 'unknown')}")
    else:
        print(f"  ⚠ No base checkpoint found at {cfg.base_checkpoint}")
        print("  Training from scratch (not recommended for SFT)")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        betas=(cfg.beta1, cfg.beta2),
        eps=cfg.eps,
        weight_decay=cfg.weight_decay,
    )

    # Data
    texts = load_sft_datasets()
    if not texts:
        print("No SFT data loaded. Exiting.")
        return

    dataset = SFTDataset(texts, tokenizer, cfg.seq_len)
    dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=0)
    data_iter = iter(dataloader)
    print(f"Batches per epoch: ~{len(dataset) // cfg.batch_size}")

    use_amp = cfg.device == "cuda" and torch.cuda.is_bf16_supported()
    scaler = torch.cuda.amp.GradScaler() if use_amp and hasattr(torch.cuda.amp, "GradScaler") else None

    # Training loop
    t0 = time.time()
    running_loss = 0.0
    step = 0

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")

    while step < cfg.max_steps:
        step += 1
        lr = get_lr(step, cfg)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        accumulated_loss = 0.0
        for micro_step in range(cfg.gradient_accumulation_steps):
            try:
                x, y = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                x, y = next(data_iter)

            x, y = x.to(cfg.device), y.to(cfg.device)
            ctx = torch.autocast(device_type=cfg.device, dtype=torch.bfloat16 if use_amp else torch.float32)
            with ctx:
                _, loss = model(x, y)
                loss = loss / cfg.gradient_accumulation_steps
            accumulated_loss += loss.item()

            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

        if scaler:
            scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
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
            tokens_per_sec = (cfg.batch_size * cfg.seq_len * cfg.gradient_accumulation_steps * cfg.log_every_n_steps) / dt
            avg_loss = running_loss / cfg.log_every_n_steps
            print(f"step {step:5d} | loss {avg_loss:.4f} | lr {lr:.2e} | {dt:.1f}s | {tokens_per_sec:,.0f} tok/s")
            running_loss = 0.0

        if step % cfg.save_every_n_steps == 0:
            val_loss = avg_loss  # Use running avg as proxy
            is_best = val_loss < best_loss
            if is_best:
                best_loss = val_loss
            state = {
                "step": step,
                "loss": val_loss,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": cfg.__dict__,
            }
            tmp = cfg.output_dir / "sft_latest.tmp"
            torch.save(state, tmp)
            os.replace(tmp, cfg.output_dir / "sft_latest.pt")
            if is_best:
                os.replace(cfg.output_dir / "sft_latest.pt", cfg.output_dir / "sft_best.pt")
                print(f"  ⭐ Saved best SFT (loss {val_loss:.4f})")
            else:
                print(f"  💾 Saved latest SFT (loss {val_loss:.4f})")

    # Final save
    state = {
        "step": step,
        "loss": running_loss,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": cfg.__dict__,
    }
    torch.save(state, cfg.output_dir / "sft_final.pt")
    print(f"\n✅ SFT complete. Saved sft_final.pt | best_loss={best_loss:.4f}")

if __name__ == "__main__":
    cfg = SFTConfig()
    train_sft(cfg)
