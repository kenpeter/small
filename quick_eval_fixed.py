"""Quick eval of SFT model — load sft_best.pt and generate a few responses."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer
from pathlib import Path

# ─── Same architecture as dpo.py ──────────────────────────────────
class SimpleConfig:
    def __init__(self):
        self.vocab_size = 49152
        self.dim = 576
        self.n_layers = 30
        self.n_heads = 9
        self.n_kv_heads = 3
        self.intermediate_size = 1536
        self.max_seq_len = 8192
        self.rope_theta = 10000.0
        self.rms_norm_eps = 1e-5
        self.dropout = 0.0
        self.batch_size = 1
        self.gradient_accumulation_steps = 4
        self.max_steps = 40000
        self.learning_rate = 2e-5
        self.min_lr = 1e-6
        self.warmup_steps = 100
        self.weight_decay = 0.1
        self.max_grad_norm = 1.0
        self.beta1 = 0.9
        self.beta2 = 0.95
        self.eps = 1e-8
        self.save_every_n_steps = 50
        self.log_every_n_steps = 10
        self.val_every_n_steps = 0
        self.device = "cpu"
        self.use_gradient_checkpointing = False
        self.compile = False
        self.seq_len = 1024
        self.val_frac = 0.05
        self.checkpoint_dir = "../checkpoints"
        self.data_dir = "../data/sft_shards"

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
        self.head_dim = cfg.dim // cfg.n_heads
        self.q_proj = nn.Linear(cfg.dim, cfg.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.dim, cfg.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.dim, cfg.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(cfg.n_heads * self.head_dim, cfg.dim, bias=False)
        self.rotary = RotaryEmbedding(self.head_dim, max_seq_len=cfg.max_seq_len)

    def forward(self, x, mask=None):
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim)
        q = self.rotary(q, T)
        k = self.rotary(k, T)
        if self.n_kv_heads != self.n_heads:
            k = repeat_kv(k, self.n_heads // self.n_kv_heads)
            v = repeat_kv(v, self.n_heads // self.n_kv_heads)
        att = (q @ k.transpose(-2, -1)) * (1.0 / (self.head_dim ** 0.5))
        if mask is not None:
            att = att.masked_fill(mask == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.o_proj(y)
        return y

class MLP(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim, bias=False)
        self.fc2 = nn.Linear(hidden_dim, dim, bias=False)
    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))

class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.attn = CausalSelfAttention(cfg)
        self.ffn = MLP(cfg.dim, cfg.intermediate_size)
        self.attn_norm = RMSNorm(cfg.dim, eps=cfg.rms_norm_eps)
        self.ffn_norm = RMSNorm(cfg.dim, eps=cfg.rms_norm_eps)

    def forward(self, x):
        x = x + self.attn(self.attn_norm(x))
        x = x + self.ffn(self.ffn_norm(x))
        return x

class SmolLM2(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.layers = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.dim, eps=cfg.rms_norm_eps)
        self.output = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)

    def forward(self, idx):
        x = self.tok_emb(idx)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        logits = self.output(x)
        return logits

# ─── Load model ───────────────────────────────────────────────────
ckpt_path = "/home/kenpeter/work/checkpoints/sft_best.pt"
print(f"Loading {ckpt_path}...")
state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
sd = state["model_state_dict"]
if any(k.startswith("_orig_mod.") for k in sd.keys()):
    sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}

cfg = SimpleConfig()
model = SmolLM2(cfg).cpu().eval()
model.load_state_dict(sd)
print(f"Loaded step {state.get('step', 'unknown')} | loss {state.get('loss', 'unknown')}")

print("WARNING: Running on CPU because GPU is occupied by DPO. Generation will be slow.")

tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M", trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ─── Generate ─────────────────────────────────────────────────────
@torch.no_grad()
def generate(prompt, max_new=128, temperature=0.7, top_p=0.9):
    text = f"{tokenizer.bos_token}user\n{prompt}assistant\n"
    ids = tokenizer(text, add_special_tokens=False, return_tensors="pt")["input_ids"].cpu()
    for _ in range(max_new):
        logits = model(ids[:, -cfg.max_seq_len:])[:, -1, :]
        if temperature > 0:
            probs = F.softmax(logits / temperature, dim=-1)
            # top-p sampling
            sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
            cumsum = torch.cumsum(sorted_probs, dim=-1)
            sorted_probs[cumsum > top_p] = 0
            sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
            # Clamp to avoid NaN from numerical issues
            sorted_probs = sorted_probs.clamp(min=0)
            if sorted_probs.sum() <= 0 or torch.isnan(sorted_probs).any():
                next_tok = logits.argmax(dim=-1, keepdim=True)
            else:
                next_tok = sorted_idx[torch.multinomial(sorted_probs, num_samples=1).item()].unsqueeze(0).unsqueeze(0)
        else:
            next_tok = logits.argmax(dim=-1, keepdim=True)
        ids = torch.cat([ids, next_tok], dim=1)
        if next_tok.item() == tokenizer.eos_token_id:
            break
    out = tokenizer.decode(ids[0], skip_special_tokens=False)
    # Strip the prompt part
    if "assistant\n" in out:
        out = out.split("assistant\n")[1]
    out = out.replace("�", "").strip()
    return out

prompts = [
    "What is the capital of France?",
    "Explain quantum computing in one sentence.",
    "Write a Python function to reverse a string.",
    "What is 7 times 8?",
    "Tell me a joke.",
]

print("\n" + "="*60)
for p in prompts:
    print(f"\n💬 PROMPT: {p}")
    print(f"🤖 RESPONSE: {generate(p)}")
    print("-"*40)