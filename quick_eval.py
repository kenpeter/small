"""Quick eval of SFT model — load sft_best.pt and generate a few responses."""
import torch, torch.nn as nn, torch.nn.functional as F
from transformers import AutoTokenizer
from pathlib import Path

# ─── Same architecture as dpo.py ──────────────────────────────────
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
    def forward(self, idx):
        x = self.tok_embeddings(idx)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return self.lm_head(x)

class SimpleConfig:
    vocab_size = 49152
    dim = 576
    n_layers = 30
    n_heads = 9
    n_kv_heads = 3
    intermediate_size = 1536
    max_seq_len = 8192
    rope_theta = 10000.0
    rms_norm_eps = 1e-5

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
    text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
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
            sorted_probs = sorted_probs.clamp(min=0)
            if sorted_probs.sum() <= 0 or torch.isnan(sorted_probs).any():
                next_tok = logits.argmax(dim=-1, keepdim=True)
            else:
                idx = torch.multinomial(sorted_probs[0], num_samples=1).item()
                next_tok = sorted_idx[:, idx:idx+1]
        else:
            next_tok = logits.argmax(dim=-1, keepdim=True)
        ids = torch.cat([ids, next_tok], dim=1)
        if next_tok.item() == tokenizer.eos_token_id:
            break
    out = tokenizer.decode(ids[0], skip_special_tokens=False)
    # Strip the prompt part
    if "<|im_start|>assistant\n" in out:
        out = out.split("<|im_start|>assistant\n")[1]
    out = out.replace("<|im_end|>", "").strip()
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
