import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer
from pathlib import Path
import math

# ----- Architecture from sft.py -----
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

# ----- Config -----
class EvalConfig:
    vocab_size = 49152
    dim = 576
    n_layers = 30
    n_heads = 9
    n_kv_heads = 3
    intermediate_size = 1536
    max_seq_len = 2048
    rope_theta = 10000.0
    rms_norm_eps = 1e-5
    dropout = 0.0
    # Not needed for generation but keep for compatibility
    bos_token_id = 1
    eos_token_id = 2

def load_model(ckpt_path, device='cpu'):
    cfg = EvalConfig()
    model = SmolLM2(cfg)
    print(f"Loading checkpoint from {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    # Handle possible _orig_mod prefix
    state_dict = checkpoint["model_state_dict"]
    if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print(f"Loaded step {checkpoint.get('step', 'unknown')} | loss {checkpoint.get('loss', 'unknown')}")
    return model

@torch.no_grad()
def generate(model, tokenizer, prompt, max_new=128, temperature=0.7, top_p=0.9, device='cpu'):
    # Format prompt as in training
    text = f"{tokenizer.bos_token}user\n{prompt}assistant\n"
    ids = tokenizer(text, add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)
    for _ in range(max_new):
        # Crop to max_seq_len
        input_ids = ids[:, -model.cfg.max_seq_len:]
        logits, _ = model(input_ids)
        logits = logits[:, -1, :]  # (B, vocab)
        if temperature > 0:
            probs = F.softmax(logits / temperature, dim=-1)
            # Top-p sampling
            sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            # Remove tokens with cumulative probability > top_p
            sorted_indices_to_remove = cumulative_probs > top_p
            # Shift right to keep first token above threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            # Scatter back to original ordering
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_idx, sorted_indices_to_remove)
            probs[indices_to_remove] = 0
            # Renormalize
            probs = probs / probs.sum(dim=-1, keepdim=True)
            # Sample
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
        ids = torch.cat([ids, next_token], dim=1)
        if next_token.item() == tokenizer.eos_token_id:
            break
    # Decode
    output = tokenizer.decode(ids[0], skip_special_tokens=False)
    # Extract assistant part
    if "assistant\n" in output:
        output = output.split("assistant\n")[1]
    # Remove any stray � tokens
    output = output.replace("�", "").strip()
    return output

def main():
    device = "cpu"  # GPU is occupied by training
    # Choose checkpoint: best or latest
    ckpt_path = "/home/kenpeter/work/checkpoints/sft_best.pt"
    # ckpt_path = "/home/kenpeter/work/checkpoints/sft_latest.pt"
    model = load_model(ckpt_path, device)
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M", trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

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
        resp = generate(model, tokenizer, p, max_new=128, temperature=0.7, top_p=0.9, device=device)
        print(f"🤖 RESPONSE: {resp}")
        print("-"*40)

if __name__ == "__main__":
    main()