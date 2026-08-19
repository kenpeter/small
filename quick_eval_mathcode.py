"""Math + Code focused quick eval of a pretrain checkpoint.

Runs CPU-safe (CUDA_VISIBLE_DEVICES="" to avoid pausing live training).
Focus: math reasoning + code generation, greedy AND sampled.
Usage: CUDA_VISIBLE_DEVICES="" venv/bin/python quick_eval_mathcode.py --ckpt PATH
"""
import argparse
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM

_ap = argparse.ArgumentParser()
_ap.add_argument("--ckpt", default="/home/kenpeter/work/checkpoints/pretrained_quickeval_71800.pt")
CKPT = _ap.parse_args().ckpt
DEV = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading {CKPT}...")
state = torch.load(CKPT, map_location="cpu", weights_only=False)
sd = state["model_state_dict"]
if any(k.startswith("_orig_mod.") for k in sd):
    sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
print(f"  step={state.get('step','?')} loss={state.get('loss','?')} best={state.get('best_loss','?')}")

cfg = LlamaConfig(
    vocab_size=49152, hidden_size=1536, intermediate_size=4608,
    num_hidden_layers=32, num_attention_heads=12, num_key_value_heads=4,
    max_position_embeddings=8192, rope_theta=10000.0, rms_norm_eps=1e-5,
    hidden_act="silu", tie_word_embeddings=False, attention_bias=False,
    mlp_bias=False, head_dim=128,
)
model = LlamaForCausalLM(cfg).to(DEV).eval()
missing, unexpected = model.load_state_dict(sd, strict=False)
if missing:
    print(f"  WARNING missing keys: {missing[:5]}... ({len(missing)})")
model = model.to(torch.bfloat16)
print(f"  Model on {DEV}: {sum(p.numel() for p in model.parameters())/1e9:.2f}B params")

tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M", trust_remote_code=True)

@torch.no_grad()
def generate(prompt, max_new=128, temperature=0.0):
    ids = tok(prompt, return_tensors="pt")["input_ids"].to(DEV)
    for _ in range(max_new):
        logits = model(ids[:, -2048:]).logits[:, -1, :].float()
        if temperature > 0:
            probs = F.softmax(logits / temperature, dim=-1)
            sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
            cumsum = torch.cumsum(sorted_probs, dim=-1)
            sorted_probs[cumsum > 0.9] = 0
            s = sorted_probs.sum(dim=-1, keepdim=True)
            if not torch.isfinite(s).all() or s.item() <= 0:
                nxt = logits.argmax(dim=-1, keepdim=True)
            else:
                sorted_probs = sorted_probs / s
                idx = torch.multinomial(sorted_probs[0], num_samples=1).item()
                nxt = sorted_idx[:, idx:idx+1]
        else:
            nxt = logits.argmax(dim=-1, keepdim=True)
        ids = torch.cat([ids, nxt], dim=1)
        if nxt.item() == tok.eos_token_id:
            break
    return tok.decode(ids[0], skip_special_tokens=True)

@torch.no_grad()
def ppl(text):
    ids = tok(text, return_tensors="pt")["input_ids"].to(DEV)[:, :1024]
    logits = model(ids).logits[0, :-1].float()
    nll = F.cross_entropy(logits, ids[0, 1:])
    return nll.exp().item(), nll.item()

# ── Math-focused ──
math_prompts = [
    "2 + 2 = 4. 3 + 5 = 8. 7 * 6 =",
    "Solve for x: 3x + 5 = 20. x =",
    "The area of a circle with radius 3 is pi * 3^2 =",
    "If there are 12 apples and each basket holds 4 apples, the number of baskets needed is",
]
# ── Code-focused ──
code_prompts = [
    "def reverse_string(s):\n    ",
    "def fibonacci(n):\n    ",
    "def is_prime(n):\n    ",
    "class Solution:\n    def twoSum(self, nums, target):\n        ",
]

print("\n" + "=" * 60)
print("MATH — greedy")
for p in math_prompts:
    print(f"\n💬 {p!r}\n🤖 {generate(p)}\n" + "-" * 40)
print("\nMATH — sampled (temp 0.7)")
for p in math_prompts[:3]:
    print(f"\n💬 {p!r}\n🤖 {generate(p, temperature=0.7)}\n" + "-" * 40)

print("\n" + "=" * 60)
print("CODE — greedy")
for p in code_prompts:
    print(f"\n💬 {p!r}\n🤖 {generate(p)}\n" + "-" * 40)
print("\nCODE — sampled (temp 0.7)")
for p in code_prompts[:3]:
    print(f"\n💬 {p!r}\n🤖 {generate(p, temperature=0.7)}\n" + "-" * 40)

print("\n" + "=" * 60)
print("📊 Perplexity (math + code)")
for t in [
    "2 + 2 = 4\n3 + 5 = 8\n7 * 6 = 42\n12 / 4 = 3",
    "x^2 - 5x + 6 = 0, so x = 2 or x = 3",
    "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
    "def is_prime(n):\n    if n < 2:\n        return False\n    for i in range(2, int(n**0.5) + 1):\n        if n % i == 0:\n            return False\n    return True",
    "left = 0\nright = len(arr) - 1\nwhile left <= right:\n    mid = (left + right) // 2",
]:
    p, loss = ppl(t)
    print(f"  PPL={p:8.2f}  loss={loss:.4f}  | {t[:55]!r}")
print("DONE")
