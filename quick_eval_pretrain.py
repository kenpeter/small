"""Quick eval of pretrain checkpoint — load megatrain_latest.pt into HF LlamaForCausalLM, generate + PPL."""
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM
from pathlib import Path

CKPT = "/home/kenpeter/work/checkpoints/megatrain_latest.pt"
DEV = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading {CKPT}...")
state = torch.load(CKPT, map_location="cpu", weights_only=False)
sd = state["model_state_dict"]
if any(k.startswith("_orig_mod.") for k in sd):
    sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
# NOTE: checkpoint keys are HF-style (model.embed_tokens.weight, model.layers.N.*,
# model.norm.weight, lm_head.weight) — LlamaForCausalLM expects exactly these.
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
if unexpected:
    print(f"  WARNING unexpected keys: {unexpected[:5]}... ({len(unexpected)})")
model = model.to(torch.bfloat16)
print(f"  Model on {DEV}: {sum(p.numel() for p in model.parameters())/1e9:.2f}B params")

tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M", trust_remote_code=True)

@torch.no_grad()
def generate(prompt, max_new=96, temperature=0.7, top_p=0.9):
    ids = tok(prompt, return_tensors="pt")["input_ids"].to(DEV)
    for _ in range(max_new):
        logits = model(ids[:, -2048:]).logits[:, -1, :].float()  # fp32 for stable sampling
        if temperature > 0:
            probs = F.softmax(logits / temperature, dim=-1)
            sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
            cumsum = torch.cumsum(sorted_probs, dim=-1)
            sorted_probs[cumsum > top_p] = 0
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

prompts = [
    "The capital of France is",
    "def reverse_string(s):\n    ",
    "2 + 2 = 4. 3 + 5 = 8. 7 * 6 =",
    "Once upon a time, in a land far away,",
    "The first law of thermodynamics states that",
]

print("\n" + "=" * 60)
for p in prompts:
    print(f"\n💬 PROMPT: {p!r}")
    print(f"🤖 {generate(p)}")
    print("-" * 40)

print("\n📊 Perplexity on sample texts:")
for t in [
    "The quick brown fox jumps over the lazy dog.",
    "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
    "2 + 2 = 4\n3 + 5 = 8\n7 * 6 = 42\n12 / 4 = 3",
]:
    p, loss = ppl(t)
    print(f"  PPL={p:8.2f}  loss={loss:.4f}  | {t[:60]!r}")
