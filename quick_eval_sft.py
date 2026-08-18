"""Quick eval for SFT checkpoint — RAW format matching training data.

Training format: raw problem text (no chat markers) → answer continuation.
"""
import torch, torch.nn.functional as F
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM

CKPT = "/home/kenpeter/work/checkpoints/sft_best.pt"
DEV = "cuda" if torch.cuda.is_available() else "cpu"

state = torch.load(CKPT, map_location="cpu", weights_only=False)
sd = state["model_state_dict"]
if any(k.startswith("_orig_mod.") for k in sd):
    sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
print(f"ckpt: step={state.get('step','?')} loss={state.get('loss','?'):.4f} on {DEV}", flush=True)

cfg = LlamaConfig(
    vocab_size=49152, hidden_size=1536, intermediate_size=4608,
    num_hidden_layers=32, num_attention_heads=12, num_key_value_heads=4,
    max_position_embeddings=8192, rope_theta=10000.0, rms_norm_eps=1e-5,
    hidden_act="silu", tie_word_embeddings=False, attention_bias=False,
    mlp_bias=False, head_dim=128,
)
model = LlamaForCausalLM(cfg).to(DEV).eval()
missing, unexpected = model.load_state_dict(sd, strict=False)
assert not missing, f"MISSING KEYS: {missing[:10]}"
model = model.to(torch.bfloat16)

tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M", trust_remote_code=True)

@torch.no_grad()
def generate(prompt, max_new=200, temperature=0.0, top_p=0.9):
    ids = tok(prompt, return_tensors="pt")["input_ids"].to(DEV)
    for _ in range(max_new):
        logits = model(ids[:, -2048:]).logits[:, -1, :].float()
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

PROMPTS = [
    ("math", "What is 7 * 6 + 4?"),
    ("math", "Factor x^2 - 5x + 6."),
    ("code", "Write a Python function that reverses a string."),
    ("prose", "Explain the first law of thermodynamics."),
    ("prose", "What is the capital of France?"),
]

for name, prompt in PROMPTS:
    print(f"\n=== {name} ===")
    print(generate(prompt))