"""Quick eval of SFT model — load sft_best.pt and generate a few responses."""
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from pathlib import Path

from model import ModelConfig, SmolLM2

max_seq_len = 8192

# ─── Load model ───────────────────────────────────────────────────
ckpt_path = "/home/kenpeter/work/checkpoints/sft_best.pt"
print(f"Loading {ckpt_path}...")
state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
sd = state["model_state_dict"]
if any(k.startswith("_orig_mod.") for k in sd.keys()):
    sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}

cfg = ModelConfig(max_seq_len=max_seq_len)
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
        logits, _ = model(ids[:, -max_seq_len:])
        logits = logits[:, -1, :]
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
