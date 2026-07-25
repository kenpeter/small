import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from pathlib import Path

from model import ModelConfig, SmolLM2

def load_model(ckpt_path, device='cpu'):
    cfg = ModelConfig()
    model = SmolLM2(cfg)
    print(f"Loading checkpoint from {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
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
    text = f"{tokenizer.bos_token}user\n{prompt}assistant\n"
    ids = tokenizer(text, add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)
    for _ in range(max_new):
        input_ids = ids[:, -model.cfg.max_seq_len:]
        logits, _ = model(input_ids)
        logits = logits[:, -1, :]
        if temperature > 0:
            probs = F.softmax(logits / temperature, dim=-1)
            # top-p sampling
            sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            # shift right to keep first above threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_idx, sorted_indices_to_remove)
            probs[indices_to_remove] = 0
            probs = probs / probs.sum(dim=-1, keepdim=True)
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
        ids = torch.cat([ids, next_token], dim=1)
        if next_token.item() == tokenizer.eos_token_id:
            break
    output = tokenizer.decode(ids[0], skip_special_tokens=False)
    if "assistant\n" in output:
        output = output.split("assistant\n")[1]
    output = output.replace("�", "").strip()
    return output

def main():
    device = "cpu"
    # Choose checkpoint
    ckpt_path = "/home/kenpeter/work/checkpoints/dpo_best.pt"  # or dpo_latest.pt
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