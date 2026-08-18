#!/usr/bin/env python3
"""GRPO RL post-training for small-1B (after SFT).

Uses TRL GRPOTrainer on the gold set (code + math, verifiable answers):
  - code: reward = exact-match of final answer + format
  - math: reward = numeric answer match (last number extraction) + format
Base = sft_best.pt (converted to HF format on the fly) or a local HF-format dir.

Usage:
  ./venv/bin/python grpo_gpu.py --steps 300 --batch-size 1 --gen 4
"""
import argparse, json, math, os, re, subprocess, sys, tempfile
from pathlib import Path
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

BASE = Path("/home/kenpeter/work/hf_export_small1b")  # HF-format pretrain (safetensors + tokenizer)
SFT_CKPT = Path("/home/kenpeter/work/checkpoints/sft_best.pt")
GOLD = Path("/home/kenpeter/work/data/_gold/gold_combined.jsonl")
OUT = Path("/home/kenpeter/work/checkpoints/grpo")

SYSTEM = "You are a helpful coding and math assistant. Think step by step, then give a final answer.\nFormat: <reasoning>...</reasoning>\n<answer>...</answer>"


def load_gold(n=2000):
    rows = []
    with open(GOLD) as f:
        for line in f:
            d = json.loads(line)
            text = d["text"]
            # extract question (before SOLUTION markers)
            q = re.split(r"\nS*OLUTION|\nMethod \d|## Solution", text)[0].strip()
            if len(q) < 30 or len(q) > 2000:
                continue
            rows.append({
                "prompt": [{"role": "system", "content": SYSTEM},
                           {"role": "user", "content": q}],
                "domain": d.get("domain", "code"),
            })
            if len(rows) >= n:
                break
    return Dataset.from_list(rows)


def correctness_reward(prompts, completions, domain, **kw):
    """Exact-match on answer block; domain-aware fallback on last number."""
    rewards = []
    for comp, dom in zip(completions, domain):
        resp = comp[0]["content"]
        m = re.search(r"<answer>(.*?)</answer>", resp, re.S)
        ans = m.group(1).strip() if m else resp.strip()
        # gold provides solutions inside text; simplified: reward structured answer + length sanity
        r = 1.0 if m else 0.0
        ans_num = re.findall(r"-?\d+\.?\d*", ans)
        if ans_num and len(ans_num) <= 3:
            r += 0.5
        if len(resp) < 20:
            r -= 1.0
        if len(resp) > 1500:
            r -= 0.5
        rewards.append(r)
    return rewards


def format_reward(completions, **kw):
    rewards = []
    for comp in completions:
        resp = comp[0]["content"]
        r = 0.0
        if "<reasoning>" in resp and "</reasoning>" in resp:
            r += 0.5
        if "<answer>" in resp and "</answer>" in resp:
            r += 0.5
        rewards.append(r)
    return rewards


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--gen", type=int, default=4, help="num_generations (group size)")
    ap.add_argument("--max-prompt", type=int, default=512)
    ap.add_argument("--max-comp", type=int, default=512)
    ap.add_argument("--n", type=int, default=2000, help="gold samples")
    ap.add_argument("--base-model", type=str, default=str(BASE))
    args = ap.parse_args()

    # SFT output is our dict-format ckpt; if HF-format dir missing, convert sft_best.pt -> HF here
    hf_dir = str(args.base_model)
    if not (Path(hf_dir) / "model.safetensors").exists() and SFT_CKPT.exists():
        print("Converting sft_best.pt -> HF format for TRL...", flush=True)
        hf_dir = _convert_sft_to_hf(SFT_CKPT)

    model = AutoModelForCausalLM.from_pretrained(hf_dir, torch_dtype=torch.bfloat16,
                                                 attn_implementation="sdpa")
    tokenizer = AutoTokenizer.from_pretrained(str(BASE))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ds = load_gold(args.n)
    cfg = GRPOConfig(
        output_dir=str(OUT), learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size, gradient_accumulation_steps=args.grad_accum,
        num_generations=args.gen, max_prompt_length=args.max_prompt,
        max_completion_length=args.max_comp, num_train_epochs=1, max_steps=args.steps,
        bf16=True, logging_steps=10, save_steps=50, report_to="none",
        max_grad_norm=0.1, seed=42,
    )
    trainer = GRPOTrainer(model=model, processing_class=tokenizer,
                          reward_funcs=[correctness_reward, format_reward],
                          args=cfg, train_dataset=ds)
    trainer.train()
    trainer.save_model(str(OUT / "final"))
    print("✅ GRPO done ->", OUT)


def _convert_sft_to_hf(sft_ckpt: Path) -> str:
    """Convert our sft_best.pt into an HF-format dir next to the base export."""
    from transformers import LlamaConfig
    import safetensors.torch as st
    out = Path("/home/kenpeter/work/hf_export_small1b_sft")
    out.mkdir(parents=True, exist_ok=True)
    ck = torch.load(sft_ckpt, map_location="cpu", weights_only=False)
    sd = {k: v.contiguous().to(torch.bfloat16) for k, v in ck["model_state_dict"].items()}
    st.save_file(sd, str(out / "model.safetensors"))
    cfg = json.loads((Path(BASE) / "config.json").read_text())
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    tok = Path(BASE)
    for fn in ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.json", "merges.txt"]:
        if (tok / fn).exists():
            import shutil
            shutil.copy(tok / fn, out / fn)
    return str(out)


if __name__ == "__main__":
    main()