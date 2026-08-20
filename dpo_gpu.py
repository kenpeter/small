#!/usr/bin/env python3
"""DPO (Direct Preference Optimization) post-training for small-1B (after SFT).

Base = SFT model (hf_export_sft1b). Preference pairs from build_dpo_pairs.py:
  - prompt   = raw problem statement (matches SFT raw-format training)
  - chosen   = gold correct solution
  - rejected = SFT model's wrong/echo generation
Reference model = policy at init (ref_model=None -> standard DPO from SFT anchor).

Memory-lean on the 12GB card: bf16, gradient checkpointing, 8-bit Adam (bnb,
requires LD_LIBRARY_PATH set at launch), small batch + grad accum.

Saves stage-isolated checkpoint dpo_best.pt in raw dict-format (model_state_dict)
so downstream stages (GRPO/eval) can load it, per the sft->dpo stage convention.

Usage:
  export LD_LIBRARY_PATH=.../nvidia/cu13/lib:.../nvidia/nvjitlink/lib
  ./venv/bin/python dpo_gpu.py --steps 300 --lr 1e-5
"""
import argparse, json, os, re, shutil, subprocess, sys, time
from pathlib import Path
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer

BASE = Path("/home/kenpeter/work/hf_export_sft1b")
PAIRS = Path("/home/kenpeter/work/data/_dpo/dpo_pairs.jsonl")
OUT = Path("/home/kenpeter/work/checkpoints/dpo")
RAW_OUT = Path("/home/kenpeter/work/checkpoints/dpo_best.pt")  # stage-isolated raw ckpt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--loss-type", type=str, default="sigmoid")
    ap.add_argument("--max-length", type=int, default=1024)
    ap.add_argument("--max-prompt-length", type=int, default=512)
    ap.add_argument("--save-every", type=int, default=100)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42)

    print(f"Loading SFT base from {BASE}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(BASE), torch_dtype=torch.bfloat16, attn_implementation="sdpa"
    )
    tokenizer = AutoTokenizer.from_pretrained(str(BASE))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # ensure no chat template is force-applied: our pairs are raw-format; TRL
    # treats chosen/rejected as plain text completions.
    model.config.pad_token_id = tokenizer.pad_token_id

    print(f"Loading DPO pairs from {PAIRS}", flush=True)
    ds = load_dataset("json", data_files=str(PAIRS), split="train")
    print(f"  {len(ds)} preference pairs", flush=True)

    cfg = DPOConfig(
        output_dir=str(OUT),
        learning_rate=args.lr,
        beta=args.beta,
        loss_type=args.loss_type,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        max_length=args.max_length,
        bf16=True,
        optim="adamw_bnb_8bit",
        gradient_checkpointing=True,
        max_steps=args.steps,
        num_train_epochs=1,
        save_steps=args.save_every,
        logging_steps=10,
        report_to="none",
        max_grad_norm=0.1,
        seed=42,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,  # reference = policy at init (standard DPO from SFT)
        args=cfg,
        processing_class=tokenizer,
        train_dataset=ds,
    )
    trainer.train()
    trainer.save_model(str(OUT / "final"))
    print("✅ DPO trainer finished ->", OUT, flush=True)

    # Extract final policy weights -> stage-isolated raw dpo_best.pt
    pol = trainer.model
    sd = {k: v.detach().contiguous().to(torch.bfloat16) for k, v in pol.state_dict().items()}
    torch.save(
        {"model_state_dict": sd, "step": cfg.max_steps, "loss": None,
         "config": vars(args) | {"optim": "adamw_bnb_8bit"}},
        str(RAW_OUT),
    )
    print(f"Saved stage checkpoint -> {RAW_OUT} ({os.path.getsize(RAW_OUT)/1e9:.2f} GB)", flush=True)


if __name__ == "__main__":
    main()
