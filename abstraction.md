# SmolLM2-135M Replication — Full Pipeline Abstraction

> This document maps the complete lifecycle of a small language model: from raw internet text to a helpful, instruction-following assistant.

---

## Phase 1: Pretraining (Self-Supervised Learning)

**Goal**: Teach the model the structure of language, facts, code syntax, and reasoning patterns.
**Method**: Next-token prediction on massive raw text corpora.
**Architecture**: Transformer++ (RMSNorm, SwiGLU, RoPE, GQA, KV Cache)

| Spec | Value |
|------|-------|
| Parameters | ~162.8 M (embedding 28.3 M + 30 layers 106.2 M + LM head 28.3 M, untied) |
| Hidden dimension | 576 |
| Layers | 30 |
| Attention heads | 9 (query) / 3 (key-value) — GQA |
| FFN intermediate size | 1536 (SwiGLU) |
| Max sequence length | 8192 |
| RoPE base θ | 10 000 |
| RMSNorm ε | 1e-5 |

### Data Sources (Curated Mixture)

| Dataset | Source | Ratio | Size | What It Contains |
|---------|--------|-------|------|------------------|
| **FineWeb-Edu** | `HuggingFaceFW/fineweb-edu` | 50% | ~110 GB tokenized | High-quality educational web pages (Common Crawl filtered for educational value) |
| **DCLM** | `mlfoundations/dclm-baseline-1.0` | 20% | ~44 GB tokenized | Diverse cleaned language modeling data (books, articles, encyclopedias) |
| **Stack-Edu** | `bigcode/the-stack-dedup` | 10% | ~22 GB tokenized | Educational code repositories (Python, JavaScript, C++, etc.) |
| **FineMath** | `HuggingFaceTB/finemath` | 10% | ~22 GB tokenized | Mathematical text, proofs, derivations, LaTeX-formatted math |
| **Infimm-WebMath** | `OpenCoder-LLM/InfIMMCorpus` | 5% | ~11 GB tokenized | Web-scraped math content with intermediate reasoning steps |
| **Cosmopedia** | `HuggingFaceTB/cosmopedia` | 5% | ~11 GB tokenized | Synthetic textbooks, encyclopedia articles, educational content |

### Pretraining Configuration

| Parameter | Value |
|-----------|-------|
| Total tokens target | ~118 billion (220 GB uint16 shards) |
| Sequence length | 2048 tokens (not 8192; memory constraint) |
| Tokenizer | `HuggingFaceTB/SmolLM2-135M` (BPE, uint16 output) |
| Shard format | `.bin` files (~2 GB each, 1 B tokens) |
| Batch size | 2 per step × 4 gradient accumulation = effective 8 |
| Learning rate | 5e-4 with cosine warmup (2000 steps) + decay |
| Precision | `bfloat16` (mixed) |
| Optimizer | AdamW (β₁=0.9, β₂=0.95, weight_decay=0.1) |
| Gradient clipping | max_norm = 1.0 |
| Compilation | Disabled (`torch.compile = False`) |
| Attention | Flash Attention via `F.scaled_dot_product_attention` |
| Causal mask | No `torch.tril` buffer (saves ~8 GB VRAM) |
| Checkpointing | Every 1000 steps → `checkpoint_latest.pt` + `checkpoint_best.pt` |

### Pretraining Script

```bash
cd /home/kenpeter/work/small
source venv/bin/activate
python3 train.py  # Automatically loads shards from /home/kenpeter/work/data/
```

**Resume capability**: `checkpoint_latest.pt` contains model weights, optimizer state, scheduler state, RNG seeds, and step count. Restarting the script restores exact training state.

---

## Phase 2: Supervised Fine-Tuning (SFT)

**Goal**: Convert the pretrained "text completer" into an instruction-following assistant that understands prompts and generates helpful responses.
**Method**: Train on `(instruction, response)` pairs using next-token prediction (only the response tokens are trained; instruction tokens are masked with `loss_weight=0`).

### SFT Data Sources

| Dataset | Source | Size | Strength |
|---------|--------|------|----------|
| **OpenHermes 2.5** | `teknium/OpenHermes-2.5` | ~1M conversations | Extremely high quality, diverse tasks (coding, reasoning, creative writing, roleplay) |
| **OpenOrca** | `Open-Orca/OpenOrca` | ~4.2M entries | GPT-4 distilled reasoning — massive scale, high reasoning quality |
| **Ultrachat** | `stingning/ultrachat` | ~1.5M multi-turn | Multi-turn conversational dialogue training |
| **Alpaca-GPT4** | `vicgalle/alpaca-gpt4` | ~52K entries | Clean instruction-following format (seed for many derivations) |
| **Code-Alpaca** | `sahil2801/CodeAlpaca-20k` | ~20K coding tasks | Code-specific instruction tuning |

### SFT Data Format

Each sample must be formatted as:

```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Explain quantum computing in simple terms."},
    {"role": "assistant", "content": "Quantum computing uses quantum bits, or qubits..."}
  ]
}
```

**Loss masking**: Only tokens in `assistant` responses contribute to loss. `system` and `user` tokens are masked (`loss_weight=0`).

### SFT Configuration

| Parameter | Value |
|-----------|-------|
| Base model | Output from Phase 1 (`checkpoint_best.pt`) |
| Learning rate | 2e-5 (much lower than pretraining) |
| Epochs | 1-3 (overfitting is a real risk in SFT) |
| Batch size | 16-32 |
| Sequence length | 2048 (SFT samples are shorter than pretraining chunks) |
| LoRA rank | Optional: 64 (if full fine-tuning is too heavy) |
| Precision | `bfloat16` |

### SFT Script (to be created)

```bash
python3 sft.py \
  --base_checkpoint checkpoint_best.pt \
  --dataset OpenHermes-2.5 \
  --lr 2e-5 \
  --epochs 2 \
  --batch_size 16
```

---

## Phase 3: Alignment (RLHF / DPO)

**Goal**: Make the model helpful, harmless, and honest. Reduce toxic outputs, hallucinations, and unsafe responses. Go beyond "imitating good responses" to "understanding preferences."

### Method A: RLHF (Reinforcement Learning from Human Feedback)

**Three steps:**

1. **Collect preference data**: For each prompt, generate 2 responses. Humans (or a strong model) rank which is better.
2. **Train a Reward Model**: A small classifier that predicts "how good is this response?" (outputs scalar reward).
3. **PPO Optimization**: Use the reward model as a critic. The language model (policy) is updated with PPO (Proximal Policy Optimization) to maximize expected reward while staying close to the SFT model (KL divergence penalty).

**RLHF Datasets:**

| Dataset | Source | Size | Description |
|---------|--------|------|-------------|
| **Anthropic HH-RLHF** | `Anthropic/hh-rlhf` | ~170K | Human preference data on helpfulness vs harmlessness |
| **SHP (Stanford Human Preferences)** | `stanfordnlp/SHP` | ~380K | Reddit-based preference data across domains |
| **OpenAssistant Conversations** | `OpenAssistant/oasst1` | ~161K | Human-annotated quality rankings |
| **Ultrafeedback** | `openbmb/UltraFeedback` | ~64K | GPT-4 judged preference pairs with fine-grained scores |

**RLHF Flow:**

```
SFT Model → Generate response pairs → Reward Model scores → PPO updates policy
     ↑___________________________________________________________↓
```

### Method B: DPO (Direct Preference Optimization) ⭐ Recommended for small models

**DPO is the modern replacement for RLHF** — it achieves the same goal without:
- A separate reward model
- Complex PPO training loops
- Unstable hyperparameter tuning

**How DPO works:**
- Instead of learning a reward function and then optimizing with PPO, DPO directly optimizes the policy to satisfy the preference data.
- **Mathematical insight**: The optimal reward model has a closed-form relationship with the optimal policy. DPO exploits this to skip the reward model entirely.
- Loss function: maximize log-ratio of winning response probability vs losing response probability, relative to the reference (SFT) model.

**DPO Datasets** (same as RLHF, but formatted differently):

| Dataset | Format |
|---------|--------|
| **HuggingFace H4/ultrafeedback_binarized** | `{prompt, chosen, rejected}` |
| **Intel/orca_dpo_pairs** | `{question, chatgpt_answer, llama2-13b_answer}` |
| **OpenBMB/UltraFeedback** | `{instruction, response_a, response_b, score_a, score_b}` |

**DPO Data Format:**

```json
{
  "prompt": "Write a Python function to reverse a string.",
  "chosen": "def reverse_string(s):\n    return s[::-1]",
  "rejected": "def reverse_string(s):\n    for i in range(len(s)):\n        print(s[i])"
}
```

### DPO Configuration

| Parameter | Value |
|-----------|-------|
| Base model | SFT output (`sft_best.pt`) |
| Reference model | Frozen SFT model (provides KL anchor) |
| Learning rate | 5e-7 (very low — DPO is sensitive) |
| β (beta) | 0.1-0.5 (controls deviation from reference) |
| Batch size | 4-8 (pairs are processed together) |
| Epochs | 1-2 |
| LoRA | Optional rank 64 for 135M model (full DPO is feasible at this scale) |

### DPO Script (to be created)

```bash
python3 dpo.py \
  --sft_checkpoint sft_best.pt \
  --dataset ultrafeedback_binarized \
  --beta 0.1 \
  --lr 5e-7 \
  --epochs 1
```

---

## Full Pipeline Summary

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          DATA PREPARATION                                │
│  Raw datasets → Tokenize → Shard into .bin → Stream during training      │
│  (FineWeb, DCLM, Stack, FineMath, Infimm, Cosmopedia)                    │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│  PHASE 1: PRETRAINING (Self-Supervised)                                  │
│  Input: Next-token prediction on ~118B tokens                            │
│  Output: Base model — knows language, code, math, facts                   │
│  Script: train.py → checkpoint_best.pt                                     │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│  PHASE 2: SFT (Supervised Fine-Tuning)                                   │
│  Input: (instruction, response) pairs from OpenHermes, OpenOrca, etc.  │
│  Output: Chat model — follows instructions, answers questions            │
│  Script: sft.py → sft_best.pt                                            │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│  PHASE 3: ALIGNMENT (DPO recommended)                                    │
│  Input: Preference pairs (chosen vs rejected)                            │
│  Output: Aligned model — helpful, harmless, honest                       │
│  Script: dpo.py → final_model.pt                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Key Decisions & Rationale

| Decision | Why |
|----------|-----|
| **Keep real curated data** (not synthetic CoT) | Synthetic latent CoT is a post-training technique. Pretraining on real web data + math is the established recipe for base models. |
| **Transformer++ architecture** | SOTA for <1B parameters in 2024-2026 (SmolLM2, Qwen3, Llama 3.2, Gemma 3). Mamba-2/SSD and MoE are alternatives but this is the published baseline. |
| **DPO over RLHF** | DPO is simpler, more stable, and achieves comparable alignment quality. For a 135M model, skipping PPO complexity is pragmatic. |
| **Truncate long texts to 50K chars** | 500K-character outliers cause tokenizer slowdowns. Most information is in the first ~10K tokens anyway. |
| **Only 2 checkpoints** | Disk space conservation. `latest.pt` for resume, `best.pt` for downstream use. |
| **Gradient accumulation = 4** | Simulates larger batch size on limited GPU memory. |

---

## Hardware Requirements

| Phase | GPU VRAM | RAM | Disk | Time Estimate |
|-------|----------|-----|------|---------------|
| Data prep (download + tokenize) | None | 4 GB | 220 GB | ~15-25 hours (parallel wget) |
| Pretraining 163M @ 52-118B tokens | 8 GB (RTX 4070 Ti 12 GB) | 16 GB | 220 GB | ~40-60 hours on single GPU |
| SFT | 4 GB | 8 GB | +2 GB | ~2-4 hours |
| DPO | 4 GB | 8 GB | +2 GB | ~1-2 hours |

---

## Files in This Directory

| File | Purpose |
|------|---------|
| `train.py` | Phase 1: Pretraining script (resumable, Transformer++) |
| `prepare_data_v2.py` | Data preparation: parallel download + batched tokenization |
| `abstraction.md` | This document — full pipeline roadmap |
| `sft.py` | *(Phase 2 — to be created)* |
| `dpo.py` | *(Phase 3 — to be created)* |

---

## References

- [SmolLM2 Technical Report](https://huggingface.co/blog/smollm2) — Architecture & data mixture
- [FineWeb-Edu Paper](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) — Educational web filtering
- [Direct Preference Optimization (Rafailov et al., 2023)](https://arxiv.org/abs/2305.18290) — DPO theory
- [OpenHermes 2.5](https://huggingface.co/datasets/teknium/OpenHermes-2.5) — SFT dataset
- [UltraFeedback](https://huggingface.co/datasets/openbmb/UltraFeedback) — DPO preference data

---

*Last updated: July 13, 2026*
*Status: Phase 1 ready — 49 shards (~98 GB, ~52.6 B tokens) prepared; training configured for batch=2, seq=2048. Data download resuming.*
