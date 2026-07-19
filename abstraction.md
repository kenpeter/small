# SmolLM2-0.5B Training — Full Pipeline Abstraction

> This document maps the complete lifecycle of a small language model: from raw internet text to a helpful, instruction-following assistant.

---

## Phase 1: Pretraining (Self-Supervised Learning)

**Goal**: Teach the model the structure of language, facts, code syntax, and reasoning patterns.
**Method**: Next-token prediction on massive raw text corpora.
**Architecture**: Transformer++ (RMSNorm, SwiGLU, RoPE, GQA, KV Cache)

| Spec | Value |
|------|-------|
| Parameters | ~503 M (embedding 50.3 M + 32 layers 402.4 M + LM head 50.3 M, untied) |
| Hidden dimension | 1024 |
| Layers | 32 |
| Attention heads | 8 (query) / 4 (key-value) — GQA |
| FFN intermediate size | 3072 (SwiGLU) |
| Max sequence length | 8192 |
| RoPE base θ | 10 000 |
| RMSNorm ε | 1e-5 |

### Data Sources (Curated Mixture)

| Dataset | Source | Ratio | Raw Size | What It Contains |
|---------|--------|-------|----------|------------------|
| **FineWeb-Edu** | `HuggingFaceFW/fineweb-edu` | 45% | 64 GB parquet | High-quality educational web pages (Common Crawl filtered for educational value) |
| **FineMath-3Plus** | `HuggingFaceTB/finemath-3plus` | 20% | 58 GB parquet | Advanced mathematical text, proofs, derivations |
| **Cosmopedia** | `HuggingFaceTB/cosmopedia` | 12% | ~80 GB parquet | Synthetic textbooks, encyclopedia articles |
| **OpenWebMath** | `open-web-math/open-web-math` | 10% | 2.5 GB parquet | High-quality math content from the web |
| **FineMath** | `HuggingFaceTB/finemath` | 8% | 2.2 GB parquet | Mathematical text, proofs, derivations |

### Pretraining Configuration

| Parameter | Value |
|-----------|-------|
| Total tokens target | ~15 billion |
| Sequence length | 2048 tokens |
| Tokenizer | `HuggingFaceTB/SmolLM2-135M` (BPE, uint16 output) |
| Shard format | `.bin` files (~256 M tokens each) |
| Batch size | 2 per step × 24 gradient accumulation = effective 48 (98,304 tok/step) |
| Learning rate | 4e-4 with cosine warmup (2000 steps) + decay to 1e-4 |
| Precision | `bfloat16` |
| Optimizer | 8-bit AdamW (bitsandbytes) — reduces optimizer memory from 8→2 bytes/param |
| Weight decay | 0.1 |
| Gradient clipping | max_norm = 1.0 |
| Compilation | Disabled (`torch.compile = False`) |
| Gradient checkpointing | Enabled (essential for 0.5B on 12 GB) |
| Attention | Flash Attention via `F.scaled_dot_product_attention` |
| Checkpointing | Every 2000 steps → `checkpoint_latest.pt` + `checkpoint_best.pt` |

### Pretraining Script

```bash
cd /home/kenpeter/work/small
source venv/bin/activate
python3 train.py  # 0.5B config default, loads .bin shards from data dir
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
│  Raw datasets → Filter + Dedup → Tokenize → Shard into .bin → Train      │
│  (FineWeb-Edu, FineMath-3Plus, Cosmopedia, OpenWebMath, FineMath)        │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│  PHASE 1: PRETRAINING (Self-Supervised)                                  │
│  Input: Next-token prediction on ~15B high-quality tokens                │
│  Output: 0.5B base model — knows language, code, math, facts             │
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
| **0.5B over 1B** | 1B doesn't fit on 12GB with reasonable batch size (51 days at batch=1). 0.5B fits at batch=2 (8.1 GB, 38 days) — the sweet spot for this hardware. |
| **8-bit Adam** | Reduces optimizer memory from 12 bytes/param to 4 bytes/param. Essential for fitting 0.5B with optimizer + gradients + activations on 12 GB. |
| **Gradient checkpointing** | Only store one layer's activations at a time. Without it, 0.5B OOMs at batch=2. |
| **DPO over RLHF** | DPO is simpler, more stable, and achieves comparable alignment quality. For a 0.5B model, skipping PPO complexity is pragmatic. |
| **Truncate long texts to 50K chars** | 500K-character outliers cause tokenizer slowdowns. Most information is in the first ~10K tokens anyway. |
| **Only 2 checkpoints** | Disk space conservation. `latest.pt` for resume, `best.pt` for downstream use. |
| **Gradient accumulation = 24** | Simulates larger effective batch (98K tok/step) for stable training at small batch size. |

---

## Hardware Requirements

| Phase | GPU VRAM | RAM | Disk | Time Estimate |
|-------|----------|-----|------|---------------|
| Data prep (download + filter) | None | 4 GB | 150 GB raw → ~40 GB filtered | ~4 hours (parallel filter) |
| Tokenize → .bin shards | None | 8 GB | ~30 GB tokenized | ~30 min |
| Pretraining 0.5B @ 15B tokens | 8.1 GB (RTX 4070 Ti 12 GB) | 16 GB | ~30 GB shards | ~38 days (batch=2, 4500 tok/s) |
| SFT | 8 GB | 8 GB | +2 GB | ~4 hours |
| DPO | 8 GB | 8 GB | +2 GB | ~2 hours |

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


## Recent History (July 2026)

### 163M → 0.5B Pivot
- Original plan: 163M model on ~20B tokens → loss ~3.0
- 163M trained to ~3.26 loss on sample data
- Decided to pivot to 0.5B for better quality vs training time tradeoff
- Data pipeline rewritten for quality filtering (quality_filter_v3.py)

### Data Pipeline v3
- `quality_filter_v3.py`: heuristic filtering + URL dedup + exact text dedup (MD5) + prefix dedup
- 6-worker parallel processing via ProcessPoolExecutor → OOM with in-memory lists
- Fixed: streaming to temp files → 500 MB/worker
- 4-dataset parallel launch (3 workers each) for faster throughput

### Model & Config Decisions
- 0.5B config: dim=1024, L=32, h=8, kv=4, ffn=3072
- 8-bit Adam (bitsandbytes) instead of standard AdamW to fit optimizer on 12 GB VRAM
- Gradient checkpointing essential (0.5B OOMs at batch=2 without it)
- `view()` → `reshape()` fixes needed for non-contiguous tensors with gradient checkpointing
- Batch=2, accum=24, seq=2048 → 4,500 tok/s → ~38 days for 15B tokens

### Next Steps
- Wait for data filtering → tokenize → start 0.5B pretraining
- After pretraining: SFT on instruction data, then DPO for alignment
