# 1B Training — Full Pipeline Abstraction

> This document maps the complete lifecycle of a small language model: from raw internet text to a helpful, instruction-following assistant.

---

## Phase 1: Pretraining (Self-Supervised Learning)

**Goal**: Teach the model the structure of language, facts, code syntax, and reasoning patterns.
**Method**: Next-token prediction on massive raw text corpora.
**Architecture**: Transformer++ (RMSNorm, SwiGLU, RoPE, GQA)

### Architecture

| Spec | Value |
|------|-------|
| Total parameters | 1,031,898,624 (~1032 M) |
| Embedding params | 75,497,472 (49152 × 1536) |
| Per-layer params | 27,528,192 |
| LM head params | 75,497,472 (1536 × 49152, untied) |
| Hidden dimension (dim) | 1536 |
| Layers (n_layers) | 32 |
| Attention heads (n_heads) | 12 |
| Key-value heads (n_kv_heads) | 4 (GQA ratio 3:1) |
| Head dimension (head_dim) | 128 |
| FFN intermediate size | 4608 (SwiGLU: gate+up+down) |
| Max sequence length | 8192 |
| RoPE base θ | 10,000 |
| RMSNorm ε | 1e-5 |
| Dropout | 0.0 |
| Weight tying | No (embedding ≠ lm_head) |
| Activation | SiLU (via SwiGLU) |
| Position encoding | Rotary (RoPE) |

### Per-Layer Breakdown

| Component | Shape | Params |
|-----------|-------|--------|
| `q_proj` | 1536 × 1536 | 2,359,296 |
| `k_proj` | 1536 × 512 | 786,432 |
| `v_proj` | 1536 × 512 | 786,432 |
| `o_proj` | 1536 × 1536 | 2,359,296 |
| `gate_proj` | 1536 × 4608 | 7,077,888 |
| `up_proj` | 1536 × 4608 | 7,077,888 |
| `down_proj` | 4608 × 1536 | 7,077,888 |
| `attn_norm` | 1536 | 1,536 |
| `mlp_norm` | 1536 | 1,536 |
| **Layer total** | | **27,528,192** |

## Data Inventory (Current)

**Disk total**: 506 GB — **151 GB used**, 330 GB free

| What | Location | Size | Format | Status |
|------|----------|------|--------|--------|
| **Pretrain .bin shards** | `_shards_final/` | **7.4 GB** | 16× `.bin` (uint16, 3.97B tokens) | ✅ Uploaded to HF (`kenpeter123/small-pretrain-data`) |
| **SFT .pt shards** | `_sft_final_shards/` | **24 GB** | 71× `.pt` (43 OpenOrca, 15 UltraChat, 11 OpenHermes, 1 Alpaca, 1 CodeAlpaca) | 🕐 Uploading to HF |
| **Checkpoints** | `checkpoints/` | **12 GB** | 2× `.pt` (megatrain_latest + megatrain_best) | Active training |
| **Raw parquet (downloading)** | `_staging_multi/` | ~**15-100 GB** | `.parquet` (FineWeb-Edu 200 files, FineMath-3Plus, Cosmopedia, OpenWebMath) | 🕐 Downloading (low-impact, 3 workers) |
| **LeetCode datasets** | various | **~1 GB** | `.parquet` | Small code data |
| **CoT raw** | `_cot_raw/` | **67 MB** | text | Small reasoning data |

### Original Data Sources (Before Filtering)

| Dataset | Source | Raw Size | What It Contains |
|---------|--------|----------|------------------|
| **FineWeb-Edu** | `HuggingFaceFW/fineweb-edu` | ~180 GB (2410 parquet files) | High-quality educational web pages (Common Crawl) |
| **FineMath-3Plus** | `HuggingFaceTB/finemath` | ~58 GB (128 parquet files) | Advanced mathematical text |
| **Cosmopedia** | `HuggingFaceTB/cosmopedia` | ~80 GB (13 parquet files) | Synthetic textbooks |
| **OpenWebMath** | `open-web-math/open-web-math` | ~2.5 GB (114 parquet files) | High-quality math content |
| **FineMath** | `HuggingFaceTB/finemath` | ~2.2 GB | Mathematical text |

### Pipeline History

- **First pass (deleted)**: Raw parquets downloaded → aggressive heuristics (~90% rejection) → 3.97B tokens → **raw parquets deleted**
- **Current download (in progress)**: Re-downloading with `download_data_relaxed.py`. Low-impact (3 workers, nice 19). Will keep raws permanently.
- **Plan**: Dedup-only processing (skip heuristics) → tokenize to `.bin` → combine with existing 3.97B shards → continue training.

### Pretraining Configuration

| Parameter | Value |
|-----------|-------|
| Total tokens so far | ~3.97 billion (top-10% filtered) |
| Sequence length | 2048 tokens |
| Tokenizer | `HuggingFaceTB/SmolLM2-135M` (BPE, uint16 output) |
| Shard format | `.bin` files (~256 M tokens each, 17 shards) |
| Batch size | 1 per step × 48 gradient accumulation = effective 48 (98,304 tok/step) |
| Learning rate | 4e-4 with cosine warmup (2000 steps) + decay to 1e-4 |
| Precision | `bfloat16` |
| Optimizer | 8-bit AdamW (bitsandbytes) — reduces optimizer memory from 8→2 bytes/param |
| Weight decay | 0.1 |
| Gradient clipping | max_norm = 1.0 |
| Compilation | Disabled (`torch.compile = False`) |
| Gradient checkpointing | Enabled (essential for 1B on 12 GB) |
| CPU offloading | Optional (`cpu_offload=True`) — moves model to CPU between steps |
| Attention | Flash Attention via `F.scaled_dot_product_attention` |
| Checkpointing | Every 2000 steps → `megatrain_latest.pt` + `megatrain_best.pt` |

### Pretraining Script

```bash
cd /home/kenpeter/work/small
source venv/bin/activate
python3 pretrain_megatrain.py --batch-size 8 --grad-accum 6 --lr 4e-4 --num-steps 242188 --num-grad-slabs 6
```

**Resume capability**: `megatrain_latest.pt` contains model weights, optimizer state, scheduler state, RNG seeds, and step count. Restarting the script restores exact training state. Active training: step ~6000, loss ~4.23, ~2.8k tok/s.

---

## Phase 2: Supervised Fine-Tuning (SFT)

**Goal**: Convert the pretrained "text completer" into an instruction-following assistant.
**Method**: Train on `(instruction, response)` pairs using next-token prediction.

### SFT Data

SFT data is pre-tokenized and stored in `_sft_final_shards/` (24 GB, 71 shards). Raw LeetCode datasets (~1 GB) available in JSONL format for conversion.

---

## Phase 3: Alignment (DPO)

Not yet started. DPO data was cleaned up — will need fresh preparation when ready.

---

## Full Pipeline Summary

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          DATA PREPARATION                                │
│  Raw datasets → Filter + Dedup → Tokenize → Shard into .bin → Train      │
│  (FineWeb-Edu, FineMath-3Plus, Cosmopedia, OpenWebMath, FineMath)        │
│  Output: _shards_final/ (17 shards, ~3.97B tokens)                      │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│  PHASE 1: PRETRAINING (Self-Supervised)                                  │
│  Input: Next-token prediction on ~3.97B high-quality tokens              │
│  Output: 1B base model — knows language, code, math, facts              │
│  Script: train.py → megatrain_best.pt                                    │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│  PHASE 2: SFT (Supervised Fine-Tuning)                                   │
│  Input: (instruction, response) pairs from _sft_final_shards/            │
│  Output: Chat model — follows instructions, answers questions            │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│  PHASE 3: ALIGNMENT (DPO recommended)                                    │
│  Input: Preference pairs (chosen vs rejected)                            │
│  Output: Aligned model — helpful, harmless, honest                       │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Key Decisions & Rationale

| Decision | Why |
|----------|-----|
| **1B model on 12 GB VRAM** | 8-bit Adam + gradient checkpointing + optional CPU offloading make it fit (~4 GB model states + ~1 GB activations) |
| **3.97B tokens (not 15B)** | Best-of-best filtered data (~90% rejection rate). Quality over quantity. Enough for a competent 1B base. |
| **Transformer++ architecture** | SOTA for <1B parameters (SmolLM2, Qwen3, Llama 3.2). |
| **8-bit Adam** | Reduces optimizer memory from 12 bytes/param to 2 bytes/param. |
| **Gradient checkpointing** | Only store one layer's activations at a time. Without it, 1B OOMs at batch=1. |
| **CPU offloading** | Falls back to CPU between steps if GPU memory gets tight. Model moves to CPU after optimizer step, back to GPU for next step. |
| **Batch=1, accum=48** | Effective batch 48 (98K tok/step) for stable training. |
| **Only 2 checkpoints** | Disk space conservation. `megatrain_latest.pt` for resume, `megatrain_best.pt` for downstream use. |

---

## Hardware Requirements

| Phase | GPU VRAM | RAM | Disk | Time Estimate |
|-------|----------|-----|------|---------------|
| Pretraining 1B @ 3.97B tokens | 3.2 GB (RTX 4070 Ti 12 GB, power limit 150W) | 93 GB | 7.4 GB shards | ~3 days (batch=8, accum=6, ~2.8k tok/s) |
| SFT | 8 GB | 8 GB | +24 GB | ~4 hours |
| DPO | 8 GB | 8 GB | +2 GB | ~2 hours |

---

## Files in This Directory

| File | Purpose |
|------|---------|
| `train.py` | Phase 1: Pretraining script (1B config, resumable, Transformer++) |
| `pretrain_megatrain.py` | Alternate pretraining script using MegaTrain CPU-offload engine |
| `quality_filter_v3.py` | Data quality filtering pipeline (heuristic + dedup + tokenize) |
| `abstraction.md` | This document — full pipeline roadmap |

---

## Recent History (July 2026)

### 163M → 1B Pivot
- Original: 163M on ~20B tokens → trained to ~3.26 loss
- Pivoted to 0.5B, then to 1B after discovering 8-bit Adam + CPU offloading
- All old checkpoints cleaned up (163M SFT/DPO/DPT models removed)

### Data Pipeline
- `quality_filter_v3.py`: heuristic filtering + URL dedup + exact text dedup (MD5) + prefix dedup
- Best-of-best filtering: ~90% rejection rate → 3.97B tokens
- Intermediate data (_staging_multi, _filtered_best) cleaned up post-processing

### Disk Space Management
- 506 GB total. Data prep creates ~280G intermediate files — always clean up staging/filtered dirs after each run.
- **CRITICAL**: Never delete raw parquet downloads in `_staging_multi/` — they're the source of truth. Without them, we can't re-tokenize with different filters.
- Checkpoint rules: keep only latest + best per phase (~12 GB total for 1B model)

### Current Config
- 1B model: dim=1536, L=32, h=12, kv=4, ffn=4608
- batch=8, grad_accum=6, slabs=6, seq=2048 (MegaTrain)
- PyTorch AdamW (CPU) — DeepSpeedCPUAdam caused NaN at step 36
- GPU power limit: 150W (temperatures ~70-74°C)
- Training: ~2.8k tok/s, GPU memory 3.16 GB/12.3 GB
- Loss trajectory: 11.0 (step 0) → 4.23 (step 6000)
