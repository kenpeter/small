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
| Total tokens | ~3.97 billion (best-of-best filtered) |
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
python3 train.py  # 1B config default, loads .bin shards from _shards_final
```

**Resume capability**: `megatrain_latest.pt` contains model weights, optimizer state, scheduler state, RNG seeds, and step count. Restarting the script restores exact training state.

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
| Pretraining 1B @ 3.97B tokens | ~5 GB (RTX 4070 Ti 12 GB) | 16 GB | 7.9 GB shards | ~30 days (batch=1, ~1500 tok/s) |
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
- Checkpoint rules: keep only latest + best per phase (~4 GB total)

### Current Config
- 1B model: dim=1536, L=32, h=12, kv=4, ffn=4608
- batch=1, accum=48, seq=2048
- 8-bit Adam + gradient checkpointing + optional CPU offload
