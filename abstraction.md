# 1B Training — Full Pipeline Abstraction

> This document maps the complete lifecycle: from raw internet text to an instruction-following assistant.

---

## Phase 1: Pretraining (Self-Supervised Learning)

**Goal**: Teach the model language structure, facts, code syntax, and reasoning.  
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

---

## Data Inventory (Current — Updated 2026-07-22)

### Pretraining Tokens (Ready to Feed)

| Directory | Files | Size | Tokens | Domain | Status |
|---|---|---|---|---|---|
| `_shards_final/` | 17 `.bin` | 7.94 GB | ~3.97B uint16 | 100% FineWeb-Edu (web text) | ✅ Active |

**Notes:**
- Shard size target: 512 MB (`shard_000000.bin` = 260 MB, `shard_000016.bin` = 0 B empty)
- Tokenizer: SmolLM2-135M vocab (49152)
- Currently **100% web** — math / synthetic shards do not exist yet

### Raw Downloads (Staging → Needs Tokenization)

| Dataset | Files | Size | Total Expected | Domain | Status |
|---|---|---|---|---|---|
| `fineweb-edu/` | 42 `.parquet` | 1.91 GB | ~10 GB | Web / educational | ✅ Complete |
| `finemath-3plus/` | 24 `.parquet` | 4.71 GB | ~28 GB | Math (grade 3+) | ⬇️ In Progress (24/128) |
| `cosmopedia/` | — | — | ~8–10 GB | Synthetic / encyclopedic | ❌ Not started |
| `open-web-math/` | — | — | ~7–8 GB | Math / research | ❌ Not started |

**Notes:**
- Download worker: `download_3workers_direct.py` (3 workers, 1.5 s stagger)
- Missing datasets block true 60/25/15 stratified mix

### SFT / Instruction Data (Post-Pretraining)

| Directory | Files | Size | Sources | Status |
|---|---|---|---|---|
| `_sft_final_shards/` | 71 `.pt` | 25.76 GB | Alpaca-GPT4, Code-Alpaca, OpenHermes, etc. | ✅ Ready for SFT |

**Notes:**
- Pre-tokenized `.pt` shards (not `.bin`)
- Consumed after base model pretraining finishes

### Math / Code Raw Datasets (LeetCode Cluster)

| Dataset | Size | Format | Quality | Notes |
|---|---|---|---|---|
| `LeetCode_YT_CC_CoT_Summary/` | 0.67 GB | mixed | Medium | YouTube + CoT summaries |
| `newfacade_LeetCodeDataset/` | 0.10 GB | `.jsonl` | Medium | Train + test split |
| `high_quality_leetcode/` | 0.05 GB | `.jsonl` | Medium | Filtered subset |
| `greengerong_LeetCode/` | 0.02 GB | `.jsonl` | Low-Med | Java/Python solutions |
| Others (DenCT, LimYeri, NanDo, juyoungml, mesolitica, vovw) | ~0.06 GB | `.parquet` | Low | Tiny / niche |

**Verdict:** ~0.88 GB total. Small, heterogeneous, mostly LeetCode solutions. **Not currently used in pretraining.** Could be filtered + tokenized into a "code" domain bucket if desired.

### Other / Auxiliary

| Dataset | Size | Format | Purpose | Status |
|---|---|---|---|---|
| `_cot_raw/numina_50k.jsonl` | 0.07 GB | `.jsonl` | Chain-of-Thought reasoning | Hold for SFT or synthetic mix |
| `bluemoon_roleplay/` | 0.27 GB | `.json` + `.arrow` | Roleplay / conversational | Hold for SFT |

---

## Pipeline Mapping

```
┌─────────────────────────────────────────────────────────────────────────┐
│  RAW STAGING (_staging_multi/)                                         │
│    fineweb-edu      ──►                                               │
│    finemath-3plus   ──►──► tokenize.py ──► _shards_*/    .bin          │
│    cosmopedia       ──►      (SmolLM2 vocab)                           │
│    open-web-math    ──►                                                │
└─────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  PRETRAIN (pretrain_megatrain.py)                                       │
│    _shards_final/   ──► StratifiedBinShardDataset                       │
│    _shards_math/    ──► (60/25/15 web/math/synth)                    │
│    _shards_synth/   ──► dedup=True (disabled at startup)              │
└─────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  SFT / RLHF                                                           │
│    _sft_final_shards/  ──► supervised fine-tuning                      │
│    _cot_raw/           ──► reasoning boost                            │
│    bluemoon_roleplay/  ──► conversational style                        │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Pretraining Configuration (Current)

| Parameter | Value |
|-----------|-------|
| Total tokens ready | ~3.97 billion (top-10% filtered FineWeb-Edu) |
| Sequence length | 2048 tokens |
| Tokenizer | `HuggingFaceTB/SmolLM2-135M` (BPE, uint16 output) |
| Shard format | `.bin` files (~256 M tokens each, 17 shards) |
| Batch size | 1 per step × 12 gradient accumulation = effective 12 (24,576 tok/step) |
| Precision | `bfloat16` |
| Optimizer | **Kimi K2 MuonClip** (Newton-Schulz 5-step, RMS scaling, QK-Clip, momentum warmup) |
| Muon lr (2D weights) | 0.01 |
| AdamW lr (1D scalars/embed/head) | 0.003 |
| AdamW betas | (0.8, 0.95) for 2D weights; (0.9, 0.95) for scalars |
| Momentum warmup | 0.85 → 0.95 over first 300 steps |
| Weight decay | 0.1 |
| QK-Clip tau | 100 (every optimizer step) |
| Gradient clipping | Disabled on Muon params; max_norm = 1.0 on AdamW params |
| Compilation | Disabled (`torch.compile = False`) |
| Gradient checkpointing | Enabled |
| CPU offloading | Enabled (CPUMasterModel) |
| Attention | Flash Attention via `F.scaled_dot_product_attention` |
| Checkpointing | Every 2000 steps → `megatrain_latest.pt` + `megatrain_best.pt` |

### Pretraining Script

```bash
cd /home/kenpeter/work/small
source venv/bin/activate
bash run_pretrain.sh
```

**Current training state:** Fresh MuonClip from step 0. PID 442720. Step 120 loss ~8.61. ~93 s/step. First checkpoint at step 2000.

---

## Phase 2: Supervised Fine-Tuning (SFT)

**Goal**: Convert the pretrained "text completer" into an instruction-following assistant.  
**Method**: Train on `(instruction, response)` pairs using next-token prediction.  

SFT data is pre-tokenized and stored in `_sft_final_shards/` (25.76 GB, 71 shards).

---

## Phase 3: Alignment (DPO)

Not yet started. DPO data was cleaned up — will need fresh preparation when ready.

---

## Full Pipeline Summary

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          DATA PREPARATION                                │
│  Raw datasets → Filter + Dedup → Tokenize → Shard into .bin → Train    │
│  (FineWeb-Edu, FineMath-3Plus, Cosmopedia, OpenWebMath, FineMath)      │
│  Output: _shards_final/ (17 shards, ~3.97B tokens)                    │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│  PHASE 1: PRETRAINING (Self-Supervised)                                  │
│  Input: Next-token prediction on ~3.97B high-quality tokens              │
│  Output: 1B base model — knows language, code, math, facts              │
│  Script: pretrain_megatrain.py → megatrain_best.pt                       │
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
| **1B model on 12 GB VRAM** | Gradient checkpointing + CPUMasterModel offloading makes it fit (~5.1 GB VRAM active) |
| **3.97B tokens (not 15B)** | Best-of-best filtered data (~90% rejection rate). Quality over quantity. |
| **Transformer++ architecture** | SOTA for <1B parameters (SmolLM2, Qwen3, Llama 3.2). |
| **Kimi K2 MuonClip optimizer** | ~40% fewer steps vs AdamW. Newton-Schulz orthogonal updates explore full-rank space. |
| **QK-Clip + momentum warmup** | Stabilizes MuonClip on fresh init. Without these, MuonClip NaNs immediately. |
| **Batch=1, accum=12** | Effective batch 12 (24K tok/step). Smaller than old accum=48 because MuonClip overhead eats ~2–3 s/step. |
| **Only 2 checkpoints** | Disk space conservation. `megatrain_latest.pt` for resume, `megatrain_best.pt` for downstream. |
| **Stratified 60/25/15 mix** | Forces domain balance per batch. Prevents web-only collapse. Currently inactive (missing math/synth shards). |
| **13-gram dedup** | Exact hash collision drop. 5–10% token savings. Disabled at startup to avoid long scan; toggled via flag. |
| **Direct curl downloads** | Bypasses HF API 502 errors. 3–8× faster than hf_hub_download for large parquet files. |
| **Power limit 180W** | Keeps GPU ~72–77°C vs 86°C. Persistence mode enabled. |

---

## Hardware Requirements

| Phase | GPU VRAM | RAM | Disk | Time Estimate |
|-------|----------|-----|------|---------------|
| Pretraining 1B @ 3.97B tokens | ~5.1 GB (RTX 4070 Ti 12 GB, power limit 180W) | 93 GB | 7.94 GB shards | ~52 hours to first checkpoint (2000 steps) |
| SFT | 8 GB | 8 GB | +24 GB | ~4 hours |
| DPO | 8 GB | 8 GB | +2 GB | ~2 hours |

---

## Files in This Directory

| File | Purpose |
|------|---------|
| `train.py` | Phase 1: Pretraining script (1B config, resumable, Transformer++) |
| `pretrain_megatrain.py` | Alternate pretraining script using MegaTrain CPU-offload engine + MuonClip |
| `run_pretrain.sh` | Wrapper that unsets bad env vars and launches pretrain_megatrain.py |
| `download_3workers_direct.py` | 3-worker curl downloader with stagger for raw datasets |
| `tokenize_final.py` | Tokenization script: raw parquet → .bin shards |
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
- Intermediate data cleaned up post-processing
- **CRITICAL**: Never delete raw parquet downloads in `_staging_multi/` — they're the source of truth.

### AdamW → MuonClip Pivot (2026-07-22)
- Warm-starting Muon from AdamW checkpoint caused loss jump 4.98 → 10.12 → oscillation
- **Deleted all old checkpoints.** Started fresh from step 0 with full Kimi K2 MuonClip.
- Includes: Newton-Schulz 5-step, RMS scaling, QK-Clip, momentum warmup 0.85→0.95, separate AdamW groups for 1D params.
- `non_blocking=False` fix in `cpu_master.py` (race condition on D2H copies).

### Disk Space Management
- 506 GB total. 151 GB used, 330 GB free.
- Data prep creates ~280 GB intermediate files — always clean up staging/filtered dirs after each run.
- Checkpoint rules: keep only latest + best per phase (~12 GB total for 1B model).

---

## Current Blockers

| Blocker | Impact | Fix |
|---|---|---|
| `finemath-3plus` download incomplete | Cannot build math shards | Wait for 128/128 files |
| `cosmopedia` missing | Cannot build synthetic shards | Queue download after finemath |
| `open-web-math` missing | Math diversity gap | Queue download after cosmopedia |
| `_shards_math/` does not exist | Stratified loader falls back to web | Tokenize finemath → _shards_math/ |
| `_shards_synth/` does not exist | Stratified loader falls back to web | Tokenize cosmopedia → _shards_synth/ |

---

## Summary

- **Tokens ready now:** ~3.97B (web only)
- **Tokens needed for 60/25/15 mix:** ~6.6B web + ~2.75B math + ~1.65B synthetic = ~11B total
- **SFT assets ready:** 25.76 GB
- **Next action:** Complete downloads → tokenize math + synth → update `SHARD_DIRS` → restart training with true stratified mix.
