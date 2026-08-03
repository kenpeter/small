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

| Directory | Size | Tokens | Domain | Status |
|---|---|---|---|---|
| `_shards_math_{easy,medium,hard}/` | 25.6B easy / 144M med / 5.1G hard | — | Math (finemath-3plus + open-web-math) | ✅ Active |
| `_shards_web_{easy,medium,hard}/` | 21G / 5.3G / 1.2G | — | Web (fineweb-edu + QuRatedPajama) | ✅ Active |
| `_shards_synth_*` | 72M easy / 757M med / 59M hard | — | Synthetic (cosmopedia) | ✅ Active |
| `_shards_code_*` | 75M easy / 132M med / 5.9M hard | — | Code (github-code) | ✅ Active |
| `_shards_reformat_easy/` | ~8.5M | — | Textbook + QA (reformatted) | ✅ Active |

**Notes:**
- Shard format: `.bin` uint16 arrays, SmolLM2-135M vocab (49152)
- Tiered per domain: easy → medium → hard; consumed as a flat farm via `FARM_DIR` (`_shards_final`) in pretrain_megatrain.py (x-small style, `FlatFarmDataset`). G1-G4 stratified sampling (`SHARD_DIRS`) kept but dormant.

### Raw Downloads (Staging → Needs Tokenization)

| Dataset | Files | Size | Total Expected | Domain | Status |
|---|---|---|---|---|---|
| `fineweb-edu/` | 42 `.parquet` | 1.91 GB | ~10 GB | Web / educational | ✅ Complete |
| `finemath-3plus/` | 24 `.parquet` | 4.71 GB | ~28 GB | Math (grade 3+) | ✅ Complete |
| `cosmopedia/` | — | — | ~8–10 GB | Synthetic / encyclopedic | ✅ Complete |
| `open-web-math/` | — | — | ~7–8 GB | Math / research | ✅ Complete |

**Notes:**
- Download worker: `download_3workers_direct.py` (3 workers, resume + size validation) → `_raw_original/`
- QRP (QuRatedPajama-260B) web-hard source: `download_qrp_par.py` (adaptive parallel wget, resumable)

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
│  RAW STAGING (_raw_original/)                                          │
│    fineweb-edu      ──►                                               │
│    finemath-3plus   ──►──► tokenize_domain_parallel.py ──►             │
│    cosmopedia       ──►      (per-domain easy/medium/hard .bin)        │
│    open-web-math    ──►                                                │
│    github-code      ──►                                                │
└─────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  PRETRAIN (pretrain_megatrain.py)                                       │
│    _shards_final/   ──► FlatFarmDataset (x-small style, sequential)     │
│    1098-bin farm    ──► md5 split: 1088 train / 10 val                  │
│    G1-G4 stratified ──► legacy, dormant (StratifiedShardDataset)        │
└─────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  SFT / RLHF (planned)                                                  │
│    _sft_final_shards/  ──► supervised fine-tuning                      │
│    _cot_raw/           ──► reasoning boost                            │
│    bluemoon_roleplay/  ──► conversational style                        │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Pretraining Configuration (Current)

| Parameter | Value |
|-----------|-------|
| Total tokens ready | ~78 GB across 5 domains (per-domain easy/medium/hard tiered shards) |
| Sequence length | 2048 tokens |
| Tokenizer | `HuggingFaceTB/SmolLM2-135M` (BPE, uint16 output) |
| Shard format | `.bin` uint16 arrays, per tier (see Data Inventory) |
| Batch size | 8 × grad-accum 4 = effective 32 (65,536 tok/step) |
| Precision | `bfloat16` |
| Optimizer | **AdamW** (3e-4 → cosine → 1e-6); MuonClip was removed after plateauing at loss ~5.0 |
| AdamW betas | (0.9, 0.95) |
| Weight decay | 0.1 |
| Gradient clipping | max_norm = 1.0 |
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

**Current training state:** AdamW from step 0. Resume via `megatrain_latest.pt`.

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
│  Raw datasets → Filter + Tier → Tokenize → Shard into .bin → Train     │
│  (FineWeb-Edu, FineMath-3Plus, Cosmopedia, OpenWebMath, GitHub-Code)   │
│  Output: per-domain easy/medium/hard shards (see Data Inventory)        │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│  PHASE 1: PRETRAINING (Self-Supervised)                                  │
│  Input: Next-token prediction on ~78 GB tiered tokens                    │
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
| **~78 GB tiered data (not 15B)** | Best-of-best filtered data across 5 domains. Quality over quantity. |
| **Transformer++ architecture** | SOTA for <1B parameters (SmolLM2, Qwen3, Llama 3.2). |
| **AdamW optimizer** | Stable and proven on 12GB VRAM. MuonClip caused a plateau at loss ~5.0 and was removed. |
| **QK-Clip + momentum warmup** | Stabilizes MuonClip on fresh init. Removed with MuonClip. |
| **Batch=8, accum=4** | Effective batch 32 (65K tok/step). |
| **Only 2 checkpoints** | Disk space conservation. `megatrain_latest.pt` for resume, `megatrain_best.pt` for downstream. |
| **Flat farm data org (x-small style)** | `_shards_final` uniform random interleave, sequential consumption. G1–G4 stratified mix kept dormant. |
| **13-gram dedup** | Exact hash collision drop. 5–10% token savings. Disabled at startup to avoid long scan; toggled via flag. |
| **Direct wget downloads** | Bypasses HF API / Xet throttling. 3–8× faster than hf_hub_download for large parquet files. |
| **Power limit 180W** | Keeps GPU ~72–77°C vs 86°C. Persistence mode enabled. |

---

## Hardware Requirements

| Phase | GPU VRAM | RAM | Disk | Time Estimate |
|-------|----------|-----|------|---------------|
| Pretraining 1B @ ~78 GB tokens | ~5.1 GB (RTX 4070 Ti 12 GB, power limit 180W) | 93 GB | ~7.9 GB shards | — |
| SFT | 8 GB | 8 GB | +24 GB | ~4 hours |
| DPO | 8 GB | 8 GB | +2 GB | ~2 hours |

---

## Files in This Directory

| File | Purpose |
|------|---------|
| `pretrain_megatrain.py` | Pretraining script (1B config, resumable, MegaTrain CPU-offload + AdamW) |
| `run_pretrain.sh` | Wrapper that unsets bad env vars and launches pretrain_megatrain.py |
| `download_3workers_direct.py` | 3-worker wget downloader with resume for raw datasets |
| `download_qrp_par.py` | Adaptive parallel wget downloader for QuRatedPajama (resumable) |
| `tokenize_domain_parallel.py` | Generic tiered tokenizer (math/code/synth/web → easy/medium/hard .bin shards) |
| `tokenize_reformat.py` | Tokenizer for reformatted textbook+QA JSONL → reformat_easy shards |
| `reformat_data.py` | Kimi K2-style data reformatting via vLLM chat API |
| `quick_eval_pretrain.py` | Quick eval of 1B checkpoint (generation + perplexity) |
| `check_status.sh` | Combined status report: training + downloads + disk |
| `test_pretrain.py` | Test suite for pretrain_megatrain.py |
| `abstraction.md` | This document — full pipeline roadmap |

---

## Recent History (July 2026)

### 163M → 1B Pivot
- Original: 163M on ~20B tokens → trained to ~3.26 loss
- Pivoted to 0.5B, then to 1B after discovering 8-bit Adam + CPU offloading
- All old checkpoints cleaned up (163M SFT/DPO/DPT models removed)

### Data Pipeline
- `download_3workers_direct.py` → raw parquet downloads in `_raw_original/`
- `tokenize_domain_parallel.py` → filter (via `filter_all.py`) + tier + tokenize → per-domain .bin shards
- Intermediate data cleaned up post-processing
- **CRITICAL**: Never delete raw parquet downloads in `_raw_original/` — they're the source of truth.

### Optimizer History (July 2026)
- AdamW → MuonClip (2026-07-22): warm-starting Muon from AdamW caused loss jump 4.98 → 10.12 → oscillation; deleted checkpoints, restarted fresh with Kimi K2 MuonClip.
- **MuonClip → AdamW (reverted)**: Muon plateaued at loss ~5.0. Switched back to `torch.optim.AdamW` (3e-4 → cosine → 1e-6) with G1–G4 curriculum.
- `non_blocking=False` fix in `cpu_master.py` (race condition on D2H copies).

### Disk Space Management
- 506 GB total. 151 GB used, 330 GB free.
- Data prep creates ~280 GB intermediate files — always clean up staging/filtered dirs after each run.
- Checkpoint rules: keep only latest + best per phase (~12 GB total for 1B model).

---

## Current State (2026-08-03)

- **Training:** fresh 1B run on flat farm (x-small org), AdamW — see train_small.log
- **Data:** all 5 domains downloaded and tokenized into per-domain easy/medium/hard tiered shards
- **Reformat:** textbook + QA reformatted via Qwen3-8B (14 MB, reformat_easy tier)
- **Next:** reach loss 2.7 → SFT → DPO

---

## Summary

- **Tokens ready:** ~78 GB across 5 domains (math, web, code, synth, reformat), tiered easy/medium/hard
- **Training:** 1B pretraining at loss ~2.77 → 2.7 with AdamW + G1–G4 curriculum
- **SFT assets ready:** 25.76 GB
- **Next action:** reach loss 2.7 → SFT → DPO.
