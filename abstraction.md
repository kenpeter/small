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
- Tiered per domain: easy → medium → hard; consumed via the stratified curriculum sampler (`SHARD_DIRS`, `--curriculum`) — **Boundary Sharpening · Cyclic Scheduling · Curriculum Continuity · Local Diversity (G1–G4)** — with the flat farm (`FARM_DIR`, `FlatFarmDataset`) as the no-curriculum fallback.

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
│  PRETRAIN (pretrain_gpu.py — pure GPU)                                    │
│    _shards_final/   ──► FlatFarmDataset (x-small style, sequential)     │
│    1088-bin farm    ──► md5 split: 1088 train / 10 val                  │
│  Curriculum sampler (G1–G4: Sharpening/Scheduling/Continuity/Diversity)│
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

## Curriculum Pipeline — Boundary Sharpening · Cyclic Scheduling · Curriculum Continuity · Local Diversity (G1–G4)

> **Status: ACTIVE (2026-08-04)** — the pure-GPU run uses `StratifiedShardDataset`
> with `--curriculum`: smooth **2-fold reverse** curriculum (see diagram). The flat farm
> (`_shards_final`) remains the fallback when `--curriculum` is off.

```
 tiered .bin shards — 5 domains × 3 tiers (13 dirs, SHARD_DIRS)
   math │ web │ code │ synth │ reformat   ×   easy │ medium │ hard
        │
        ▼
┌─ ① Boundary Sharpening (G1) — 2-FOLD REVERSE ────────────────────────┐
│   fold 1 (t<0.5): easy 0.30→0.175 · hard 0.25→0.475 (easy→hard)    │
│   fold 2 (t≥0.5): MIRRORED hard→easy — start ≈ end easy-heavy,     │
│   hard peaks mid-run → order bias cancels (paper STR/SAW-2)         │
└───────────────────────────────────────────────────────────────────┘
        │  tier weights × within-tier domain splits (sum=1/tier)
        ▼
┌─ ② Curriculum Continuity (G3) ──────────────────────────────────────┐
│   ratios recomputed every 2000 steps → blend glides, no cliffs    │
└───────────────────────────────────────────────────────────────────┘
        │
        ▼
┌─ ③ Cyclic Scheduling (G2) — review wave ────────────────────────────┐
│   easy gets periodic boost +0.12·cos(2π·step/cycle) (anti-        │
│   forgetting), cycle = ⅛ of training → renormalize               │
└───────────────────────────────────────────────────────────────────┘
        │  final per-domain ratios → weighted shard sampling
        ▼
┌─ ④ Local Diversity (G4) — windowed shuffle ─────────────────────────┐
│   JIT windowed shuffle (w=5000): jitter batch order locally,      │
│   keep global easy→hard trend                                     │
└───────────────────────────────────────────────────────────────────┘
        │
        ▼
   TRAINING — every batch mixes all 5 domains; tier ratio glides
   easy-heavy → HARD PEAK mid-run → easy-heavy (mirrored, bias cancels)
```

**Worked example (t = 0.5, mid-training = fold peak):** base weights easy .175 / med .35 / hard .475;
at the Cyclic Scheduling review peak (G2, ~+0.12 to easy) → renormalized ≈ easy .29 / med .31 / hard .40 —
a visible "easy review week" inside the mid-run hard peak. After t=0.5 the curve mirrors
back: easy rises toward .30, hard falls toward .25 → start ≈ end (order bias cancels).

**How it differs from the active flat farm:** flat farm = *uniform random interleave of
ALL 1088 tiered shards, no curriculum* (x-small style, `FlatFarmDataset`). The curriculum = the
same shards, but *ratio-controlled over time*: easy-heavy → hard-heavy → easy-heavy with periodic
easy reviews. Same data, different scheduling — switchable via `SHARD_DIRS` vs `FARM_DIR`.

### Concrete Example: One Week of Training (40,000 steps, ~5,700 steps/day)

> All four mechanisms active simultaneously — every batch mixes all 3 tiers, only the
> percentages move. Data is consumed once, never replayed (40K steps ≈ 2.6B tokens ≈ 6% of corpus).

```
Mon          Wed         ★Thu         Fri          Sun
|←──── FOLD 1: easy→hard ────→|←── FOLD 2: hard→easy ──→|
```

| Day | Steps | Fold | Batch mix trend | Cyclic Scheduling wave (G2) |
|-----|-------|------|-----------------|------------------------------|
| Mon | 0–5,700 | 1 | easy 30%→26%, hard 25%→31% | peak ~Mon night |
| Tue | 5,700–11,400 | 1 | easy 26%→23%, hard 31%→37% | peak Tue |
| Wed | 11,400–17,100 | 1 | easy 23%→20%, hard 37%→43% | peak Wed |
| **Thu (step 20,000)** | | **★ FOLD POINT** | **HARD PEAK: easy 17%, hard 48%** | |
| Thu–Fri | 17,100–22,800 | 2 | easy 20%→23%, hard 43%→37% | peak Thu night |
| Fri–Sat | 22,800–28,500 | 2 | easy 23%→26%, hard 37%→31% | peak Sat |
| Sun | 28,500–40,000 | 2 | easy 26%→**30%**, hard 31%→**25%** | peak Sun |

**All four genes during the week:**
- **Boundary Sharpening (G1)** — the 2-fold arc: fold 1 climbs easy→hard (Mon→Wed), fold 2 mirrors back (Thu→Sun)
- **Curriculum Continuity (G3)** — every ~7.7h (2,000 steps) the mix re-glides → 20 smooth steps of change, no jumps
- **Cyclic Scheduling (G2)** — 8 review waves (one per 5,000 steps ≈ every 19h): easy gets a temporary +12% boost at each peak, then renormalized
- **Local Diversity (G4)** — at every G3 rebuild, example order reshuffles in 5,000-item windows

**What a batch looks like at 3 moments:**
- Mon 9am (step 500): 30% easy + 45% med + 25% hard
- Thu 2pm (step 20,000): 17% easy + 35% med + 48% hard ← hardest week moment
- Sun 9pm (step 39,500): 30% easy + 45% med + 25% hard (Monday's mix, different examples)

---

## Pretraining Configuration (Current)

| Parameter | Value |
|-----------|-------|
| Total tokens ready | ~78 GB across 5 domains (per-domain easy/medium/hard tiered shards) |
| Sequence length | 2048 tokens |
| Tokenizer | `HuggingFaceTB/SmolLM2-135M` (BPE, uint16 output) |
| Shard format | `.bin` uint16 arrays, per tier (see Data Inventory) |
| Batch size | 4 × grad-accum 8 = effective 32 (65,536 tok/step) — batch 4 fits only with fused CE (~9.9 GB peak) |
| Precision | `bfloat16` |
| Optimizer | **CautiousAdamW** (`--cautious`, arXiv:2411.16085) — fused `torch._foreach_*` moments; AdamW-compatible state (resumes old checkpoints losslessly, hot-swappable mid-run). LR 3e-4 → cosine → 1e-6. MuonClip was removed after plateauing at loss ~5.0 |
| Betas | (0.9, 0.95) |
| Weight decay | 0.1 |
| Gradient clipping | max_norm = 1.0 |
| Compilation | Disabled (`torch.compile = False`) |
| Gradient checkpointing | Enabled |
| CPU offloading | Disabled — **pure GPU** (bf16 weights + bf16 AdamW states on GPU). CPUMasterModel offload was the previous approach. |
| Attention | Flash Attention via `F.scaled_dot_product_attention` |
| Memory config | `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (kills fragmentation) |
| Checkpointing | Every 1000 steps (or 20 min) → `megatrain_latest.pt` + `megatrain_best.pt`; plus lean SWA snapshots (`swa_tail/swa_<step>.pt`, model-only bf16 ~2.1 GB each, last 24 kept via `--swa-window 24`) |
| Resume rule | **ALWAYS `megatrain_latest.pt`, NEVER best** — auto-default in code since 2026-08-11 (explicit `--resume-from` still wins; `--init-from` = intentional weights-only warm start) |
| Fused kernels | Liger (`--liger`: RMSNorm/SwiGLU/RoPE) + fused linear+CE (`--fused-ce`: no [B,S,V] logits materialization, frees ~1.9 GB) |
| Power limit | 100W (user live-tunes 200W→180W→100W; ~14 s/step @ 100W vs 9.5 s @ 180W, cooler) |

### Pretraining Scripts

```bash
cd /home/kenpeter/work/small
source venv/bin/activate
bash run_pretrain_gpu_60k.sh     # CURRENT: 60K warm-restart cycle (batch 4 × accum 8, --curriculum --liger --cautious --fused-ce --swa-window 24)
bash run_pretrain_gpu.sh         # original 30K pure-GPU entry (kept for reference)
# bash run_pretrain.sh           # legacy: CPUMaster offload (kept for reference)
```

**Current training state:** ▶️ **RUNNING 60K cycle** — resumed 2026-08-11 21:39 from `megatrain_latest.pt` at **step 36,586/60,000**, best loss **1.7101**, 100W, ~14 s/step (~4,560 tok/s), ~9.9 GB VRAM. First SWA snapshot lands at step 37,000. ETA ~Aug 12 ~10:25.

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
│  Script: pretrain_gpu.py → megatrain_best.pt (pure-GPU, 2.1× faster)    │
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
| **1B model on 12 GB VRAM** | Pure-GPU fit (10.2 GB peak): bf16 weights + bf16 AdamW states on GPU, grad checkpointing, `expandable_segments:True`, checkpoint loaded on CPU then freed, chunked fp32 loss (no full-logits spike). CPUMasterModel offload was the previous approach (~5.1 GB active) but is 2.1× slower. |
| **~78 GB tiered data (not 15B)** | Best-of-best filtered data across 5 domains. Quality over quantity. |
| **Transformer++ architecture** | SOTA for <1B parameters (SmolLM2, Qwen3, Llama 3.2). |
| **AdamW optimizer** | Stable and proven on 12GB VRAM. MuonClip caused a plateau at loss ~5.0 and was removed. |
| **QK-Clip + momentum warmup** | Stabilizes MuonClip on fresh init. Removed with MuonClip. |
| **Batch=8, accum=4** | Effective batch 32 (65K tok/step). |
| **Only 2 checkpoints** | Disk space conservation. `megatrain_latest.pt` for resume, `megatrain_best.pt` for downstream. |
| **Flat farm data org (x-small style)** | `_shards_final` uniform random interleave, sequential consumption. Curriculum sampler (Boundary Sharpening/Cyclic Scheduling/Continuity/Diversity) now ACTIVE on the pure-GPU run. |
| **13-gram dedup** | Exact hash collision drop. 5–10% token savings. Disabled at startup to avoid long scan; toggled via flag. |
| **Direct wget downloads** | Bypasses HF API / Xet throttling. 3–8× faster than hf_hub_download for large parquet files. |
| **Pure-GPU training (2.1× faster)** | Batch-16 experiment found the bottleneck was single-thread CPU offload, not batch size → moved bf16 AdamW states to GPU → 4,800 tok/s vs 2,100 (CPUMaster). Warm-start from CPUMaster checkpoint preserved the overnight run. |
| **Power limit 200W** | Default power limit; pure-GPU runs 75–78°C vs 86°C at 180W CPUMaster. Persistence mode enabled. |

---

## Hardware Requirements

| Phase | GPU VRAM | RAM | Disk | Time Estimate |
|-------|----------|-----|------|---------------|
| Pretraining 1B @ ~78 GB tokens | ~10.2 GB of 12 GB peak (RTX 4070 Ti, pure-GPU) | 93 GB | ~7.9 GB shards | ~2.4 days (15K steps @ 4,800 tok/s) |
| SFT | 8 GB | 8 GB | +24 GB | ~4 hours |
| DPO | 8 GB | 8 GB | +2 GB | ~2 hours |

---

## Files in This Directory

| File | Purpose |
|------|---------|
| `pretrain_megatrain.py` | Legacy pretraining script (CPUMaster CPU-offload + AdamW) — replaced by pretrain_gpu.py |
| `run_pretrain.sh` | Legacy wrapper for pretrain_megatrain.py (CPUMaster) |
| `pretrain_gpu.py` | **Current** pure-GPU 1B trainer: bf16 AdamW on GPU, grad checkpointing, CPU-side checkpoint load, warm-start support |
| `run_pretrain_gpu.sh` | Pure-GPU 30K wrapper (kept for reference): batch 2 × accum 16, warm-start from megatrain_latest.pt → train_small.log |
| `run_pretrain_gpu_60k.sh` | **Current** wrapper: 60K warm-restart (batch 4 × accum 8, `--curriculum --liger --cautious --fused-ce --swa-window 24`), resumes from megatrain_latest.pt |
| `swa_average.py` | SWA finalizer: averages `swa_tail/` lean snapshots → `megatrain_swa.pt` (flat-basin weights, zero training cost) |
| `download_3workers_direct.py` | 3-worker wget downloader with resume for raw datasets |
| `download_qrp_par.py` | Adaptive parallel wget downloader for QuRatedPajama (resumable) |
| `tokenize_domain_parallel.py` | Generic tiered tokenizer (math/code/synth/web → easy/medium/hard .bin shards) |
| `tokenize_reformat.py` | Tokenizer for reformatted textbook+QA JSONL → reformat_easy shards |
| `reformat_data.py` | Kimi K2-style data reformatting via vLLM chat API |
| `quick_eval_pretrain.py` | Quick eval of ANY 1B checkpoint (generation + perplexity) — `--ckpt PATH` (default megatrain_latest.pt) |
| `check_status.sh` | Combined status report: training + downloads + disk |
| `test_pretrain.py` | Test suite (63 tests — run `./venv/bin/python test_pretrain.py`; MUST pass before any training change) |
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
| **MuonClip → AdamW (reverted)**: Muon plateaued at loss ~5.0. Switched back to `torch.optim.AdamW` (3e-4 → cosine → 1e-6) with the G1–G4 curriculum (Boundary Sharpening · Cyclic Scheduling · Curriculum Continuity · Local Diversity).
- `non_blocking=False` fix in `cpu_master.py` (race condition on D2H copies).

### Pure-GPU Transition (2026-08-04)
- Batch-16 experiment measured SLOWER (>7.8s/step vs 6.8s baseline) → bottleneck was single-thread CPU offload, not batch size
- Built `pretrain_gpu.py`: bf16 weights + bf16 AdamW states on GPU, grad checkpointing, warm-start from CPUMaster checkpoint
- **OOM bug found & fixed**: `--init-from` loaded the 6.2 GB checkpoint onto the GPU and never freed it → fixed with `map_location="cpu"` + `del ckpt, sd` after `load_state_dict`
- **OOM bug #2 (loss spike)**: run still crashed at first forward pass — HF's internal loss does `logits.float()` on the full [2×2048×49152] tensor (+768 MB). Replaced with **chunked fp32 cross-entropy** (512-token slices, identical math, ~100 MB per chunk) → peak **10.2 GB**, 1.4 GB headroom, zero coin-flip. Same 4,800 tok/s.
- Result: **4,800 tok/s vs 2,100 (2.1×)** → ETA ~2.4 days instead of 4.7
- Warm-started from step 9,000 checkpoint (loss 2.25-era) → new run continues at step 400+, not from zero
- Ops hardening: run launched **detached** (PPID 1/systemd) after a session interruption SIGTERM'd the tracked process; watchdog updated for `pretrain_gpu.py` + re-enabled (5-min auto-restart); 30-min progress cron reports step/loss

### Disk Space Management
- 506 GB total. 151 GB used, 330 GB free.
- Data prep creates ~280 GB intermediate files — always clean up staging/filtered dirs after each run.
- Checkpoint rules: keep only latest + best per phase (~12 GB total for 1B model).

---

## Current State (2026-08-04)

- **Training:** ⏸️ **PAUSED** at ~step 1,890 / 15,000 (frozen 23:15, SIGSTOP). Loss 2.8704 @ step 1600, LR 2.99e-04, 15.57s/step, 4,210 tok/s, GPU 10.2GB peak. See **Pause & Resume** below.
- **Data:** all 5 domains downloaded and tokenized into per-domain easy/medium/hard tiered shards
- **Reformat:** textbook + QA reformatted via Qwen3-8B (14 MB, reformat_easy tier)
- **Ops:** training runs detached (survives session resets) + 5-min watchdog + 30-min progress cron — **both cron jobs PAUSED with training**
- **Power limit:** 100W (user's final choice; card is memory-bound, throughput identical at 100–250W)
- **Next:** resume → reach loss 2.7 → SFT → DPO

---

## Pause & Resume (2026-08-04)

> Training was paused at the user's request ("Save everything. And pause"). The process was
> **frozen with SIGSTOP, not killed** — the clock is halted mid-run, nothing was lost.

### Frozen state (if process still alive)

| Item | Value |
|------|-------|
| Process | pid `980795` (STAT `Tl` = stopped), started 15:30 |
| Frozen at | ~step 1,890 / 15,000 |
| Log | `/home/kenpeter/work/train_small.log` |
| VRAM while frozen | 11.5 GB held (GPU otherwise idle, ~10W) |

**Resume (zero loss):**
```bash
kill -CONT 980795        # unfreeze — training continues exactly where it stopped
```
Verify: `ps -o pid,stat -p 980795` shows `Rl`, then `tail -f /home/kenpeter/work/train_small.log`.
Checkpoint saves resume on schedule (every 1,000 steps → next at step 2,000).

**If the frozen process died** (reboot/kill): relaunch the exact command — it auto-resumes
from `megatrain_latest.pt` (currently **step 1,000**, saved 19:24; best 2.9666). This loses
~890 steps (~3.9h) of post-checkpoint progress:
```bash
cd /home/kenpeter/work/small
source venv/bin/activate
bash run_pretrain_gpu.sh        # batch 2 × accum 16, 15K steps, warm-start, log → train_small.log
```
(`run_pretrain_gpu.sh` preflight aborts if a training process is already running.)

### Cron jobs to re-enable after resume

Both were paused 2026-08-04 23:17 to prevent the watchdog from auto-restarting (or the
progress cron from reporting 🔴 DOWN) while paused:

| Job | ID | Schedule | Purpose |
|-----|----|----------|---------|
| pretrain-watchdog | `e25e7b032689` | every 5m | auto-restart if training dies (writes `/tmp/pretrain_restarted.flag`) |
| training-progress | `6fca7f20fe2b` | every 30m | one-line step/loss/ETA report to Telegram |

Re-enable with `cronjob action=resume` on both IDs.

### Notes

- **No signal handler in `pretrain_gpu.py`** — it saves ONLY at `step % 1000 == 0`. A hard
  kill mid-interval always costs up to 1,000 steps of progress. If this matters, add a
  `signal.signal(SIGTERM, ...)` handler that saves the current state before exiting.
- **Power limit:** 100W (persists until reboot; re-apply with `sudo nvidia-smi -pl 100`).
- Progress milestone for next stage: loss **2.7** → then SFT (data ready in `_sft_final_shards/`).

---

## Failure Prevention (Ops Rules — learned 2026-08-04)

| Failure | Prevention (enforced) |
|---------|----------------------|
| **Process killed by session reset** | Launch detached (PPID 1/systemd) via `start_new_session=True`; 5-min watchdog auto-restarts; 30-min cron reports |
| **OOM: checkpoint on GPU** | `--init-from` must `map_location="cpu"` + `del ckpt` after `load_state_dict` |
| **OOM: HF full-logits fp32 loss spike (+768MB)** | Chunked fp32 cross-entropy (512-token slices) — peak 10.2GB, 1.4GB headroom |
| **OOM: double-launch / no headroom** | `run_pretrain_gpu.sh` preflight: abort if process already running, <2GB free VRAM, <10GB disk, or checkpoint missing |
| **OOM: "it fit" fallacy** | Smoke test MUST use exact real-run flags (incl. `--init-from`); fresh-init measure runs are NOT representative of warm-start |
| **Secret in repo (HF token)** | Pre-commit hook `scripts/scan_secrets.sh` (installed at `.git/hooks/pre-commit`) blocks token patterns; GitHub push protection as backstop; tokens live in `~/.huggingface/token` (chmod 600) or env vars |
| **Silent death** | Watchdog writes `/tmp/pretrain_restarted.flag`; 30-min cron reports 🔴 DOWN + crash reason |

**Golden rule:** keep ≥1GB VRAM headroom. "It fits" at 100MB margin is luck, not stability.

---

## Current State (2026-08-11)

- **Training:** ▶️ RUNNING 60K cycle @ **100W** (user live-tunes: 200W→180W→100W — cooler, slower OK). Resumed 21:39 from `megatrain_latest.pt` → **step 36,586/60,000**, best loss **1.7101** (Aug 9 01:47). Batch 4 × accum 8 (eff 32), seq 2048, LR 3e-4→1e-6 cosine, `--curriculum --liger --cautious --fused-ce --swa-window 24`. ~14 s/step, ~9.9 GB VRAM.
- **Curriculum:** fold 2 (hard→easy mirror) active since step 30K; ratios re-glide every 2000 steps (G1–G4).
- **SWA tail collection LIVE:** lean model-only snapshots (bf16, ~2.1 GB) in `checkpoints/swa_tail/` every 1000 steps, window 24 ≈ 50 GB (disk 221 GB free). At run end: `swa_average.py` → `megatrain_swa.pt`, then quick-eval vs `megatrain_best.pt` (auto-finalize wired into the 30-min training-progress cron).
- **Resume rule:** ALWAYS `megatrain_latest.pt`, NEVER best — auto-default in code (explicit path wins).
- **Power-limit footgun:** resets to 285W after reboot/GPU idle — re-apply `sudo nvidia-smi -pl 100` (sudo pwd in short_term.md).
- **x-small 135M:** parked (stopped at step 45,200; watchdog paused).

### Recent History (August 2026)

- **Aug 10 — speed session (3 changes, 57/57 tests):** fused CautiousAdamW moments via `torch._foreach_*` (400→4 launches, bitwise-identical, VRAM-safe); Liger fused CE `--fused-ce` (bitwise-identical to chunked CE, frees ~1.9 GB → batch 4 fits at 9,865 MiB); batch 2→4 × accum 8 (eff 32). Commits cc0ce82 + bdcaae3.
- **Aug 11 — SWA + resume hardening (63/63 tests):** `--swa-window N` lean tail snapshots + `swa_average.py`; auto-resume-latest default (`resolve_resume_path`); post-resume log-stats fix (`compute_log_stats` — first line after resume divided by 100 instead of true step count → fake Loss 0.27, real ~1.98); `quick_eval_pretrain.py --ckpt`. Commits ec8ae4d → 47b5828.
- **Aug 11 — machine powered off 06:44 AEST, back ~21:16** (clean shutdown, not a crash; killed a resume that had run 4 min). Power limit had reset to 285W. Training resumed from latest with zero step loss (05:51 checkpoint untouched).

---

## Summary

- **Tokens ready:** ~78 GB across 5 domains (math, web, code, synth, reformat), tiered easy/medium/hard
- **Training:** ▶️ RUNNING 60K cycle @ 100W — step 36,586/60,000 (resumed Aug 11 21:39), best loss 1.7101, batch 4 × accum 8, CautiousAdamW + Liger fused CE, ~14 s/step, ~9.9 GB VRAM. SWA tail collection live (window 24).
- **SFT assets ready:** 25.76 GB (`_sft_final_shards/`)
- **Next action:** finish 60K → SWA average → eval `megatrain_swa.pt` vs `megatrain_best.pt` → pick final base → SFT → DPO.
