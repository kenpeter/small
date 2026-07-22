# Data Abstraction — Training Assets on Machine

Generated: 2026-07-22  
Purpose: Single source of truth for what data exists, where it lives, and how it feeds into the pipeline.

---

## 1. Pretraining Tokens (Ready to Feed)

| Directory | Files | Size | Tokens | Domain | Status |
|---|---|---|---|---|---|
| `_shards_final/` | 17 `.bin` | 7.94 GB | ~3.97B uint16 | 100% FineWeb-Edu (web text) | ✅ Active |

**Notes:**
- Shard size target: 512 MB (except `shard_000000.bin` at 260 MB, `shard_000016.bin` at 0 B empty)
- Tokenizer: SmolLM2-135M vocab (49152)
- Currently **100% web** — math / synthetic shards do not exist yet

---

## 2. Raw Downloads (Staging → Needs Tokenization)

| Dataset | Files | Size | Total Expected | Domain | Status |
|---|---|---|---|---|---|
| `fineweb-edu/` | 42 `.parquet` | 1.91 GB | ~10 GB | Web / educational | ✅ Complete |
| `finemath-3plus/` | 24 `.parquet` | 4.71 GB | ~28 GB | Math (grade 3+) | ⬇️ In Progress (24/128) |
| `cosmopedia/` | — | — | ~8–10 GB | Synthetic / encyclopedic | ❌ Not started |
| `open-web-math/` | — | — | ~7–8 GB | Math / research | ❌ Not started |

**Notes:**
- Download worker: `download_3workers_direct.py` (3 workers, 1.5 s stagger)
- Missing datasets block true 60/25/15 stratified mix

---

## 3. SFT / Instruction Data (Post-Pretraining)

| Directory | Files | Size | Sources | Status |
|---|---|---|---|---|
| `_sft_final_shards/` | 71 `.pt` | 25.76 GB | Alpaca-GPT4, Code-Alpaca, OpenHermes, etc. | ✅ Ready for SFT stage |

**Notes:**
- Pre-tokenized `.pt` shards (not `.bin`)
- Consumed after base model pretraining finishes

---

## 4. Math / Code Raw Datasets (LeetCode Cluster)

| Dataset | Size | Format | Quality | Notes |
|---|---|---|---|---|
| `LeetCode_YT_CC_CoT_Summary/` | 0.67 GB | mixed | Medium | YouTube + CoT summaries |
| `newfacade_LeetCodeDataset/` | 0.10 GB | `.jsonl` | Medium | Train + test split |
| `high_quality_leetcode/` | 0.05 GB | `.jsonl` | Medium | Filtered subset |
| `greengerong_LeetCode/` | 0.02 GB | `.jsonl` | Low-Med | Java/Python solutions |
| `DenCT_LeetCode/` | 0.01 GB | `.parquet` | Low | Tiny |
| `LimYeri_LeetCode/` | 0.01 GB | `.parquet` | Low | Tiny |
| `NanDo_LeetCodeContests/` | ~0 GB | `.parquet` | Low | Tiny |
| `juyoungml_LeetCodeRosetta/` | ~0 GB | `.parquet` | Low | Rosetta-style |
| `mesolitica_LeetCodeQwQ/` | 0.02 GB | `.parquet` | Low | QwQ-style reasoning |
| `vovw_LeetCode/` | ~0 GB | `.parquet` | Low | Tiny |

**Verdict:** ~0.88 GB total. Small, heterogeneous, mostly LeetCode solutions. **Not currently used in pretraining.** Could be filtered + tokenized into a "code" domain bucket if desired, but `finemath-3plus` + `open-web-math` are the priority for math.

---

## 5. Other / Auxiliary

| Dataset | Size | Format | Purpose | Status |
|---|---|---|---|---|
| `_cot_raw/numina_50k.jsonl` | 0.07 GB | `.jsonl` | Chain-of-Thought reasoning | Hold for SFT or synthetic mix |
| `bluemoon_roleplay/` | 0.27 GB | `.json` + `.arrow` | Roleplay / conversational | Hold for SFT |

---

## 6. Pipeline Mapping

```
┌─────────────────────────────────────────────────────────────┐
│  RAW STAGING (_staging_multi/)                               │
│    fineweb-edu      ──┐                                     │
│    finemath-3plus   ──┼──► tokenize.py ──► _shards_*/     │
│    cosmopedia       ──┘      (SmolLM2 vocab)   .bin       │
│    open-web-math                                           │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  PRETRAIN (pretrain_megatrain.py)                            │
│    _shards_final/   ──► StratifiedBinShardDataset           │
│    _shards_math/    ──► (60/25/15 web/math/synth)          │
│    _shards_synth/   ──► dedup=True (disabled at startup)    │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  SFT / RLHF                                                  │
│    _sft_final_shards/  ──► supervised fine-tuning           │
│    _cot_raw/           ──► reasoning boost                   │
│    bluemoon_roleplay/  ──► conversational style              │
└─────────────────────────────────────────────────────────────┘
```

---

## 7. Current Blockers

| Blocker | Impact | Fix |
|---|---|---|
| `finemath-3plus` download incomplete | Cannot build math shards | Wait for 128/128 files |
| `cosmopedia` missing | Cannot build synthetic shards | Queue download after finemath |
| `open-web-math` missing | Math diversity gap | Queue download after cosmopedia |
| `_shards_math/` does not exist | Stratified loader falls back to web | Tokenize finemath → _shards_math/ |
| `_shards_synth/` does not exist | Stratified loader falls back to web | Tokenize cosmopedia → _shards_synth/ |

---

## 8. Summary

- **Tokens ready now:** ~3.97B (web only)
- **Tokens needed for 60/25/15 mix:** ~6.6B web + ~2.75B math + ~1.65B synthetic = **~11B total**
- **SFT assets ready:** 25.76 GB
- **Next action:** Complete downloads → tokenize math + synth → update `SHARD_DIRS` → restart training with true stratified mix.
