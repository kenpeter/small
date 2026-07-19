# Project Memory

## Disk Space Management (Critical)

**506 GB total** on `/dev/nvme0n1p5`. Never fill it up — training/eval will crash.

### Data Pipeline Rules
- **Original data**: `_cot_raw/` (keep)
- **Final data**: `_shards_final/`, `_sft_shards/` (keep)
- **Intermediate data**: `_staging_*`, `_filtered_*`, `_dpo_staging/`, `_sft_staging/`, `_cache/` — remove after processing
- Data prep creates ~280G of intermediate files. Always clean up staging/filtered dirs after each pipeline run.

### Checkpoint Rules (keep only latest + best per phase)
- Pretraining: `checkpoint_latest.pt` + `checkpoint_best.pt`
- SFT: `sft_latest.pt` + `sft_best.pt`
- DPO: `dpo_latest.pt` + `dpo_best.pt`
- Delete old `pretrained_*`, `sft_final`, `dpo_final` duplicates when no longer needed.
- Each checkpoint = 1.9 GB. Don't keep 3 copies per phase.
