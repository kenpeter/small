# small — 1.03B LLM Pretraining

Custom 1.03B parameter LLM (dim=1536, L=32, h=12, kv=4, FFN=4608) trained from scratch on ~78GB of quality-filtered data across 5 domains.

## Training Status

**Current:** Fresh run (x-small style flat farm) — see train_small.log
**Target:** Loss **2.7** → then SFT → DPO
**Optimizer:** AdamW (3e-4 → cosine → 1e-6)
**Dtype:** bfloat16
**Hardware:** RTX 4070 Ti 12GB + CPUMasterModel (CPU offloaded)
**Speed:** ~2400 tok/s, 6.8s/step @ 180W

## Curriculum

**Active — x-small style flat farm:** `_shards_final/` (1098-bin symlink farm, uniform random interleave of all tiered shards, ~42B tokens), consumed sequentially via `FlatFarmDataset` (md5-hash split: 1088 train / 10 val; per-shard hash offset + paired reversal). Static order — no ratio rebuilds.

**Legacy (kept in code, dormant):** G1–G4 curriculum (SAW-style): smooth easy→hard tier blend with cyclic easy-review wave and JIT windowed shuffle. Ratios rebuilt every 2000 steps.

| Tier | Ratio (start → end) | Domains |
|------|---------------------|---------|
| Easy | 0.30 → 0.05 (review floor) | math, web, synth, code |
| Medium | fills the hump | math, web, synth, code |
| Hard | 0.25 → 0.70 | math, synth, code (web_hard now LIVE: QuRatedPajama) |

## Data Domains

- **Math:** fine-math-3plus + open-web-math (48.4 GB)
- **Web:** fineweb-edu (26.3 GB)
- **Code:** github-code (742 MB)
- **Synth:** cosmopedia (3.4 GB)
- **Reformat:** textbook + QA (14 MB)

## Pipeline

```
download_3workers_direct.py → _raw_original/*.parquet
  tokenize_domain_parallel.py → _shards_{math,web,synth,code}_{easy,medium,hard}/.bin
  tokenize_reformat.py       → _shards_reformat_easy/.bin
  symlink farm (one-off)     → _shards_final/00000..01097.bin (uniform random mix)
pretrain_megatrain.py        → megatrain_latest.pt / megatrain_best.pt
quick_eval_pretrain.py       → generation + perplexity
```

- `run_download.sh` / `run_pretrain.sh` — wrappers for download / training
- `download_qrp_par.py` + `watchdog_qrp.sh` — QuRatedPajama web-hard source (adaptive parallel wget)
- `check_status.sh` — combined status report (training + downloads + disk)
- `test_pretrain.py` — test suite for pretrain_megatrain.py

## Key Changes

- Switched from KimiMuonClip → torch.optim.AdamW (Muon caused plateau at loss ~5.0)
- G1–G4 curriculum replaces hard phase switching — ratios shift smoothly every 2000 steps
- Data org: x-small style flat farm (`_shards_final` via `FlatFarmDataset`) — G1–G4 stratified sampling superseded (dormant)
- Per-Head Muon (Kimi K3 §2.5) was removed — too aggressive for 12-head model
