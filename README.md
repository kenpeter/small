# small — 1.03B LLM Pretraining

Custom 1.03B parameter LLM (dim=1536, L=32, h=12, kv=4, FFN=4608) trained from scratch on ~78GB of quality-filtered data across 5 domains.

## Training Status

**Current:** Step 8800/60000 — **Loss 2.77** (best: 2.77)
**Target:** Loss **2.7** → then Medium → Hard curriculum phases
**Optimizer:** AdamW (3e-4 → cosine → 1e-6)
**Dtype:** bfloat16
**Hardware:** RTX 4070 Ti 12GB + CPUMasterModel (CPU offloaded)
**Speed:** ~2400 tok/s, 6.8s/step @ 180W

## Curriculum

| Phase | Steps | Mix |
|-------|-------|-----|
| Easy+Medium | 0-60000 | math(30%), code(15%), synth(25%), web(30%) — 40% easy / 60% medium |
| Medium | (next) | TBD |
| Hard | (next) | TBD |

## Data Domains

- **Math:** fine-math-3plus + open-web-math (48.4 GB)
- **Web:** fineweb-edu (26.3 GB)
- **Code:** github-code (742 MB)
- **Synth:** cosmopedia (3.4 GB)
- **Reformat:** textbook + QA (14 MB)

## Key Changes

- Switched from KimiMuonClip → torch.optim.AdamW (Muon caused plateau at loss ~5.0)
- Mixed easy + medium data provides harder signal without destabilizing
- Per-Head Muon (Kimi K3 §2.5) was removed — too aggressive for 12-head model
