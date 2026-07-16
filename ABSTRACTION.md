# Training Abstractions & Frontier Findings

## 1. Architecture Baseline
We replicate the **SmolLM2-135M** architecture exactly:
- 30 layers, dim=576, FFN=1536
- GQA: 9 query / 3 KV heads
- SmolLM2 tokenizer (vocab 49152)
- RoPE θ=10,000, RMSNorm, SwiGLU
- **Our params**: 162.8M (vs SmolLM2's 135M due to full vocab size)

## 2. Frontier Comparison (L20-Edu-135M)
Source: *"L20-Edu-135M: An Auditable Single-GPU Study of Data-Efficient Small Language Modeling"* (arXiv:2606.22189, June 2026).

### 2.1 Architecture Match
L20 uses the **exact same** layer/hidden/FFN/GQA/tokenizer configuration. This paper is the closest published analog to our model.

### 2.2 Training Hyperparameters
| Setting | L20-Edu-135M | Ours (Before) | Ours (After) |
|---|---|---|---|
| Peak LR | **4.0e-4** | 2.0e-4 | **4.0e-4** |
| Optimizer | AdamW | AdamW | AdamW |
| Schedule | warmup + cosine | warmup + cosine | warmup + cosine |
| Tokens/step | **528,384** | 98,304 | **196,608** |
| `torch.compile` | **YES** | NO | **YES** |
| Gradient checkpointing | YES | YES | YES |
| Context | 2048 | 4096→2048 | 2048 |

**Action taken**: bumped LR to 4e-4, enabled `torch.compile`, doubled grad_accum to 24 (effective batch 96), fixed max_steps to 90k for 5 epochs.

### 2.3 Data Strategy
- **L20** trains on **13B total tokens** (10B base + 3B continued pretraining).
- **SmolLM-135M** trains on ~600B tokens.
- **SmolLM2-135M** trains on ~2T tokens.
- **Ours** trains on **3.53B tokens**.

L20 reaches **val loss 2.87** after 18,928 steps on 10B tokens. With only 3.53B tokens, our realistic target is **~3.0–3.2**.

#### Quality vs Quantity
L20 rejects **70%** of candidate documents. Our pipeline rejects **~90%**. Our 3.53B is **best-of-best**: heavily deduped, filtered for alpha ratio, unique word counts, and length. While L20 has 3.7× more total tokens, our per-token quality is likely higher. Still, small models are notoriously data-hungry; 3.5B is the lower bound for competent performance.

#### L20 Stage-4 Mixture (3B continued pretrain)
| Source | Ratio | Tokens |
|---|---|---|
| FineWeb-Edu score 3+ | 40.0% | 1.2B |
| DCLM educational shards | 30.0% | 900M |
| High-quality edu replay | 10.0% | 300M |
| Dolmino PES2O | 7.0% | 210M |
| Dolmino Wikipedia | 5.0% | 150M |
| Dolmino StackExchange | 3.0% | 90M |
| FineMath score 4+ | 3.0% | 90M |
| MixtureVitae tutorials / reasoning / math word | ~2.0% | ~60M |

**Gaps in our data**: No DCLM, no Dolmino sources, no Wikipedia, no StackExchange. Our mixture is primarily FineWeb-Edu + FineMath + some SlimPajama/code. Adding Wikipedia and StackExchange would improve factual/Q&A capability.

## 3. Current Pipeline Status
| Phase | Status | Data |
|---|---|---|
| Pretraining | **Running** (step ~16,200, loss ~3.28) | 13 shards, 3.53B tokens |
| SFT | Paused at step 100, loss 1.23 | 6.78M samples |
| DPO | **Not started** | 234K pairs (raw, not tokenized) |

## 4. Key Constraints
- **GPU**: RTX 4070 Ti 12GB
- **Batch size**: 4 is VRAM ceiling at seq_len=2048 with gradient checkpointing
- **Speed**: ~20k tok/s (before compile/accum changes)
- **No torch.compile** previously due to caution; now enabled for free 15–25% speedup

## 5. Lessons Learned
1. **Loss plateau at 3.3–3.4** was caused by tiny effective batch (8→24→48), not data corruption. Bumping grad_accum immediately broke the plateau.
2. **Gradient checkpointing** is mandatory for batch>1 on 12GB with 30-layer model.
3. **Causal attention bias buffers** were wasting ~8GB; removing them saved enough VRAM to increase batch size.
4. **max_steps must be recalculated** whenever batch/seq_len/accum changes. 539k steps at old config became ~15 epochs at new config.
5. **L20's LR (4e-4)** is 2× ours. With our small batch, 2e-4 was overly conservative.
