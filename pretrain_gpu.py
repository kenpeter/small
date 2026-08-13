#!/usr/bin/env python3
"""
Pure-GPU pretraining for the 1B model — replaces CPUMaster offload.

Removes the serial CPU<->GPU layer-streaming bottleneck (measured: 1 hot
CPU thread, ~2.3K tok/s). Model + AdamW moments live on GPU in bf16
(12GB VRAM budget: 2.1GB params + 4.1GB moments + 2.1GB grads + ~1GB
activations with gradient checkpointing). Effective batch: 2 × accum 16
× seq 2048 = 65K tokens/step.

Loss is computed with chunked fp32 cross-entropy (512-token slices) to
avoid HF's full-logits fp32 spike — peak GPU ~10.3GB, under 11.59GB.

Speed features: fused AdamW, flash SDPA (no mask), GPU loss accumulation,
async checkpoints, optional Liger fused kernels (--liger) and torch.compile
(--compile, with OOM→eager auto-fallback).

Supports --init-from warm-start (checkpoint loaded on CPU, freed after
load_state_dict — the OOM fix) and --resume-from true-resume (model +
optimizer + step + best_loss restored, so the LR/curriculum schedule
continues and best.pt is never clobbered by a fresh best_loss=inf).
"""
import argparse, logging, math, os, threading, time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaConfig

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except ImportError:
    _HAS_TRITON = False

from pretrain_megatrain import (
    FARM_DIR,
    SHARD_DIRS,
    FlatFarmDataset,
    StratifiedShardDataset,
    collate_pretrain,
    get_lr,
    get_curriculum_ratios,
    CURRICULUM_UPDATE_INTERVAL,
    save_checkpoint_robust,
    should_save_checkpoint,
)

logger = logging.getLogger("pretrain_gpu")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

SEQ_LEN = 2048
CE_CHUNK = 512           # fp32 CE slice size. NOT 2048: the full-seq slice's
                         # fp32 transient (2047×49152×4 = 805MB) OOMs the 12GB
                         # card on allocator fragmentation (measured 2026-08-05,
                         # step ~570, twice). 512 keeps the transient at 201MB.
WARMUP_DEFAULT = 400     # #4: 1000-step re-warmup wastes ~2h on warm-starts
OUTPUT_DIR = Path("/home/kenpeter/work/checkpoints")
LATEST = OUTPUT_DIR / "megatrain_latest.pt"
BEST = OUTPUT_DIR / "megatrain_best.pt"


def chunked_ce(logits, labels, chunk_size=CE_CHUNK):
    """fp32 cross-entropy computed in sequence slices.

    Identical math to one big call (sum of parts = whole), but the fp32
    logits conversion peaks at chunk_size×V×4 bytes instead of S×V×4
    (~805MB) — keeps the run under the 11.59GB ceiling. CE_CHUNK=512 is the
    proven-safe size (2048's 805MB transient OOMs on fragmentation).
    """
    B, S, V = logits.shape
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    n_tok = shift_labels.numel()
    loss = None
    for i in range(0, S - 1, chunk_size):
        ce = torch.nn.functional.cross_entropy(
            shift_logits[:, i : i + chunk_size, :].float().reshape(-1, V),
            shift_labels[:, i : i + chunk_size].reshape(-1),
            reduction="sum",
            ignore_index=-100,
        )
        loss = ce if loss is None else loss + ce
    return loss / n_tok


_fused_ce_fn = None


def get_fused_ce_fn():
    """Lazy singleton for the Liger fused linear+CE loss (needs CUDA)."""
    global _fused_ce_fn
    if _fused_ce_fn is None:
        from liger_kernel.transformers.fused_linear_cross_entropy import (
            LigerFusedLinearCrossEntropyLoss,
        )
        _fused_ce_fn = LigerFusedLinearCrossEntropyLoss(
            ignore_index=-100, reduction="sum"
        )
    return _fused_ce_fn


def fused_ce(hidden_states, lm_head_weight, labels):
    """Liger fused linear+CE — the loss WITHOUT materializing the [B,S,V]
    logits tensor (~805MB at batch 4) or fp32 chunk slices (~201MB each).

    This is what unlocks --batch-size 4 on the 12GB card: the ~1.2GB the
    chunked path holds as logits+chunk transients is freed for activations.

    Shift semantics are IDENTICAL to chunked_ce: predict the next token
    (hidden[t] against labels[t+1]), mean over ALL tokens (incl. ignored,
    matching chunked_ce's sum/n_tok — no padding in packed data anyway).
    Liger's kernel wants flat inputs: hidden [B*S, H], labels [B*S].
    """
    fn = get_fused_ce_fn()
    B, S, H = hidden_states.shape
    h = hidden_states[:, :-1].reshape(-1, H).contiguous()
    tgt = labels[:, 1:].reshape(-1).contiguous()
    return fn(lm_head_weight, h, tgt) / tgt.numel()


def apply_liger_if_requested(enable: bool):
    """#2 — swap Llama's RMSNorm/MLP/RoPE for Liger fused Triton kernels.

    MUST run before the model is built (it monkey-patches the transformers
    module). State-dict key names are unchanged, so --init-from warm-start
    stays compatible (strict=False). CE replacements are NOT enabled: the
    training loop computes its own chunked CE.
    """
    if not enable:
        return
    try:
        from liger_kernel.transformers import apply_liger_kernel_to_llama
    except ImportError:
        logger.warning("liger-kernel not installed — continuing with vanilla ops")
        return
    apply_liger_kernel_to_llama(
        rope=True, rms_norm=True, swiglu=True,
        cross_entropy=False, fused_linear_cross_entropy=False,
    )
    logger.info("⚡ Liger fused kernels ON (RMSNorm/SwiGLU/RoPE → Triton)")


def _unwrap_compiled(model):
    """Return the underlying eager module if model is torch.compile-wrapped
    (compile stores the original as _orig_mod)."""
    return getattr(model, "_orig_mod", model)


def _compile_warmup(model, dl, device):
    """One real-shape forward+backward to trigger inductor JIT (both graphs),
    so any OOM during compilation surfaces here — where main() can fall back
    to eager — instead of mid-training."""
    data_iter = iter(dl)
    batch = next(data_iter)
    input_ids = batch["input_ids"].to(device, non_blocking=True)
    labels = batch["labels"].to(device, non_blocking=True)
    out = model(input_ids=input_ids)
    loss = chunked_ce(out.logits, labels)
    loss.backward()
    model.zero_grad(set_to_none=True)


def run_microbatches(model, data_iter, dl, device, args):
    """One gradient-accumulation window (16 micro-batches). Returns the GPU
    loss accumulator and the (possibly refreshed) data iterator."""
    acc_loss = torch.zeros((), device=device, dtype=torch.float32)
    for _ in range(args.grad_accum):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dl)
            batch = next(data_iter)
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        # No attention_mask: collate's 4D mask is a pure causal tril(ones)
        # (packed data, zero padding) — bitwise-identical to SDPA's is_causal,
        # so shipping + processing the 8MB mask every micro-batch is pure waste.
        if getattr(args, "fused_ce", False):
            # Liger fused linear+CE: skip lm_head entirely — no [B,S,V] logits
            # tensor (~805MB at batch 4) and no fp32 chunk transients. This is
            # what makes --batch-size 4 fit in 12GB.
            out = model.model(input_ids=input_ids)
            hidden = model.model.norm(out.last_hidden_state)
            loss = fused_ce(hidden, model.lm_head.weight, labels) / args.grad_accum
        else:
            out = model(input_ids=input_ids)
            logits = out.logits  # bf16 [B, S, V] — no labels → HF skips its fp32 loss
            loss = chunked_ce(logits, labels) / args.grad_accum
        acc_loss = accumulate_loss(acc_loss, loss)
        loss.backward()
    return acc_loss, data_iter


def build_model(dtype: torch.dtype) -> torch.nn.Module:
    hf_config = LlamaConfig(
        vocab_size=49152,
        hidden_size=1536,
        intermediate_size=4608,
        num_hidden_layers=32,
        num_attention_heads=12,
        num_key_value_heads=4,
        max_position_embeddings=8192,
        rope_theta=10000.0,
        rms_norm_eps=1e-5,
        hidden_act="silu",
        tie_word_embeddings=False,
        attention_bias=False,
        mlp_bias=False,
        initializer_range=0.02,
        torch_dtype=dtype,
        head_dim=128,
        architectures=["LlamaForCausalLM"],
    )
    model = AutoModelForCausalLM.from_config(
        hf_config, dtype=dtype, trust_remote_code=True, attn_implementation="sdpa"
    )
    model.gradient_checkpointing_enable()
    # transformers 5.5.3: Llama forward has NO gradient-checkpoint dispatch —
    # verified empirically (zero torch.utils.checkpoint.checkpoint calls, Aug 13).
    # The run therefore trains recompute-free; VRAM fits via Liger fused CE +
    # flash SDPA + bf16. If a future transformers upgrade activates gc, watch
    # for a speed regression (recompute) and re-evaluate the VRAM budget.
    logger.info("Gradient checkpointing: INERT in transformers 5.5.3 — run is recompute-free")
    model.config.use_cache = False
    n = sum(p.numel() for p in model.parameters())
    logger.info(f"Model: {n:,} params ({n/1e9:.2f}B), dtype={dtype}")
    return model


class CautiousAdamW(torch.optim.Optimizer):
    """AdamW + cautious mask (arXiv:2411.16085, 'Cautious Optimizers').

    Drop-in replacement for torch.optim.AdamW with IDENTICAL optimizer state
    layout ({'step','exp_avg','exp_avg_sq'} per param) — a checkpoint saved by
    plain AdamW resumes losslessly: m/v are kept, only the step function
    changes. The mask is stateless: an update is applied only where it agrees
    with the gradient's sign (update*g > 0), killing overshoot/oscillation.
    Weight decay is applied unmasked (per the paper).

    Partially fused: the AdamW moments use torch._foreach_* multi-tensor
    kernels (one launch per op instead of one per param — ~100x fewer
    launches, pure in-place, bitwise-identical math). The per-param update
    tail (denom/update/mask) stays elementwise because materializing all
    fp32 temps as lists would cost ~4-8GB extra VRAM on top of the ~10.3GB
    weights+m/v — instant OOM on a 12GB card. Temps stay per-param, so peak
    VRAM is unchanged.
    """
    def __init__(self, params, lr=3e-4, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr = group["lr"]
            eps = group["eps"]
            wd = group["weight_decay"]
            params = [p for p in group["params"] if p.grad is not None]
            if not params:
                continue
            grads = [p.grad for p in params]
            if any(g.is_sparse for g in grads):
                raise RuntimeError("CautiousAdamW does not support sparse gradients")
            states = [self.state[p] for p in params]
            exp_avgs, exp_avg_sqs = [], []
            for p, s in zip(params, states):
                if len(s) == 0:
                    s["step"] = 0
                    s["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    s["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                step = s["step"]
                if isinstance(step, torch.Tensor):
                    step = step.item()
                s["step"] = step + 1
                exp_avgs.append(s["exp_avg"])
                exp_avg_sqs.append(s["exp_avg_sq"])
            # ── Pass 1: AdamW moments — fused multi-tensor kernels ──
            torch._foreach_mul_(exp_avgs, beta1)
            torch._foreach_add_(exp_avgs, grads, alpha=1 - beta1)
            torch._foreach_mul_(exp_avg_sqs, beta2)
            torch._foreach_addcmul_(exp_avg_sqs, grads, grads, value=1 - beta2)
            # ── Pass 2: per-param update tail (per-param temps → VRAM-safe) ──
            # Fused Triton path: 1 launch per param, temp-free, bitwise-tested
            # against the torch loop below. Falls back to the torch loop when
            # triton is unavailable or tensors are non-contiguous/non-CUDA.
            if _HAS_TRITON and all(
                p.is_cuda and p.is_contiguous() and g.is_cuda and g.is_contiguous()
                for p, g in zip(params, grads)
            ):
                _cautious_tail_triton(params, grads, states, lr, wd, eps, beta1, beta2)
            else:
                for p, g, s in zip(params, grads, states):
                    exp_avg, exp_avg_sq = s["exp_avg"], s["exp_avg_sq"]
                    step = s["step"]
                    bias_corr1 = 1 - beta1 ** step
                    bias_corr2 = 1 - beta2 ** step
                    denom = exp_avg_sq.sqrt().div_(math.sqrt(bias_corr2)).add_(eps)
                    step_size = lr / bias_corr1
                    update = exp_avg.div(denom)
                    # Cautious mask: move only where update agrees with gradient sign
                    update.mul_((update * g) > 0)
                    p.mul_(1 - lr * wd)          # weight decay (unmasked, per paper)
                    p.add_(update, alpha=-step_size)
        return loss


if _HAS_TRITON:

    @triton.jit
    def _cautious_tail_wd_kernel(p_ptr, n_elem, wd_factor, BLOCK: tl.constexpr):
        """Pass 2a (bf16 params): weight decay alone (unmasked, per paper).
        Separate kernel so the store boundary forces the p*wd_factor rounding
        (fusing it into the apply fma would skip a round). torch's bf16
        mul_ converts the scalar to bf16 and computes in fp32 — wd_factor is
        pre-rounded to bf16 by the caller to match."""
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n_elem
        p = tl.load(p_ptr + offs, mask=mask).to(tl.float32)
        p = p * wd_factor
        tl.store(p_ptr + offs, p, mask=mask)

    @triton.jit
    def _cautious_tail_wd_f64_kernel(p_ptr, n_elem, wd_factor, BLOCK: tl.constexpr):
        """Pass 2a (fp32 params): torch's fp32 mul_ with a Python-float scalar
        computes in FP64 and rounds once at store (verified: fp32-scalar math
        differs in ~all elements). Emulate: fp64 multiply, round to fp32."""
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n_elem
        p = tl.load(p_ptr + offs, mask=mask).to(tl.float32)
        p = (p.to(tl.float64) * wd_factor).to(tl.float32)
        tl.store(p_ptr + offs, p, mask=mask)

    @triton.jit
    def _cautious_tail_apply_kernel(p_ptr, g_ptr, m_ptr, v_ptr, n_elem,
                                    sqrt_bias_corr2, step_size, eps,
                                    BLOCK: tl.constexpr):
        """Pass 2b: denom, update, cautious mask, apply.

        Mirrors torch op-for-op: v.sqrt() (IEEE sqrtf) then div_ by the FP64
        math.sqrt(bias_corr2) scalar and add_ of the FP64 eps — both computed
        in fp64 with a single round to fp32, like torch's scalar kernels.
        update = m/denom is fp32 tensor/tensor division (IEEE). Mask =
        (update*g > 0). Apply = single-rounding fma (torch's add_(alpha=)).
        bf16 params round on store."""
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n_elem
        m = tl.load(m_ptr + offs, mask=mask)
        v = tl.load(v_ptr + offs, mask=mask)
        g = tl.load(g_ptr + offs, mask=mask)
        denom = tl.math.sqrt_rn(v).to(tl.float64)          # IEEE sqrtf, widen
        denom = denom / sqrt_bias_corr2                    # fp64 division (IEEE div.rn.f64)
        denom = (denom + eps).to(tl.float32)               # fp64 eps add, round fp32
        update = tl.math.div_rn(m, denom)                  # fp32 tensor div (IEEE)
        update = update * (update * g > 0)
        p = tl.load(p_ptr + offs, mask=mask).to(tl.float32)
        # torch's add_(alpha=-step_size) is a SINGLE-ROUNDING fp32 fma.
        p = tl.math.fma(-step_size, update, p)
        tl.store(p_ptr + offs, p, mask=mask)


    def _cautious_tail_triton(params, grads, states, lr, wd, eps, beta1, beta2):
        """Pass 2 via two fused Triton kernels per param (in-place, temp-free).

        Replaces ~9 torch launches per param with 2; math stays fp32 (scalar
        ops emulated in fp64 like torch). Verified bitwise against the torch
        loop in the SDLC suite (bf16 + fp32 params)."""
        for p, g, s in zip(params, grads, states):
            step = s["step"]
            bias_corr1 = 1.0 - beta1 ** step
            bias_corr2 = 1.0 - beta2 ** step
            step_size = lr / bias_corr1
            # torch's div_ kernel receives the FP64 math.sqrt(bias_corr2) and
            # eps as doubles — keep them fp64 end-to-end in the kernel.
            sqrt_bias_corr2 = math.sqrt(bias_corr2)
            n = p.numel()
            grid = (triton.cdiv(n, 1024),)
            if p.dtype == torch.bfloat16:
                # torch converts the scalar to bf16 and computes in fp32.
                wd_factor = float(torch.tensor(1.0 - lr * wd, dtype=torch.bfloat16))
                _cautious_tail_wd_kernel[grid](p, n, wd_factor, BLOCK=1024, num_warps=4)
            else:
                # torch keeps the Python-float scalar as fp64 and rounds once.
                _cautious_tail_wd_f64_kernel[grid](p, n, 1.0 - lr * wd, BLOCK=1024, num_warps=4)
            _cautious_tail_apply_kernel[grid](
                p, g, s["exp_avg"], s["exp_avg_sq"], n,
                sqrt_bias_corr2, step_size, eps,
                BLOCK=1024, num_warps=4,
            )


def make_optimizer(model, base_lr: float, cautious: bool = False):
    params = list(model.parameters())
    vocab_embed_numel = 49152 * 1536
    g2d = [p for p in params if p.ndim >= 2 and p.numel() != vocab_embed_numel]
    geh = [p for p in params if p.ndim >= 2 and p.numel() == vocab_embed_numel]
    gsc = [p for p in params if p.ndim < 2]
    groups = [
        dict(params=g2d, lr=base_lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1, fused=True),
        dict(params=geh, lr=base_lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1, fused=True),
        dict(params=gsc, lr=base_lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0, fused=True),
    ]
    if cautious:
        return CautiousAdamW(groups)
    return torch.optim.AdamW(groups)


def build_dataloader(ds, batch_size, num_workers=2):
    """DataLoader with worker processes so shard reads run off the GPU thread.

    persistent_workers stays False (default): workers are forked fresh on each
    iter(dl), so a curriculum rebuild (ds._build_stratified_order) is picked up.
    """
    return DataLoader(ds, batch_size=batch_size, collate_fn=collate_pretrain,
                      shuffle=False, num_workers=num_workers, pin_memory=True,
                      prefetch_factor=2)


def accumulate_loss(acc, loss):
    """GPU-friendly loss accumulator.

    Old pattern: acc += loss.item() — one GPU→CPU sync per micro-batch (16 per
    step on the 1B run), each draining the async pipeline. New pattern: accumulate
    the detached GPU tensor and sync ONCE at log/save time. .detach() is critical —
    without it acc would retain the autograd graph of every micro-batch (16 graphs
    alive at once → OOM).
    """
    return acc + loss.detach()


_save_lock = threading.Lock()


def _to_cpu_deep(obj):
    """Deep-copy nested state (tensors → CPU) so a background saver thread can
    write the file without racing the training loop mutating GPU weights."""
    if isinstance(obj, torch.Tensor):
        return obj.detach().to("cpu")
    if isinstance(obj, dict):
        return {k: _to_cpu_deep(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_to_cpu_deep(v) for v in obj)
    return obj


def compute_log_stats(running, n_steps, dt, batch_size, seq_len, grad_accum):
    """Mean loss + throughput for the last n_steps.

    n_steps = log_interval normally, but FEWER right after a resume (first
    log line fires before a full interval has elapsed). Dividing by the true
    count keeps that first post-resume line honest — the old bug printed
    loss×14/100 ≈ 0.27 instead of the real ~1.98.
    """
    avg = (running / n_steps).item()
    tps = batch_size * seq_len * grad_accum * n_steps / dt
    return avg, tps, dt / n_steps


def save_swa_snapshot_robust(step, cpu_model_sd, window, output_dir, logger):
    """Lean model-only snapshot (bf16 weights, ~2.1GB) for end-of-run SWA.

    Written alongside step-interval saves only (never time-based — those
    would flood the window). Prunes to the last `window` snapshots so the
    tail stays bounded (window × ~2.1GB ≈ 50GB for 24 — fits the 221GB free).
    Averaged at run end via swa_average.py → flat-basin final weights.
    """
    swa_dir = Path(output_dir) / "swa_tail"
    swa_dir.mkdir(parents=True, exist_ok=True)
    path = swa_dir / f"swa_{step:06d}.pt"
    tmp = swa_dir / f"swa_{step:06d}.tmp"
    try:
        torch.save({"step": int(step), "model_state_dict": cpu_model_sd}, tmp)
        os.replace(tmp, path)
    except Exception as e:
        logger.warning(f"swa snapshot step {step} failed: {e}")
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        return
    if window and window > 0:
        snaps = sorted(swa_dir.glob("swa_*.pt"))
        for old in snaps[:-window]:
            try:
                old.unlink()
            except OSError:
                pass


def save_checkpoint_async(state, is_best, logger, output_dir=None,
                          swa_window=0, swa_step=None):
    """Snapshot state to CPU on the calling thread, then write the ~6.2GB file
    on a background thread so training doesn't stall on disk I/O.

    Returns the thread (join it in tests). Serialized via save_checkpoint_robust
    (NaN guard + atomic tmp→rename), so a torn write never corrupts latest.pt.
    When swa_window>0, a lean model-only snapshot (swa_tail/swa_<step>.pt) is
    written in the same writer under the same lock — reuses the CPU copy, so no
    extra RAM is needed.
    """
    output_dir = output_dir or OUTPUT_DIR
    cpu_state = _to_cpu_deep(state)

    def _writer():
        with _save_lock:
            save_checkpoint_robust(cpu_state, output_dir, is_best, logger)
            if swa_window and swa_window > 0 and swa_step is not None:
                save_swa_snapshot_robust(swa_step, cpu_state["model_state_dict"],
                                         swa_window, output_dir, logger)

    t = threading.Thread(target=_writer, daemon=True, name="ckpt-saver")
    t.start()
    return t


def resolve_resume_path(explicit, latest_path):
    """User rule: ALWAYS resume from the latest checkpoint, never best.

    An explicit --resume-from wins; otherwise default to latest.pt if it
    exists; otherwise None (fresh start). --init-from callers skip the
    auto-default (weights-only warm start is intentional there).
    """
    if explicit:
        return explicit
    if latest_path and os.path.exists(latest_path):
        return latest_path
    return None


def apply_resume(ckpt_path, model, optimizer, logger):
    """True resume: restore model weights + optimizer state + step + best_loss.

    Returns (start_step, best_loss). Unlike --init-from (weights-only warm
    start), this continues the LR schedule, curriculum phase, and best-loss
    guard from where the run left off — a fresh best_loss=inf would make the
    first save unconditionally overwrite best.pt with a worse loss.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt["model_state_dict"]
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        logger.warning(f"resume: missing keys ({len(missing)}): {missing[:5]}")
    if unexpected:
        logger.warning(f"resume: unexpected keys ({len(unexpected)}): {unexpected[:5]}")
    if "optimizer_state_dict" in ckpt:
        try:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            logger.info("resume: optimizer state restored")
        except (RuntimeError, ValueError) as e:
            logger.warning(f"resume: optimizer state mismatch ({e}) — fresh optimizer")
    start_step = ckpt.get("step", 0)
    best_loss = ckpt.get("best_loss", float("inf"))
    logger.info(f"🔁 Resumed from {ckpt_path} (step {start_step}, best_loss {best_loss:.4f})")
    del ckpt, sd
    torch.cuda.empty_cache()
    return start_step, best_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--num-steps", type=int, default=60000)
    parser.add_argument("--max-seq-len", type=int, default=SEQ_LEN)
    parser.add_argument("--log-interval", type=int, default=400)
    parser.add_argument("--save-interval", type=int, default=3000)
    parser.add_argument("--save-every-minutes", type=int, default=0,
                        help="Also save every N wall-clock minutes (0 = step-based only). "
                             "Keeps worst-case loss window small regardless of step speed.")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup-steps", type=int, default=WARMUP_DEFAULT)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--init-from", type=str, default=None,
                        help="Optional checkpoint .pt to load MODEL WEIGHTS from "
                             "(warm start; optimizer starts fresh — bf16 family)")
    parser.add_argument("--resume-from", type=str, default=None,
                        help="Checkpoint .pt to FULLY resume from: model weights + "
                             "optimizer state + step + best_loss. LR schedule and "
                             "curriculum continue where they left off (unlike "
                             "--init-from, which restarts the schedule at step 0).")
    parser.add_argument("--curriculum", action="store_true",
                        help="Enable G1-G4 curriculum (StratifiedShardDataset + "
                             "get_curriculum_ratios). Default: flat farm.")
    parser.add_argument("--compile", action="store_true",
                        help="Wrap model in torch.compile (fused kernels — first steps "
                             "slower due to JIT warmup, then faster). Falls back to "
                             "eager automatically if JIT warmup OOMs.")
    parser.add_argument("--liger", action="store_true",
                        help="Swap RMSNorm/SwiGLU/RoPE for Liger fused Triton kernels "
                             "before building the model.")
    parser.add_argument("--cautious", action="store_true",
                        help="Use CautiousAdamW (arXiv:2411.16085) instead of fused AdamW. "
                             "Same m/v state layout — resumes AdamW checkpoints losslessly.")
    parser.add_argument("--fused-ce", action="store_true",
                        help="Liger fused linear+CE (no [B,S,V] logits materialization — "
                             "frees ~1.2GB, unlocks --batch-size 4 on 12GB). Same "
                             "next-token shift + mean semantics as chunked CE.")
    parser.add_argument("--swa-window", type=int, default=0,
                        help="Keep the last N lean model-only snapshots in "
                             "swa_tail/ on step-interval saves (each ~2.1GB bf16). "
                             "0 = off. Averaged at run end via swa_average.py → "
                             "flat-basin final weights (SWA).")
    args = parser.parse_args()

    # User rule: ALWAYS resume from latest, never best. Only auto-default when
    # no explicit resume AND no intentional weights-only warm start.
    if args.resume_from is None and args.init_from is None:
        args.resume_from = resolve_resume_path(None, str(LATEST))
        if args.resume_from:
            logger.info(f"🔁 No --resume-from given — auto-resumed from LATEST: {args.resume_from}")

    torch.manual_seed(42)
    device = "cuda"

    apply_liger_if_requested(args.liger)  # #2 — must precede model build
    model = build_model(torch.bfloat16).to(device)
    optimizer = make_optimizer(model, args.lr, cautious=args.cautious)
    start_step = 0
    best_loss = float("inf")
    if args.resume_from and os.path.exists(args.resume_from):
        start_step, best_loss = apply_resume(args.resume_from, model, optimizer, logger)
    elif args.init_from and os.path.exists(args.init_from):
        ckpt = torch.load(args.init_from, map_location="cpu", weights_only=False)
        sd = ckpt["model_state_dict"]
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing:
            logger.warning(f"init-from: missing keys ({len(missing)}): {missing[:5]}")
        if unexpected:
            logger.warning(f"init-from: unexpected keys ({len(unexpected)}): {unexpected[:5]}")
        logger.info(f"🔥 Warm-started from {args.init_from} (step {ckpt.get('step', '?')})")
        del ckpt, sd
        torch.cuda.empty_cache()
    if args.compile:
        logger.info("⚡ Compiling model with torch.compile (mode=reduce-overhead → CUDA graphs; JIT on first steps)...")
        model = torch.compile(model, mode="reduce-overhead")

    if args.curriculum:
        ratios0 = get_curriculum_ratios(start_step, args.num_steps)
        logger.info(f"📚 G1-G4 curriculum ON — start ratios: {ratios0}")
        ds = StratifiedShardDataset(SHARD_DIRS, seq_len=args.max_seq_len,
                                    ratios=ratios0, dedup=False)
    else:
        logger.info(f"Flat farm: {FARM_DIR}")
        ds = FlatFarmDataset(FARM_DIR, seq_len=args.max_seq_len, val_frac=0.01, is_val=False)
    dl = build_dataloader(ds, args.batch_size)
    data_iter = iter(dl)

    if args.compile:
        # #1 — surface any compile-time OOM here; fall back through
        # reduce-overhead → default mode → eager (never lose the run).
        compile_mode = "reduce-overhead"
        while True:
            try:
                _compile_warmup(model, dl, device)
                logger.info(f"⚡ torch.compile JIT warmup OK ({compile_mode}) — training compiled")
                break
            except torch.cuda.OutOfMemoryError:
                if compile_mode == "reduce-overhead":
                    logger.warning("⚡ reduce-overhead OOM during JIT — falling back to default mode")
                    model = torch.compile(_unwrap_compiled(model))
                    compile_mode = "default"
                    continue
                logger.warning("⚡ torch.compile OOM during JIT warmup — falling back to eager")
                model = _unwrap_compiled(model)
                torch.cuda.empty_cache()
                args.compile = False
                break
            except Exception as e:  # noqa: BLE001 — graph-capture/graph-break errors
                logger.warning(f"⚡ torch.compile ({compile_mode}) failed: {type(e).__name__}: {e} — falling back to eager")
                model = _unwrap_compiled(model)
                torch.cuda.empty_cache()
                args.compile = False
                break
        data_iter = iter(dl)

    logger.info(f"Starting pure-GPU pretraining (bf16 AdamW, recompute-free — gc inert)"
                f"{' from scratch' if start_step == 0 else f' — resuming at step {start_step}'}...")
    global_step = start_step
    running = torch.zeros((), device=device)
    t0 = time.time()
    last_save_time = time.time()
    last_log_step = start_step  # first log after resume has < log_interval steps

    for step in range(start_step, args.num_steps):
        # ── G1-G4 curriculum: rebuild tier ratios every CURRICULUM_UPDATE_INTERVAL ──
        # G1+G3: smooth easy→hard weights at t=step/total (no cliffs)
        # G2: cosine easy-review boost + renormalize (inside get_curriculum_ratios)
        # G4: windowed JIT shuffle (inside _build_stratified_order)
        if args.curriculum and step > 0 and step % CURRICULUM_UPDATE_INTERVAL == 0:
            new_ratios = get_curriculum_ratios(step, args.num_steps)
            ds.ratios = new_ratios
            ds._build_stratified_order()
            data_iter = iter(dl)
            logger.info(f"📚 G1-G4 curriculum @ step {step}: {new_ratios}")
        lr = get_lr(step + 1, args.warmup_steps, args.num_steps, args.lr, args.min_lr)
        for g in optimizer.param_groups:
            g["lr"] = lr

        acc_loss = torch.zeros((), device=device, dtype=torch.float32)
        try:
            acc_loss, data_iter = run_microbatches(model, data_iter, dl, device, args)
        except torch.OutOfMemoryError:
            # #1 safety net: compiled mode can OOM mid-step even when the JIT
            # warmup passed (graph capture holds intermediates → ~11.5GB peak).
            # Drop compile once and retry the step eagerly.
            if not args.compile:
                raise
            logger.warning("⚡ OOM in step — dropping torch.compile, retrying step eagerly")
            model = _unwrap_compiled(model)
            torch.cuda.empty_cache()
            args.compile = False
            optimizer.zero_grad(set_to_none=True)
            acc_loss, data_iter = run_microbatches(model, data_iter, dl, device, args)

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        global_step += 1
        running += acc_loss

        if (step + 1) % args.log_interval == 0:
            dt = time.time() - t0
            t0 = time.time()
            n_log = (step + 1) - last_log_step
            last_log_step = step + 1
            avg, tps, s_step = compute_log_stats(
                running, n_log, dt, args.batch_size, args.max_seq_len, args.grad_accum)
            running = torch.zeros((), device=device)
            mem = torch.cuda.max_memory_allocated(device) / 1024**3
            logger.info(
                f"Step {step+1}/{args.num_steps} | Loss {avg:.4f} | LR {lr:.2e} | "
                f"{s_step:.2f}s/step | {tps:.0f} tok/s | GPU {mem:.2f}GB"
            )

        step_save = (step + 1) % args.save_interval == 0 or step + 1 == args.num_steps
        if should_save_checkpoint(step + 1, args.save_interval, last_save_time,
                                  time.time(), args.save_every_minutes) or step + 1 == args.num_steps:
            acc_loss_f = acc_loss.item()  # ONE sync per save, not 16 per step
            is_best = acc_loss_f < best_loss
            if is_best:
                best_loss = acc_loss_f
            state = {
                "step": step + 1,
                "loss": acc_loss_f,
                "best_loss": best_loss,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": args.__dict__,
            }
            save_checkpoint_async(state, is_best, logger,
                                  swa_window=args.swa_window if step_save else 0,
                                  swa_step=step + 1)
            last_save_time = time.time()

    logger.info("✅ Pure-GPU pretraining complete!")


if __name__ == "__main__":
    main()
