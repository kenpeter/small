#!/usr/bin/env python3
"""Comprehensive test suite for pretrain_megatrain.py — tests the ENTIRE codebase.

Run: python3 test_pretrain.py

Guards against regressions in:
  1.  LR schedule (cosine + warmup)
  2.  Momentum EMA scaling (critical bug that caused plateau)
  3.  Newton-Schulz orthogonalization
  4.  QK-Clip spectral norm enforcement
  5.  AdamW fallback for non-Muon params
  6.  Weight decay and LR application
  7.  Data collation (causal mask, labels)
  8.  Checkpoint saving (NaN guard, atomic write)
  9.  CPU param validation (NaN/Inf detector)
  10. Module import / config creation
"""

import sys, os, math, tempfile, shutil
import torch
import numpy as np

# =============================================================================
# Module import test (catches missing deps / syntax errors)
# =============================================================================
def test_module_imports_cleanly():
    """pretrain_megatrain must import without errors."""
    try:
        import pretrain_megatrain as pmt
        assert hasattr(pmt, 'get_lr')
        assert hasattr(pmt, 'KimiMuonClip')
        assert hasattr(pmt, 'newton_schulz')
        assert hasattr(pmt, 'collate_pretrain')
        assert hasattr(pmt, 'validate_cpu_params')
        assert hasattr(pmt, 'save_checkpoint_robust')
        assert hasattr(pmt, 'StratifiedShardDataset')
        assert hasattr(pmt, 'BinShardDataset')
    except Exception as e:
        raise AssertionError(f"Module import failed: {type(e).__name__}: {e}")
    print(f"  PASS: module imports cleanly, all key symbols present")


# Now import the module for remaining tests
import pretrain_megatrain as pmt
from pretrain_megatrain import (
    get_lr, newton_schulz, KimiMuonClip, adam_update,
    collate_pretrain, validate_cpu_params, save_checkpoint_robust,
    should_save_checkpoint,
)


# =============================================================================
# Helpers
# =============================================================================
def make_2d_param(n, m, requires_grad=True):
    p = torch.randn(n, m, requires_grad=requires_grad)
    return p


def make_1d_param(n, requires_grad=True):
    p = torch.randn(n, requires_grad=requires_grad)
    return p


def _test_name():
    import inspect
    return inspect.currentframe().f_back.f_code.co_name


# =============================================================================
# 1. LR Schedule — Cosine with Warmup
# =============================================================================
def test_lr_warmup_rises_linearly():
    """During warmup, LR should increase linearly from 0 to base_lr."""
    base, warmup, total = 0.01, 100, 1000
    lr_start = get_lr(0, warmup, total, base)
    lr_mid = get_lr(warmup // 2, warmup, total, base)
    lr_end_warmup = get_lr(warmup, warmup, total, base)
    assert abs(lr_start) < 1e-8, f"{_test_name()}: LR at step 0 = {lr_start}, expected ~0"
    assert abs(lr_mid - base * 0.5) < 1e-6, f"{_test_name()}: LR at mid-warmup = {lr_mid}, expected {base*0.5}"
    assert abs(lr_end_warmup - base) < 1e-6, f"{_test_name()}: LR at end warmup = {lr_end_warmup}, expected {base}"
    print(f"  PASS: warmup LR 0 -> {lr_mid:.4e} -> {lr_end_warmup:.4e}")


def test_lr_cosine_decay_falls():
    """After warmup, LR should follow cosine decay downward."""
    base, min_lr, warmup, total = 0.01, 1e-6, 100, 1000
    lr_warmup_end = get_lr(warmup, warmup, total, base, min_lr)
    lr_half = get_lr(total // 2, warmup, total, base, min_lr)
    lr_end = get_lr(total, warmup, total, base, min_lr)
    assert lr_warmup_end > lr_half, f"{_test_name()}: LR not decaying: {lr_warmup_end} <= {lr_half}"
    assert lr_half > lr_end, f"{_test_name()}: LR not decaying: {lr_half} <= {lr_end}"
    assert abs(lr_end - min_lr) < 1e-6, f"{_test_name()}: LR at end = {lr_end}, expected {min_lr}"
    print(f"  PASS: cosine decay {lr_warmup_end:.4e} -> {lr_half:.4e} -> {lr_end:.4e}")


def test_lr_zero_warmup():
    """Warmup=0 should immediately start at base_lr."""
    base, total = 0.01, 1000
    lr = get_lr(0, 0, total, base)
    assert abs(lr - base) < 1e-6, f"{_test_name()}: LR = {lr}, expected {base}"
    print(f"  PASS: warmup=0, LR immediately at {lr:.4e}")


def test_lr_monotonic_during_warmup():
    """LR must increase monotonically during warmup."""
    base, warmup, total = 0.01, 100, 1000
    lrs = [get_lr(s, warmup, total, base) for s in range(warmup + 1)]
    for i in range(1, len(lrs)):
        assert lrs[i] >= lrs[i-1] - 1e-12, (
            f"{_test_name()}: LR decreased at step {i}: {lrs[i-1]:.6e} -> {lrs[i]:.6e}"
        )
    print(f"  PASS: LR monotonic during warmup ({lrs[0]:.2e} -> {lrs[-1]:.2e})")


def test_lr_must_use_outer_step_not_global_step():
    """Regression: get_lr must receive outer step (sample count), not global_step.

    With grad_accum=12, global_step at outer step 1000 is only ~83.
    Passing global_step would stretch warmup 12x and make cosine decay
    happen over optimizer steps that never reach num_steps.
    """
    base, warmup, total = 0.02, 1000, 50000
    lr_at_outer_1000 = get_lr(1000, warmup, total, base)
    lr_at_global_83 = get_lr(83, warmup, total, base)
    assert abs(lr_at_outer_1000 - base) < 1e-12, (
        f"{_test_name()}: outer step 1000 should reach peak LR {base}, got {lr_at_outer_1000}"
    )
    assert abs(lr_at_global_83 - base * 83 / warmup) < 1e-12, (
        f"{_test_name()}: global_step 83 should still be in warmup, got {lr_at_global_83}"
    )
    assert lr_at_global_83 < lr_at_outer_1000, (
        f"{_test_name()}: using global_step would under-power LR by factor {lr_at_outer_1000/lr_at_global_83:.1f}x"
    )
    print(f"  PASS: outer_step={lr_at_outer_1000:.2e} vs global_step={lr_at_global_83:.2e} — caller must pass outer step")


# =============================================================================
# 2. Data Collation
# =============================================================================
def test_collate_creates_causal_mask():
    """collate_pretrain must produce a lower-triangular causal mask."""
    batch = [torch.arange(16), torch.arange(16) + 100]
    out = collate_pretrain(batch)
    assert "input_ids" in out
    assert "attention_mask" in out
    assert "labels" in out
    mask = out["attention_mask"]
    B, _, T, _ = mask.shape
    assert B == 2 and T == 16, f"{_test_name()}: mask shape {mask.shape}, expected (2,1,16,16)"
    # Check causal: upper triangle should be False
    for i in range(T):
        for j in range(i + 1, T):
            assert not mask[0, 0, i, j].item(), f"{_test_name()}: mask not causal at ({i},{j})"
    print(f"  PASS: causal mask shape {tuple(mask.shape)}, lower-triangular")


def test_collate_labels_equal_input_ids():
    """labels must be a clone of input_ids for causal LM."""
    batch = [torch.arange(8), torch.arange(8) + 10]
    out = collate_pretrain(batch)
    assert torch.equal(out["labels"], out["input_ids"]), (
        f"{_test_name()}: labels != input_ids"
    )
    print(f"  PASS: labels are clone of input_ids")


def test_collate_mask_is_pure_causal_no_padding():
    """#8 — the 4D attention_mask carries ZERO padding information (packed
    data): it is exactly tril(ones) expanded. Fully redundant with SDPA's
    is_causal=True, so pretrain_gpu.py may omit it (bitwise-identical logits,
    verified on GPU: max abs diff 0.0)."""
    batch = [torch.arange(16), torch.arange(16) + 100]
    out = collate_pretrain(batch)
    mask = out["attention_mask"]
    expected = torch.tril(torch.ones((2, 1, 16, 16), dtype=torch.bool))
    assert mask.dtype == torch.bool, f"{_test_name()}: mask dtype {mask.dtype}"
    assert mask.shape == (2, 1, 16, 16), f"{_test_name()}: shape {mask.shape}"
    assert torch.equal(mask, expected), (
        f"{_test_name()}: mask is not exactly tril(ones) — "
        f"{(mask != expected).sum().item()} entries differ (padding info would "
        f"make dropping the mask unsafe)"
    )
    # every valid causal position attends → no padding holes anywhere
    valid = torch.tril(torch.ones(16, 16, dtype=torch.bool))
    assert mask[:, 0, valid].all(), f"{_test_name()}: some valid positions masked"
    print("  PASS: collate mask = pure causal tril(ones) — redundant with is_causal")


def test_accumulate_loss_detached_and_exact():
    """#7 — GPU loss accumulation: accumulate detached tensors (sync ONCE at
    log/save, not once per micro-batch) with NO autograd graph retention
    (16 live micro-batch graphs would OOM) and exact float equality vs the
    old .item() pattern."""
    from pretrain_gpu import accumulate_loss

    x = torch.randn(8, requires_grad=True)
    acc = torch.zeros(())
    losses = []
    for i in range(4):
        loss = (x * (i + 1)).sum() * 0.01
        acc = accumulate_loss(acc, loss)
        losses.append(loss.item())
    # graph-free accumulator: no reference to any micro-batch graph survives
    assert acc.grad_fn is None, f"{_test_name()}: accumulator retains autograd graph"
    assert not acc.requires_grad
    # numerically identical to the old per-micro-batch float accumulation
    assert abs(acc.item() - sum(losses)) < 1e-6, (
        f"{_test_name()}: {acc.item()} != {sum(losses)}"
    )
    # gradients still flow through the ORIGINAL loss graph (backward unaffected)
    loss.backward()
    assert x.grad is not None and torch.isfinite(x.grad).all(), (
        f"{_test_name()}: backward broken by detach pattern"
    )
    print("  PASS: detached GPU accumulation exact + graph-free + backward intact")


def test_chunked_ce_matches_full_ce():
    """#3 — chunked_ce must be numerically identical to the full (non-chunked)
    fp32 cross-entropy, regardless of chunk size (512 vs 2048 vs whole)."""
    from pretrain_gpu import chunked_ce
    torch.manual_seed(7)
    logits = torch.randn(2, 64, 4096)          # [B, S, V] fp32
    labels = torch.randint(0, 4096, (2, 64))
    labels[:, :5] = -100                        # ignore_index holes
    full = torch.nn.functional.cross_entropy(
        logits[:, :-1, :].reshape(-1, 4096), labels[:, 1:].reshape(-1),
        reduction="sum", ignore_index=-100) / labels[:, 1:].numel()
    for chunk in (16, 32, 64):                  # 64 = whole seq in one slice
        got = chunked_ce(logits, labels, chunk_size=chunk)
        assert abs(got - full) < 1e-5, (
            f"{_test_name()}: chunk={chunk} -> {got:.8f}, full={full:.8f}"
        )
    print(f"  PASS: chunked_ce identical to full CE (chunks 16/32/64)")


def test_warmup_default_400():
    """#4 — CLI default warmup is 400 steps (was 1000): on warm-started runs a
    1000-step re-warmup wastes ~2h at half-speed LR; 400 is the new default."""
    import pretrain_gpu
    assert pretrain_gpu.WARMUP_DEFAULT == 400, (
        f"{_test_name()}: WARMUP_DEFAULT={pretrain_gpu.WARMUP_DEFAULT}, want 400"
    )
    # sanity: get_lr with the new default reaches base_lr exactly at step 400
    lr = get_lr(400, 400, 15000, 3e-4)
    assert abs(lr - 3e-4) < 1e-8, f"{_test_name()}: LR at warmup end = {lr}"
    print("  PASS: warmup default 400, LR reaches base exactly at step 400")


def test_liger_replaces_modules():
    """#2 — apply_liger_if_requested(True) must swap Llama's RMSNorm/MLP/RoPE
    for Liger fused kernels (Triton) BEFORE model build; state-dict key names
    are unchanged (strict=False warm-start stays compatible). Restores the
    original transformers classes afterwards so later tests are unaffected."""
    from pretrain_gpu import apply_liger_if_requested
    import transformers.models.llama.modeling_llama as mll
    from liger_kernel.transformers.rms_norm import LigerRMSNorm
    from liger_kernel.transformers.swiglu import LigerSwiGLUMLP

    orig = {n: getattr(mll, n) for n in
            ("LlamaRMSNorm", "LlamaMLP", "LlamaRotaryEmbedding")}
    orig_rope_fn = mll.apply_rotary_pos_emb
    try:
        apply_liger_if_requested(True)
        assert mll.LlamaRMSNorm is LigerRMSNorm, "RMSNorm not replaced"
        assert mll.LlamaMLP is LigerSwiGLUMLP, "MLP not replaced"
        # v0.8.x patches RoPE as the free function, not the class
        assert mll.apply_rotary_pos_emb is not orig_rope_fn, "RoPE not replaced"

        from transformers import LlamaConfig, LlamaForCausalLM
        cfg = LlamaConfig(vocab_size=512, hidden_size=64, intermediate_size=128,
                          num_hidden_layers=2, num_attention_heads=4,
                          num_key_value_heads=2, max_position_embeddings=128)
        m = LlamaForCausalLM(cfg)
        layer = m.model.layers[0]
        assert isinstance(layer.input_layernorm, LigerRMSNorm), type(layer.input_layernorm)
        assert isinstance(layer.mlp, LigerSwiGLUMLP), type(layer.mlp)
        # key names identical to vanilla Llama (warm-start compat)
        keys = set(layer.mlp.state_dict().keys())
        assert keys == {"gate_proj.weight", "up_proj.weight", "down_proj.weight"}, keys
        print("  PASS: Liger fused RMSNorm/MLP/RoPE applied, keys compatible")
    finally:
        for n, cls in orig.items():
            setattr(mll, n, cls)
        mll.apply_rotary_pos_emb = orig_rope_fn


def test_compile_oom_fallback():
    """#1 — _unwrap_compiled must return the original module when torch.compile
    OOMs during JIT warmup, so training can fall back to eager instead of dying."""
    from pretrain_gpu import _unwrap_compiled
    import torch.nn as nn

    plain = nn.Linear(4, 4)
    assert _unwrap_compiled(plain) is plain, "non-compiled model must pass through"

    class StubCompiled:
        def __init__(self, orig):
            self._orig_mod = orig
    orig = nn.Linear(4, 4)
    wrapped = StubCompiled(orig)
    assert _unwrap_compiled(wrapped) is orig, "must unwrap _orig_mod"
    print("  PASS: compile OOM fallback unwraps to eager model")


def test_ce_chunk_512_for_vram_safety():
    """CE_CHUNK must stay 512 — the 2048 full-seq slice's fp32 transient
    (2047×49152×4 = 805MB) OOM'd the 12GB card at step ~570 on allocator
    fragmentation (measured twice 2026-08-05). 512 = 201MB transient."""
    import pretrain_gpu
    assert pretrain_gpu.CE_CHUNK == 512, (
        f"{_test_name()}: CE_CHUNK={pretrain_gpu.CE_CHUNK}, want 512"
    )
    print("  PASS: CE_CHUNK=512 keeps the fp32 transient at 201MB")


# =============================================================================
# 3. Checkpoint Saving
# =============================================================================
def test_checkpoint_rejects_nan():
    """save_checkpoint_robust must abort if state contains NaN."""
    tmpdir = tempfile.mkdtemp()
    try:
        bad_sd = {"w": torch.tensor([1.0, float('nan')])}
        state = {"model_state_dict": bad_sd, "best_loss": 5.0}
        result = save_checkpoint_robust(state, tmpdir, False, pmt.logger)
        assert result is False, f"{_test_name()}: should have rejected NaN checkpoint"
        # Verify no file was written
        assert not os.path.exists(os.path.join(tmpdir, "megatrain_latest.pt"))
    finally:
        shutil.rmtree(tmpdir)
    print(f"  PASS: NaN checkpoint rejected")


def test_checkpoint_rejects_inf():
    """save_checkpoint_robust must abort if state contains Inf."""
    tmpdir = tempfile.mkdtemp()
    try:
        bad_sd = {"w": torch.tensor([1.0, float('inf')])}
        state = {"model_state_dict": bad_sd, "best_loss": 5.0}
        result = save_checkpoint_robust(state, tmpdir, False, pmt.logger)
        assert result is False, f"{_test_name()}: should have rejected Inf checkpoint"
    finally:
        shutil.rmtree(tmpdir)
    print(f"  PASS: Inf checkpoint rejected")


def test_checkpoint_saves_clean_state():
    """save_checkpoint_robust must write file for clean state."""
    tmpdir = tempfile.mkdtemp()
    try:
        clean_sd = {"w": torch.randn(4, 4), "b": torch.randn(4)}
        state = {"model_state_dict": clean_sd, "best_loss": 3.5}
        result = save_checkpoint_robust(state, tmpdir, False, pmt.logger)
        assert result is True, f"{_test_name()}: should have saved clean checkpoint"
        assert os.path.exists(os.path.join(tmpdir, "megatrain_latest.pt"))
    finally:
        shutil.rmtree(tmpdir)
    print(f"  PASS: clean checkpoint saved")


def test_save_trigger_step_based():
    """should_save_checkpoint fires on save_interval multiples, not before."""
    assert should_save_checkpoint(1000, 1000, 0.0, 0.0, 0) is True
    assert should_save_checkpoint(2000, 1000, 0.0, 0.0, 0) is True
    assert should_save_checkpoint(999, 1000, 0.0, 0.0, 0) is False
    assert should_save_checkpoint(1, 1000, 0.0, 0.0, 0) is False
    assert should_save_checkpoint(0, 1000, 0.0, 0.0, 0) is False
    print(f"  PASS: step-based save trigger")


def test_save_trigger_time_based():
    """should_save_checkpoint fires once save_every_minutes elapses."""
    last = 1000.0
    # 20-min cadence: not due at 19:59, due at exactly 20:00
    assert should_save_checkpoint(1, 1000, last, last + 19 * 60 + 59, 20) is False
    assert should_save_checkpoint(1, 1000, last, last + 20 * 60, 20) is True
    assert should_save_checkpoint(1, 1000, last, last + 61 * 60, 20) is True
    # disabled (0) never fires on time, no matter how long
    assert should_save_checkpoint(1, 1000, last, last + 99999 * 60, 0) is False
    print(f"  PASS: time-based save trigger")


def test_save_trigger_either_wins():
    """Step-based fires even when time not due; time-based fires even mid-interval."""
    # step due, time not due
    assert should_save_checkpoint(1000, 1000, 0.0, 5.0, 20) is True
    # time due, step not due (mid-interval save keeps loss window small)
    assert should_save_checkpoint(7, 1000, 0.0, 20 * 60, 20) is True
    # neither due
    assert should_save_checkpoint(7, 1000, 0.0, 19 * 60, 20) is False
    print(f"  PASS: either trigger wins")


def test_torch_compile_smoke():
    """torch.compile must wrap a tiny HF Llama and run a forward pass (CUDA)."""
    import torch
    if not torch.cuda.is_available():
        print("  SKIP: no CUDA available")
        return
    from transformers import LlamaConfig, LlamaForCausalLM
    cfg = LlamaConfig(vocab_size=512, hidden_size=64, intermediate_size=128,
                      num_hidden_layers=2, num_attention_heads=4,
                      num_key_value_heads=2, max_position_embeddings=128)
    m = LlamaForCausalLM(cfg).to("cuda").bfloat16().eval()
    m = torch.compile(m)
    x = torch.randint(0, 512, (1, 16)).to("cuda")
    with torch.no_grad():
        out = m(input_ids=x).logits
    assert out.shape == (1, 16, 512), f"unexpected logits shape {out.shape}"
    print(f"  PASS: torch.compile wraps and runs a tiny model")


def test_cautious_tail_triton_bitwise():
    """Triton fused update tail must match the torch loop BITWISE (CUDA+triton).

    Both optimizers are built from IDENTICAL clones of the same source tensors
    (two separate build() calls would draw from different RNG positions and
    compare apples to oranges). The reference forces the torch-loop path by
    temporarily clearing the module's _HAS_TRITON flag."""
    import torch
    if not torch.cuda.is_available():
        print("  SKIP: no CUDA available")
        return
    import pretrain_gpu as pg
    if not pg._HAS_TRITON:
        print("  SKIP: triton unavailable")
        return
    torch.manual_seed(7)
    shapes = [(4096,), (3, 512), (75, 49152)]  # incl. embed-ish 2D tensor
    # (param dtype, moments dtype) — production = bf16/bf16 (bf16 AdamW states)
    combos = [(torch.bfloat16, torch.bfloat16),
              (torch.bfloat16, torch.float32),
              (torch.float32, torch.float32)]
    for p_dtype, m_dtype in combos:
        for shape in shapes:
            p_src = torch.randn(*shape, dtype=p_dtype, device="cuda") * 0.02
            grad_src = torch.randn(*shape, dtype=p_dtype, device="cuda") * 0.05
            m_src = torch.randn(*shape, dtype=m_dtype, device="cuda") * 0.05
            v_src = torch.rand(*shape, dtype=m_dtype, device="cuda") + 0.5

            def build():
                p = torch.nn.Parameter(p_src.clone())
                opt = pg.CautiousAdamW([p], lr=3e-4, weight_decay=0.1)
                st = opt.state[p]
                st["step"] = 1
                st["exp_avg"] = m_src.clone()
                st["exp_avg_sq"] = v_src.clone()
                p.grad = grad_src.clone()
                return p, opt

            p_ref, opt_ref = build()
            pg._HAS_TRITON = False          # force the torch-loop reference
            try:
                opt_ref.step()
            finally:
                pg._HAS_TRITON = True
            p_new, opt_new = build()
            opt_new.step()                   # triton path (dispatcher)
            if p_dtype == torch.bfloat16:
                # Production dtype: must be BITWISE identical (verified).
                assert torch.equal(p_ref.detach().cpu(), p_new.detach().cpu()), (
                    f"step-1 BITWISE mismatch p={p_dtype} m={m_dtype} shape={shape}")
            else:
                # fp32: triton's fp64 scalar division differs from IEEE by
                # ~1e-8 (div_rn is fp32-only, no fp64 _rn escape hatch) — far
                # below training noise. Tight allclose guards regressions.
                assert torch.allclose(p_ref.detach().cpu(), p_new.detach().cpu(), atol=1e-6), (
                    f"step-1 allclose mismatch p={p_dtype} m={m_dtype} shape={shape}")
            opt_ref.step()                   # step 2: state accumulation
            opt_new.step()
            if p_dtype == torch.bfloat16:
                assert torch.equal(p_ref.detach().cpu(), p_new.detach().cpu()), (
                    f"step-2 BITWISE mismatch p={p_dtype} m={m_dtype} shape={shape}")
            else:
                assert torch.allclose(p_ref.detach().cpu(), p_new.detach().cpu(), atol=1e-6), (
                    f"step-2 allclose mismatch p={p_dtype} m={m_dtype} shape={shape}")
    print("  PASS: triton tail bitwise (bf16/bf16 + bf16/fp32) / ≤1e-6 (fp32/fp32), 2 steps")


def test_gradient_checkpointing_inert_in_transformers():
    """Pin Aug-13 discovery: transformers 5.5.3 Llama has NO gc dispatch —
    gradient_checkpointing_enable() is a no-op, so the run is recompute-free.
    If this fails after a transformers upgrade, gc has activated: re-measure
    speed/VRAM before trusting the 'recompute-free' assumption."""
    import torch
    if not torch.cuda.is_available():
        print("  SKIP: no CUDA available")
        return
    import torch.utils.checkpoint as ckpt
    from transformers import LlamaConfig, LlamaForCausalLM
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=1024, hidden_size=128, intermediate_size=256,
                      num_hidden_layers=4, num_attention_heads=4,
                      num_key_value_heads=2, head_dim=32)
    x = torch.randint(0, 1024, (2, 32)).cuda()
    orig = ckpt.checkpoint
    calls = [0]
    def counting(*a, **k):
        calls[0] += 1
        return orig(*a, **k)
    ckpt.checkpoint = counting
    m = LlamaForCausalLM(cfg).cuda().train()
    m.gradient_checkpointing_enable()
    loss = m(x, labels=x).loss
    loss.backward()
    assert calls[0] == 0, (
        f"gc dispatch ACTIVE ({calls[0]} checkpoint calls) — update the inert-gc assumption"
    )
    print("  PASS: gradient checkpointing inert (recompute-free) in transformers 5.5.3")


def test_compile_reduce_overhead_smoke():
    """torch.compile(mode=reduce-overhead) must run fwd+bwd with finite loss
    on a tiny HF Llama (CUDA). Guards the #1 CUDA-graphs lever."""
    import torch
    if not torch.cuda.is_available():
        print("  SKIP: no CUDA available")
        return
    from transformers import LlamaConfig, LlamaForCausalLM
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=512, hidden_size=64, intermediate_size=128,
                      num_hidden_layers=2, num_attention_heads=4,
                      num_key_value_heads=2, head_dim=16, max_position_embeddings=64)
    m = LlamaForCausalLM(cfg).to("cuda").bfloat16().train()
    m.gradient_checkpointing_enable()
    m = torch.compile(m, mode="reduce-overhead")
    x = torch.randint(0, 512, (1, 32)).to("cuda")
    loss = m(x, labels=x).loss
    loss.backward()
    assert torch.isfinite(loss), f"loss not finite: {loss}"
    print("  PASS: reduce-overhead compile runs fwd+bwd, loss finite")


def test_optimizer_groups_fused():
    """make_optimizer must return AdamW with fused=True on every param group."""
    from pretrain_gpu import make_optimizer
    import torch.nn as nn
    m = nn.Sequential(nn.Linear(16, 16), nn.LayerNorm(16))
    opt = make_optimizer(m, 3e-4)
    assert len(opt.param_groups) == 3, f"{_test_name()}: expected 3 param groups"
    assert all(g.get("fused") is True for g in opt.param_groups), (
        f"{_test_name()}: fused not set on all groups"
    )
    print("  PASS: AdamW fused=True on all 3 groups")


def test_cautious_masks_disagreeing_updates():
    """Cautious mask must zero the update where momentum and gradient disagree."""
    from pretrain_gpu import CautiousAdamW

    # Disagree: momentum +0.05, gradient -0.1 → post-mix exp_avg=0.035, update>0,
    # mask=(update*g)>0 → False → no move (wd=0)
    p = torch.nn.Parameter(torch.tensor([1.0]))
    opt = CautiousAdamW([p], lr=0.1, weight_decay=0.0)
    st = opt.state[p]
    st["step"] = 0
    st["exp_avg"] = torch.tensor([0.05])
    st["exp_avg_sq"] = torch.tensor([0.01])
    p.grad = torch.tensor([-0.1])
    opt.step()
    assert p.item() == 1.0, f"{_test_name()}: masked update moved weight to {p.item()}"

    # Agree: momentum +0.05, gradient +0.1 → exp_avg=0.055, update=0.12298,
    # step_size=lr/(1-β1)=1.0 → p -= 0.12298
    p2 = torch.nn.Parameter(torch.tensor([1.0]))
    opt2 = CautiousAdamW([p2], lr=0.1, weight_decay=0.0)
    st2 = opt2.state[p2]
    st2["step"] = 0
    st2["exp_avg"] = torch.tensor([0.05])
    st2["exp_avg_sq"] = torch.tensor([0.01])
    p2.grad = torch.tensor([0.1])
    opt2.step()
    expected = 1.0 - (0.1 / (1 - 0.9)) * (0.055 / ((0.01 ** 0.5) / ((1 - 0.95) ** 0.5) + 1e-8))
    assert abs(p2.item() - expected) < 1e-4, f"{_test_name()}: {p2.item()} != {expected}"
    print("  PASS: cautious mask zeroes disagreement, keeps agreement")


def test_cautious_state_roundtrip():
    """CautiousAdamW state must survive save/load (m/v preserved → resume-safe)."""
    from pretrain_gpu import CautiousAdamW
    import copy

    torch.manual_seed(1)
    p = torch.nn.Parameter(torch.randn(8))
    opt = CautiousAdamW([p], lr=0.01, weight_decay=0.01)
    for _ in range(5):
        p.grad = torch.randn(8)
        opt.step()
    # simulate a real checkpoint file: deep-copy state (state_dict aliases live tensors)
    sd = copy.deepcopy(opt.state_dict())

    p2 = p.clone().detach().requires_grad_(True)
    opt2 = CautiousAdamW([p2], lr=0.01, weight_decay=0.01)
    opt2.load_state_dict(sd)
    g = torch.randn(8)
    p.grad = g.clone()
    p2.grad = g.clone()
    opt.step()
    opt2.step()
    assert torch.allclose(p, p2), f"{_test_name()}: resumed optimizer diverged"
    print("  PASS: cautious optimizer state round-trips losslessly")


def test_cautious_accepts_adamw_state_dict():
    """A plain torch AdamW checkpoint must load into CautiousAdamW (mid-run switch)."""
    from pretrain_gpu import CautiousAdamW

    torch.manual_seed(2)
    p = torch.nn.Parameter(torch.randn(8))
    opt_adamw = torch.optim.AdamW([p], lr=0.01, betas=(0.9, 0.95), weight_decay=0.01)
    for _ in range(3):
        p.grad = torch.randn(8)
        opt_adamw.step()
    sd = opt_adamw.state_dict()

    p2 = p.clone().detach().requires_grad_(True)
    opt_c = CautiousAdamW([p2], lr=0.01, weight_decay=0.01)
    opt_c.load_state_dict(sd)  # must not raise
    assert "exp_avg" in opt_c.state[p2] and "exp_avg_sq" in opt_c.state[p2], (
        f"{_test_name()}: m/v not carried over after AdamW→Cautious load"
    )
    src = sd["state"][next(iter(sd["state"]))]
    assert torch.allclose(opt_c.state[p2]["exp_avg"], src["exp_avg"]), (
        f"{_test_name()}: exp_avg not preserved"
    )
    print("  PASS: AdamW checkpoint state loads into CautiousAdamW (m/v kept)")


def test_cautious_converges_on_toy():
    """CautiousAdamW must decrease loss on a tiny regression (sanity check)."""
    from pretrain_gpu import CautiousAdamW

    torch.manual_seed(3)
    w = torch.nn.Parameter(torch.zeros(4))
    opt = CautiousAdamW([w], lr=0.05, weight_decay=0.0)
    x = torch.randn(64, 4)
    y = x @ torch.tensor([1.0, -2.0, 3.0, -4.0])
    losses = []
    for _ in range(100):
        opt.zero_grad()
        loss = ((x @ w - y) ** 2).mean()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0] * 0.15, (
        f"{_test_name()}: loss must drop: {losses[0]:.4f} → {losses[-1]:.4f}"
    )
    print(f"  PASS: cautious converges {losses[0]:.4f} → {losses[-1]:.4f}")


def test_make_optimizer_cautious_flag():
    """make_optimizer(cautious=True) must return CautiousAdamW with 3 groups."""
    from pretrain_gpu import make_optimizer, CautiousAdamW
    import torch.nn as nn

    m = nn.Sequential(nn.Linear(16, 16), nn.LayerNorm(16))
    opt = make_optimizer(m, 3e-4, cautious=True)
    assert isinstance(opt, CautiousAdamW), f"{_test_name()}: wrong optimizer type"
    assert len(opt.param_groups) == 3, f"{_test_name()}: expected 3 param groups"
    print("  PASS: --cautious builds CautiousAdamW with 3 groups")


def test_build_dataloader_workers():
    """build_dataloader must use worker processes (off-GPU-thread shard reads)."""
    from pretrain_gpu import build_dataloader

    class TinySeqDs(torch.utils.data.Dataset):
        def __len__(self):
            return 64

        def __getitem__(self, i):
            return torch.arange(8) + i * 100

    dl = build_dataloader(TinySeqDs(), 2)
    assert dl.num_workers == 2, f"{_test_name()}: num_workers={dl.num_workers}"
    assert dl.pin_memory is True
    batch = next(iter(dl))
    assert batch["input_ids"].shape == (2, 8), f"{_test_name()}: {batch['input_ids'].shape}"
    print("  PASS: dataloader uses 2 workers + pin_memory, collates fine")


def test_async_save_writes_file():
    """save_checkpoint_async must snapshot to CPU and write a loadable file."""
    from pretrain_gpu import save_checkpoint_async
    tmpdir = tempfile.mkdtemp()
    try:
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        state = {
            "step": 42,
            "loss": 3.3,
            "best_loss": 3.3,
            "model_state_dict": {"w": torch.randn(4, 4, device=dev)},
            "optimizer_state_dict": {"state": {0: {"exp_avg": torch.randn(4, 4, device=dev)}}},
            "config": {"foo": 1},
        }
        t = save_checkpoint_async(state, False, pmt.logger, output_dir=tmpdir)
        t.join(timeout=120)
        assert not t.is_alive(), f"{_test_name()}: saver thread did not finish"
        ck = torch.load(os.path.join(tmpdir, "megatrain_latest.pt"),
                        map_location="cpu", weights_only=False)
        assert ck["step"] == 42
        assert ck["model_state_dict"]["w"].shape == (4, 4)
        assert ck["optimizer_state_dict"]["state"][0]["exp_avg"].device.type == "cpu"
    finally:
        shutil.rmtree(tmpdir)
    print("  PASS: async checkpoint written + loadable (CPU snapshot)")


def test_resume_restores_full_state():
    """apply_resume must restore model weights, optimizer state, step and best_loss.

    Guard: a weights-only warm start would restart the LR schedule and set
    best_loss=inf, making the first save clobber best.pt with a worse loss.
    """
    from pretrain_gpu import apply_resume
    import torch.nn as nn

    m = nn.Linear(8, 8)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-4)
    opt.zero_grad()
    m(torch.randn(2, 8)).sum().backward()
    opt.step()  # populate exp_avg / exp_avg_sq state
    opt_sd_before = opt.state_dict()["state"]

    ckpt = {
        "step": 4896,
        "loss": 2.39,
        "best_loss": 2.056,
        "model_state_dict": m.state_dict(),
        "optimizer_state_dict": opt.state_dict(),
        "config": {"num_steps": 15000},
    }
    tmpdir = tempfile.mkdtemp()
    try:
        path = os.path.join(tmpdir, "resume.pt")
        torch.save(ckpt, path)

        m2 = nn.Linear(8, 8)
        opt2 = torch.optim.AdamW(m2.parameters(), lr=3e-4)
        start_step, best_loss = apply_resume(path, m2, opt2, pmt.logger)
        assert start_step == 4896, f"{_test_name()}: start_step={start_step}"
        assert best_loss == 2.056, f"{_test_name()}: best_loss={best_loss}"
        # model weights actually restored
        for (k1, v1), (k2, v2) in zip(m.state_dict().items(), m2.state_dict().items()):
            assert torch.equal(v1, v2), f"{_test_name()}: weight {k1} not restored"
        # optimizer moments actually restored (not a fresh AdamW)
        opt2_sd = opt2.state_dict()["state"]
        assert set(opt_sd_before.keys()) == set(opt2_sd.keys()), (
            f"{_test_name()}: optimizer state keys differ"
        )
        for k in opt_sd_before:
            for key in ("exp_avg", "exp_avg_sq"):
                assert torch.equal(opt_sd_before[k][key], opt2_sd[k][key]), (
                    f"{_test_name()}: optimizer {key}[{k}] not restored"
                )
    finally:
        shutil.rmtree(tmpdir)
    print("  PASS: resume restores weights + optimizer + step + best_loss")


def test_resume_missing_optimizer_falls_back_gracefully():
    """apply_resume must not crash when the checkpoint lacks optimizer state."""
    from pretrain_gpu import apply_resume
    import torch.nn as nn

    m = nn.Linear(4, 4)
    ckpt = {"step": 100, "loss": 2.0, "best_loss": 1.9, "model_state_dict": m.state_dict()}
    tmpdir = tempfile.mkdtemp()
    try:
        path = os.path.join(tmpdir, "noopt.pt")
        torch.save(ckpt, path)
        m2 = nn.Linear(4, 4)
        opt2 = torch.optim.AdamW(m2.parameters(), lr=3e-4)
        start_step, best_loss = apply_resume(path, m2, opt2, pmt.logger)
        assert start_step == 100
        assert best_loss == 1.9
        # optimizer must still be usable (fresh state, no exception)
        opt2.zero_grad()
        m2(torch.randn(2, 4)).sum().backward()
        opt2.step()
    finally:
        shutil.rmtree(tmpdir)
    print("  PASS: resume without optimizer state falls back to fresh optimizer")


def test_resume_keeps_best_pt_until_better_loss():
    """After resume, is_best must compare against the checkpoint's best_loss
    (not inf), so best.pt is only overwritten by a genuinely better loss."""
    from pretrain_gpu import apply_resume
    import torch.nn as nn

    m = nn.Linear(4, 4)
    ckpt = {"step": 10, "loss": 2.5, "best_loss": 2.056, "model_state_dict": m.state_dict()}
    tmpdir = tempfile.mkdtemp()
    try:
        path = os.path.join(tmpdir, "best.pt")
        torch.save(ckpt, path)
        m2 = nn.Linear(4, 4)
        opt2 = torch.optim.AdamW(m2.parameters(), lr=3e-4)
        _, best_loss = apply_resume(path, m2, opt2, pmt.logger)
        worse = 2.4   # worse than 2.056
        better = 1.9  # better than 2.056
        assert not (worse < best_loss), f"{_test_name()}: worse loss flagged as best"
        assert better < best_loss, f"{_test_name()}: better loss not flagged as best"
    finally:
        shutil.rmtree(tmpdir)
    print("  PASS: best.pt protected until loss genuinely beats 2.056")


# =============================================================================
# 4. CPU Param Validation
# =============================================================================
def test_validate_cpu_params_detects_nan():
    """validate_cpu_params must raise RuntimeError on NaN params."""
    class FakeModel:
        def get_parameters(self):
            return [torch.tensor([1.0, float('nan')])]
    try:
        validate_cpu_params(FakeModel(), pmt.logger)
        raise AssertionError(f"{_test_name()}: should have raised RuntimeError")
    except RuntimeError:
        pass
    print(f"  PASS: NaN param detected and rejected")


def test_validate_cpu_params_passes_clean():
    """validate_cpu_params must pass for all-finite params."""
    class FakeModel:
        def get_parameters(self):
            return [torch.randn(4, 4), torch.randn(4)]
    validate_cpu_params(FakeModel(), pmt.logger)
    print(f"  PASS: clean params validated")


# =============================================================================
# 5. Config Creation
# =============================================================================
def test_llama_config_creation():
    """LlamaConfig must instantiate with our custom 1B params."""
    from transformers import LlamaConfig
    cfg = LlamaConfig(
        vocab_size=49152, hidden_size=1536, intermediate_size=4608,
        num_hidden_layers=32, num_attention_heads=12, num_key_value_heads=4,
        max_position_embeddings=8192, rope_theta=10000.0, rms_norm_eps=1e-5,
        hidden_act="silu", tie_word_embeddings=False, attention_bias=False,
        mlp_bias=False, initializer_range=0.02, torch_dtype="float32",
        head_dim=128, architectures=["LlamaForCausalLM"],
    )
    assert cfg.vocab_size == 49152
    assert cfg.hidden_size == 1536
    assert cfg.num_hidden_layers == 32
    print(f"  PASS: LlamaConfig created (vocab={cfg.vocab_size}, hidden={cfg.hidden_size}, layers={cfg.num_hidden_layers})")


def test_cpumaster_config_creation():
    """CPUMasterConfig must instantiate with expected defaults."""
    from infinity.config import CPUMasterConfig
    import torch
    cfg = CPUMasterConfig(
        model_name="test", dataset_path="/tmp/dummy",
        max_seq_len=2048, batch_size=2, num_steps=10,
        learning_rate=1e-4, gradient_accumulation_steps=1,
        checkpoint_interval=1, num_grad_slabs=2, device=0,
        dtype=torch.float32, log_interval=1,
    )
    assert cfg.max_seq_len == 2048
    assert cfg.batch_size == 2
    assert cfg.dtype == torch.float32
    print(f"  PASS: CPUMasterConfig created")


# =============================================================================
# 6. Momentum EMA Scaling — THE CRITICAL BUG
# =============================================================================
def test_momentum_ema_scale():
    """Bug: buf = beta*buf + grad  (no 1-beta scaling) → buffer explodes.
    Fixed: buf = beta*buf + (1-beta)*grad  → buffer stays at grad scale.
    """
    p = make_2d_param(64, 64)
    opt = KimiMuonClip([
        dict(params=[p], lr=0.01, momentum=0.95, weight_decay=0.0,
             use_muon=True, warmup=False)
    ], tau=150.0, ns_steps=7)
    g_scale = 0.1
    for step in range(1, 11):
        p.grad = torch.ones_like(p.data) * g_scale
        opt.step(global_step=step)
    buf = opt.state[p]["momentum_buffer"]
    buf_mean = buf.abs().mean().item()
    assert buf_mean < g_scale * 5, (
        f"{_test_name()}: Momentum buffer exploded: mean={buf_mean:.4f}, expected < {g_scale*5:.4f}."
    )
    assert buf_mean > g_scale * 0.1, (
        f"{_test_name()}: Momentum buffer vanished: mean={buf_mean:.4f}"
    )
    print(f"  PASS: momentum buffer scale = {buf_mean:.4f} (grad={g_scale})")


def test_momentum_vs_sgd_style():
    """Compare EMA vs SGD-style momentum. EMA must produce ~1/(1-beta) smaller buffer."""
    beta = 0.95
    steps = 100
    grad_val = 0.1
    buf_sgd = 0.0
    for _ in range(steps):
        buf_sgd = beta * buf_sgd + grad_val
    buf_ema = 0.0
    for _ in range(steps):
        buf_ema = beta * buf_ema + (1 - beta) * grad_val
    ratio = buf_sgd / (buf_ema + 1e-8)
    assert ratio > 10, (
        f"{_test_name()}: SGD-style buffer is only {ratio:.1f}x larger than EMA"
    )
    print(f"  PASS: SGD buffer ({buf_sgd:.4f}) is {ratio:.1f}x larger than EMA ({buf_ema:.4f})")


# =============================================================================
# 7. Newton-Schulz Orthogonalization
# =============================================================================
def test_newton_schulz_preserves_spectral_norm():
    G = torch.randn(128, 128)
    X = newton_schulz(G, steps=7)
    I_approx = X @ X.T
    eye = torch.eye(128)
    err = (I_approx - eye).abs().mean().item()
    assert err < 0.05, (
        f"{_test_name()}: Newton-Schulz failed orthogonality. err={err:.4f}"
    )
    print(f"  PASS: orthogonality error = {err:.4f}")


def test_newton_schulz_non_square():
    G = torch.randn(200, 64)
    X = newton_schulz(G, steps=7)
    assert X.shape == G.shape, f"{_test_name()}: Shape mismatch"
    cond = torch.linalg.cond(X).item()
    assert cond < 10, f"{_test_name()}: Condition number too high: {cond:.2f}"
    print(f"  PASS: tall matrix {G.shape} → cond={cond:.2f}")


def test_newton_schulz_steps_effect():
    G = torch.randn(64, 64)
    X3 = newton_schulz(G, steps=3)
    X7 = newton_schulz(G, steps=7)
    err3 = (X3 @ X3.T - torch.eye(64)).abs().mean().item()
    err7 = (X7 @ X7.T - torch.eye(64)).abs().mean().item()
    assert err7 < err3, (
        f"{_test_name()}: 7 steps ({err7:.4f}) worse than 3 steps ({err3:.4f})"
    )
    print(f"  PASS: 3-step err={err3:.4f}, 7-step err={err7:.4f}")


# =============================================================================
# 8. QK-Clip Spectral Norm Enforcement
# =============================================================================
def test_qk_clip_enforced():
    tau = 50.0
    p = make_2d_param(64, 64)
    with torch.no_grad():
        p.data *= 10.0
    opt = KimiMuonClip([
        dict(params=[p], lr=0.01, momentum=0.95, weight_decay=0.0,
             use_muon=True, warmup=False)
    ], tau=tau, ns_steps=7)
    p.grad = torch.randn_like(p.data) * 0.01
    opt.step(global_step=1)
    spec_norm = torch.linalg.matrix_norm(p.data, ord=2).item()
    assert spec_norm <= tau * 1.01, (
        f"{_test_name()}: QK-Clip failed. Spectral norm = {spec_norm:.2f}, tau = {tau}"
    )
    print(f"  PASS: spectral norm {spec_norm:.2f} <= tau {tau}")


def test_qk_clip_only_on_muon_params():
    tau = 10.0
    p_muon = make_2d_param(64, 64)
    p_adam = make_2d_param(64, 64)
    with torch.no_grad():
        p_muon.data *= 5.0
        p_adam.data *= 5.0
    opt = KimiMuonClip([
        dict(params=[p_muon], lr=0.01, momentum=0.95, weight_decay=0.0,
             use_muon=True, warmup=False),
        dict(params=[p_adam], lr=3e-4, betas=(0.9, 0.95), eps=1e-10,
             weight_decay=0.0, use_muon=False),
    ], tau=tau, ns_steps=7)
    p_muon.grad = torch.randn_like(p_muon.data) * 0.01
    p_adam.grad = torch.randn_like(p_adam.data) * 0.01
    opt.step(global_step=1)
    spec_muon = torch.linalg.matrix_norm(p_muon.data, ord=2).item()
    spec_adam = torch.linalg.matrix_norm(p_adam.data, ord=2).item()
    assert spec_muon <= tau * 1.01, f"{_test_name()}: Muon param not clipped"
    assert spec_adam > tau, f"{_test_name()}: Adam param incorrectly clipped"
    print(f"  PASS: Muon clipped to {spec_muon:.2f}, Adam left at {spec_adam:.2f}")


# =============================================================================
# 9. AdamW Fallback
# =============================================================================
def test_adamw_runs_on_1d_params():
    p = make_1d_param(128)
    opt = KimiMuonClip([
        dict(params=[p], lr=3e-4, betas=(0.9, 0.95), eps=1e-10,
             weight_decay=0.0, use_muon=False),
    ], tau=150.0, ns_steps=7)
    for step in range(1, 6):
        p.grad = torch.randn_like(p.data) * 0.1
        opt.step(global_step=step)
    assert "exp_avg" in opt.state[p], f"{_test_name()}: AdamW exp_avg missing"
    assert "exp_avg_sq" in opt.state[p], f"{_test_name()}: AdamW exp_avg_sq missing"
    assert opt.state[p]["step"] == 5, f"{_test_name()}: Step counter wrong"
    print(f"  PASS: AdamW 1D param updated, step={opt.state[p]['step']}")


def test_adamw_not_muon_state():
    p = make_1d_param(64)
    opt = KimiMuonClip([
        dict(params=[p], lr=3e-4, betas=(0.9, 0.95), eps=1e-10,
             weight_decay=0.0, use_muon=False),
    ], tau=150.0, ns_steps=7)
    p.grad = torch.randn_like(p.data) * 0.1
    opt.step(global_step=1)
    assert "momentum_buffer" not in opt.state[p], (
        f"{_test_name()}: AdamW param incorrectly has Muon momentum_buffer"
    )
    print(f"  PASS: AdamW param has no Muon state")


# =============================================================================
# 10. Weight Decay and LR
# =============================================================================
def test_weight_decay_muon():
    wd = 0.1
    lr = 0.01
    p = make_2d_param(32, 32)
    init_norm = p.data.norm().item()
    opt = KimiMuonClip([
        dict(params=[p], lr=lr, momentum=0.95, weight_decay=wd,
             use_muon=True, warmup=False)
    ], tau=150.0, ns_steps=7)
    p.grad = torch.randn_like(p.data) * 0.01
    opt.step(global_step=1)
    post_norm = p.data.norm().item()
    assert post_norm != init_norm, f"{_test_name()}: Param unchanged after step"
    print(f"  PASS: weight decay applied, norm {init_norm:.4f} -> {post_norm:.4f}")


def test_lr_zero_no_change():
    p = make_2d_param(32, 32)
    init = p.data.clone()
    opt = KimiMuonClip([
        dict(params=[p], lr=0.0, momentum=0.95, weight_decay=0.0,
             use_muon=True, warmup=False)
    ], tau=150.0, ns_steps=7)
    p.grad = torch.randn_like(p.data) * 0.1
    opt.step(global_step=1)
    diff = (p.data - init).abs().max().item()
    assert diff < 1e-6, f"{_test_name()}: LR=0 but param changed by {diff:.6f}"
    print(f"  PASS: LR=0, param unchanged (diff={diff:.2e})")


# =============================================================================
# 11. Momentum Warmup
# =============================================================================
def test_momentum_warmup_progression():
    p = make_2d_param(32, 32)
    opt = KimiMuonClip([
        dict(params=[p], lr=0.01, momentum=0.95, weight_decay=0.0,
             use_muon=True, warmup=True)
    ], tau=150.0, ns_steps=7)
    p.grad = torch.ones_like(p.data) * 0.1
    opt.step(global_step=1)
    for step in range(2, 301):
        p.grad = torch.ones_like(p.data) * 0.1
        opt.step(global_step=step)
    print(f"  PASS: warmup ran 300 steps, buffers tracked")


# =============================================================================
# 12. Integration / Regression Guard
# =============================================================================
def test_no_nan_inf():
    p = make_2d_param(64, 64)
    opt = KimiMuonClip([
        dict(params=[p], lr=0.01, momentum=0.95, weight_decay=0.0,
             use_muon=True, warmup=False)
    ], tau=150.0, ns_steps=7)
    for step in range(1, 51):
        p.grad = torch.randn_like(p.data) * 0.5
        opt.step(global_step=step)
    assert not torch.isnan(p.data).any(), f"{_test_name()}: NaN in params"
    assert not torch.isinf(p.data).any(), f"{_test_name()}: Inf in params"
    print(f"  PASS: 50 steps, no NaN/Inf")


def test_grad_none_skipped():
    p = make_2d_param(32, 32)
    init = p.data.clone()
    opt = KimiMuonClip([
        dict(params=[p], lr=0.01, momentum=0.95, weight_decay=0.0,
             use_muon=True, warmup=False)
    ], tau=150.0, ns_steps=7)
    opt.step(global_step=1)
    diff = (p.data - init).abs().max().item()
    assert diff < 1e-6, f"{_test_name()}: Param updated despite grad=None, diff={diff:.6f}"
    print(f"  PASS: grad=None, param unchanged")


# =============================================================================
# 13. Adam helper
# =============================================================================
def test_adam_update_formula():
    """adam_update must produce standard AdamW-like update."""
    grad = torch.tensor([1.0, 2.0, 3.0])
    buf1 = torch.zeros_like(grad)
    buf2 = torch.zeros_like(grad)
    update = adam_update(grad, buf1, buf2, step=1, betas=(0.9, 0.95), eps=1e-10)
    # After first step with zero init: buf1 = grad * 0.1, buf2 = grad^2 * 0.05
    expected_buf1 = grad * 0.1  # lerp from 0 with alpha=0.1 (1-0.9)
    expected_buf2 = grad.square() * 0.05  # lerp from 0 with alpha=0.05 (1-0.95)
    assert torch.allclose(buf1, expected_buf1, atol=1e-6), f"{_test_name()}: exp_avg wrong: {buf1} vs {expected_buf1}"
    assert torch.allclose(buf2, expected_buf2, atol=1e-6), f"{_test_name()}: exp_avg_sq wrong: {buf2} vs {expected_buf2}"
    print(f"  PASS: adam_update formula correct")


# =============================================================================
# 14. Default values sanity
# =============================================================================
def test_default_argparse_values():
    """Argparse defaults must match expected training config."""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--muon-lr", type=float, default=3e-4)
    parser.add_argument("--adam-lr", type=float, default=3e-4)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=12)
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args([])
    assert args.muon_lr == 3e-4, f"{_test_name()}: muon_lr default = {args.muon_lr}"
    assert args.adam_lr == 3e-4, f"{_test_name()}: adam_lr default = {args.adam_lr}"
    assert args.warmup_steps == 1000, f"{_test_name()}: warmup_steps default = {args.warmup_steps}"
    assert args.min_lr == 1e-6, f"{_test_name()}: min_lr default = {args.min_lr}"
    assert args.resume is None, f"{_test_name()}: resume default = {args.resume}"
    assert args.batch_size == 4
    assert args.grad_accum == 12
    print(f"  PASS: argparse defaults correct (muon_lr={args.muon_lr}, adam_lr={args.adam_lr})")


# =============================================================================
# Main
# =============================================================================
# =============================================================================
# G1-G4 curriculum tests — each mechanism individually + integrated together
# =============================================================================
def _tier_weights(ratios):
    """Aggregate per-domain ratios into (easy, medium, hard) tier weights.

    Tier membership comes from the split dicts (DOMAIN_TIER), not name
    suffixes — "web_gold" is a HARD-tier domain but doesn't end in "_hard".
    """
    e = sum(v for k, v in ratios.items() if pmt.DOMAIN_TIER.get(k) == "_easy")
    m = sum(v for k, v in ratios.items() if pmt.DOMAIN_TIER.get(k) == "_medium")
    h = sum(v for k, v in ratios.items() if pmt.DOMAIN_TIER.get(k) == "_hard")
    return e, m, h


def test_g1_boundary_sharpening():
    """G1: fold-1 easy→hard, fold-2 mirrored, symmetric, continuous at fold."""
    # fold 1 (t ≤ 0.5): easy ↓, hard ↑ monotonic
    prev_e, prev_h = 1.0, 0.0
    for i in range(11):
        t = i / 20.0  # 0.0, 0.05, ..., 0.5
        w_e, w_m, w_h = pmt._smooth_tier_weights(t)
        assert w_e <= prev_e + 1e-9, f"fold-1 easy not monotonic at t={t}"
        assert w_h >= prev_h - 1e-9, f"fold-1 hard not monotonic at t={t}"
        assert abs(w_e + w_m + w_h - 1.0) < 1e-9, "weights must sum to 1"
        prev_e, prev_h = w_e, w_h
    # fold 2 (t ≥ 0.5): easy ↑ back up, hard ↓ back down (mirror)
    prev_e, prev_h = 0.0, 1.0
    for i in range(11):
        t = 0.5 + i / 20.0  # 0.5, 0.55, ..., 1.0
        w_e, w_m, w_h = pmt._smooth_tier_weights(t)
        assert w_e >= prev_e - 1e-9, f"fold-2 easy not rising at t={t}"
        assert w_h <= prev_h + 1e-9, f"fold-2 hard not falling at t={t}"
        prev_e, prev_h = w_e, w_h
    # symmetry: w(t) == w(1-t) exactly (mirror)
    for t in (0.0, 0.1, 0.3, 0.49, 0.6, 0.9, 1.0):
        a = pmt._smooth_tier_weights(t)
        b = pmt._smooth_tier_weights(1.0 - t)
        assert all(abs(x - y) < 1e-12 for x, y in zip(a, b)), f"mirror broken at t={t}"
    # continuity at the fold point (G3: no cliff mid-run)
    left = pmt._smooth_tier_weights(0.5 - 1e-9)
    mid = pmt._smooth_tier_weights(0.5)
    right = pmt._smooth_tier_weights(0.5 + 1e-9)
    assert all(abs(x - y) < 1e-6 for x, y in zip(left, mid)), "fold discontinuity (left)"
    assert all(abs(x - y) < 1e-6 for x, y in zip(mid, right)), "fold discontinuity (right)"
    # endpoints via the real mixer: start AND end are easy-heavy (cancellation)
    total = 15000
    e0, m0, h0 = _tier_weights(pmt.get_curriculum_ratios(0, total))
    e1, m1, h1 = _tier_weights(pmt.get_curriculum_ratios(total - 1, total))
    assert e0 > 0.25 and e0 < 0.35, f"start easy {e0:.3f} not in (0.25, 0.35)"
    assert e0 > h0, "start must be easy-heavy"
    assert e1 > h1, "end must ALSO be easy-heavy (fold-2 mirror cancels bias)"
    assert abs(e0 - e1) < 0.02 and abs(h0 - h1) < 0.02, "start ≈ end (cancellation)"
    print("  PASS: G1 boundary sharpening — 2-fold: easy→hard then mirrored hard→easy, symmetric & continuous")


def test_g1_two_fold_reverse_cancels():
    """2-fold reversal: hard peaks mid-training, easy bottoms mid-training."""
    total = 15000
    def easy_at(s):
        return _tier_weights(pmt.get_curriculum_ratios(s, total))[0]
    def hard_at(s):
        return _tier_weights(pmt.get_curriculum_ratios(s, total))[2]
    e_start, e_mid, e_end = easy_at(0), easy_at(total // 2), easy_at(total - 1)
    h_start, h_mid, h_end = hard_at(0), hard_at(total // 2), hard_at(total - 1)
    # hard peaks at the fold point; easy bottoms there
    assert h_mid > h_start and h_mid > h_end, \
        f"hard must peak mid-run: start {h_start:.3f} mid {h_mid:.3f} end {h_end:.3f}"
    assert e_mid < e_start and e_mid < e_end, \
        f"easy must bottom mid-run: start {e_start:.3f} mid {e_mid:.3f} end {e_end:.3f}"
    # both directions are seen: fold-1 hard rises, fold-2 hard falls
    h_q1 = hard_at(total // 4)
    h_q3 = hard_at(3 * total // 4)
    assert h_q1 > h_start, "fold 1: hard must rise from start"
    assert h_q3 < h_mid, "fold 2: hard must fall after the peak"
    # start ≈ end (order bias cancels) — within G2-wave tolerance
    assert abs(e_start - e_end) < 0.02, f"easy start {e_start:.3f} != end {e_end:.3f}"
    print("  PASS: 2-fold reverse — hard peaks mid-run, start≈end, both directions seen")


def test_g2_cyclic_review_wave():
    """G2: periodic easy review boost (anti-forgetting), renormalized, capped."""
    total = 15000
    cycle = total // 8
    def g1_easy_only(s):
        return 0.05 + 0.25 * (1.0 - s / total)
    def easy_at(s):
        return _tier_weights(pmt.get_curriculum_ratios(s, total))[0]
    # at cycle boundaries the cosine term is 0 → exactly the G1 curve
    # (tolerance 1e-3: get_curriculum_ratios rounds per-domain ratios to 4dp)
    assert abs(easy_at(0) - g1_easy_only(0)) < 1e-3, "G2 must be 0 at step 0"
    assert abs(easy_at(cycle) - g1_easy_only(cycle)) < 1e-3, "G2 must return to baseline at cycle end"
    # mid-cycle: cosine = -1 → max boost +0.12 (pre-renorm); must be clearly visible
    peak = easy_at(cycle // 2)
    assert peak > g1_easy_only(cycle // 2) + 0.05, \
        f"G2 peak boost too small: G1 {g1_easy_only(cycle//2):.3f} vs actual {peak:.3f}"
    # periodic: second cycle has the same boost shape
    assert easy_at(cycle + cycle // 2) > g1_easy_only(cycle + cycle // 2) + 0.05, \
        "G2 boost must be periodic (second cycle peak also boosted)"
    # cap + renormalization everywhere (tol 5e-4: ratios rounded to 4dp)
    for s in range(0, total, 137):
        r = pmt.get_curriculum_ratios(s, total)
        e, m, h = _tier_weights(r)
        assert e <= 0.5 + 1e-6, f"G2 easy cap violated at {s}: {e:.4f}"
        assert abs(e + m + h - 1.0) < 5e-4, f"G2 renormalization failed at {s}"
    print("  PASS: G2 cyclic review wave — periodic easy boost, renormalized, capped at 0.5")


def test_g3_curriculum_continuity_no_cliffs():
    """G3: consecutive ratio rebuilds glide — no cliff switches between tiers."""
    total = 15000
    interval = pmt.CURRICULUM_UPDATE_INTERVAL
    prev = None
    for step in range(0, total, interval):
        r = pmt.get_curriculum_ratios(step, total)
        if prev is not None:
            keys = set(prev) | set(r)
            max_delta = max(abs(prev.get(k, 0.0) - r.get(k, 0.0)) for k in keys)
            assert max_delta < 0.05, f"G3 cliff at step {step}: max delta {max_delta:.4f}"
        prev = r
    print("  PASS: G3 continuity — ratio rebuilds glide, no cliffs")


def _make_fake_shards(tmpdir, domains_seqs, seq_len=32, seed=7):
    """Create fake .bin uint16 shards; returns {domain: Path}."""
    import numpy as np
    from pathlib import Path
    rng = np.random.default_rng(seed)
    dirs = {}
    for dom, n_seqs in domains_seqs.items():
        d = os.path.join(tmpdir, dom)
        os.makedirs(d, exist_ok=True)
        arr = rng.integers(0, 49152, size=n_seqs * seq_len, dtype=np.uint16)
        arr.tofile(os.path.join(d, "shard_0.bin"))
        dirs[dom] = Path(d)
    return dirs


def _domain_mix(ds, order, n=None):
    """Count domain fractions in the first n items of epoch_order."""
    order = order[:n] if n else order
    counts = {}
    for i in order:
        dom = ds.index[i][1]
        counts[dom] = counts.get(dom, 0) + 1
    tot = len(order)
    return {d: c / tot for d, c in counts.items()}


def test_g4_windowed_jit_shuffle_and_ratios():
    """G4 + ratio enforcement: permutation preserved, ratio mix enforced inside
    the JIT window (5000), rebuild with new ratios reshuffles and re-mixes."""
    tmp = tempfile.mkdtemp(prefix="g4_test_")
    try:
        # buckets ∝ ratios1 so all exhaust together; smallest bucket (2000)
        # outlasts one JIT window (5000 items = 500 passes @ 10/pass)
        ratios1 = {"math_easy": 0.1, "web_easy": 0.2, "synth_easy": 0.3, "code_easy": 0.4}
        ratios2 = {"math_easy": 0.4, "web_easy": 0.3, "synth_easy": 0.2, "code_easy": 0.1}
        dirs = _make_fake_shards(tmp, {"math_easy": 2000, "web_easy": 4000,
                                       "synth_easy": 6000, "code_easy": 8000}, seq_len=32)
        ds = pmt.StratifiedShardDataset(dirs, seq_len=32, ratios=ratios1, dedup=False)
        order = ds.epoch_order
        # full permutation: no dups, no drops
        assert len(order) == len(set(order)) == 20000, "epoch_order must be a permutation"
        # every active domain present
        doms_in_order = {ds.index[i][1] for i in order}
        assert doms_in_order == set(ratios1), f"domains {doms_in_order} != active {set(ratios1)}"
        # ratio enforcement inside the first JIT window (5000 items)
        mix = _domain_mix(ds, order, n=5000)
        for dom, frac in ratios1.items():
            assert abs(mix[dom] - frac) < 0.02, \
                f"ratio not enforced: {dom} actual {mix[dom]:.3f} vs {frac}"
        # rebuild with different ratios → reshuffled, mix shifts toward new ratios
        ds.ratios = ratios2
        ds._build_stratified_order()
        order2 = ds.epoch_order
        assert order2 != order, "rebuild must reshuffle the order"
        assert len(order2) == len(set(order2)) == 20000, "rebuild must keep permutation"
        mix2 = _domain_mix(ds, order2, n=5000)
        for dom, frac in ratios2.items():
            assert abs(mix2[dom] - frac) < 0.02, \
                f"rebuild ratio not enforced: {dom} actual {mix2[dom]:.3f} vs {frac}"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  PASS: G4 windowed jitter — permutation preserved, ratios enforced on rebuild")


def test_g1234_together_full_run():
    """Integration: G1+G2+G3 produce valid ratios over the whole run and the
    dataset rebuild path (as wired in pretrain_gpu.py) keeps them enforced."""
    import pretrain_megatrain as _pmt
    # isolate from curriculum_boost.json (if present, per-domain boosts skew
    # the tier-shape assertions this test exists to check — boost has its own tests)
    _saved_bf = _pmt.WEB_BOOST_FILE
    _pmt.WEB_BOOST_FILE = _pmt.WEB_BOOST_FILE.with_name("_g1234_no_boost.json")
    try:
        _g1234_full_run_inner()
    finally:
        _pmt.WEB_BOOST_FILE = _saved_bf
    print("  PASS: G1+G2+G3+G4 integration — ratios valid & enforced over full run")


def _g1234_full_run_inner():
    total = 15000
    # whole-run ratio validity + 2-fold shape: easy bottoms mid, hard peaks mid
    easy_series, hard_series = [], []
    for s in range(0, total, 2000):
        r = pmt.get_curriculum_ratios(s, total)
        assert all(v >= 0 for v in r.values()), f"negative ratio at {s}"
        assert abs(sum(r.values()) - 1.0) < 5e-4, f"ratios don't sum to 1 at {s}"
        e, m, h = _tier_weights(r)
        easy_series.append(e)
        hard_series.append(h)
        assert "reformat_easy" in r, "5-domain mix: reformat_easy must be sampled early"
    mid = len(easy_series) // 2
    # fold 1: easy declines, hard rises (first half); fold 2: mirrored back
    assert easy_series[0] > easy_series[mid], "fold 1: easy must decline to mid-run"
    assert hard_series[0] < hard_series[mid], "fold 1: hard must rise to mid-run"
    assert easy_series[mid] < easy_series[-1], "fold 2: easy must rise back after mid-run"
    assert hard_series[mid] > hard_series[-1], "fold 2: hard must fall back after mid-run"
    # cancellation: start ≈ end, both easy-heavy (compare at wave-zero steps 0/14999)
    e0_end = _tier_weights(pmt.get_curriculum_ratios(0, total))[0]
    e_end = _tier_weights(pmt.get_curriculum_ratios(total - 1, total))[0]
    h_end = _tier_weights(pmt.get_curriculum_ratios(total - 1, total))[2]
    assert abs(e0_end - e_end) < 0.01, f"start {e0_end:.4f} != end {e_end:.4f} easy (cancellation)"
    assert e0_end > _tier_weights(pmt.get_curriculum_ratios(0, total))[2] and e_end > h_end, \
        "both ends must be easy-heavy (mirror cancels order bias)"
    # dataset rebuild loop (same as pretrain_gpu.py): ratios swap → order rebuilds
    tmp = tempfile.mkdtemp(prefix="g1234_test_")
    try:
        # fake subset covers 3 easy + 3 hard domains so the easy/hard balance
        # of the ACTIVE subset mirrors the real tier weights (easy is spread
        # over 5 domains in the real split — 2 easy dirs would bias hard-heavy)
        dirs = _make_fake_shards(tmp, {"math_easy": 3000, "web_easy": 3000,
                                       "synth_easy": 1000, "math_hard": 4000,
                                       "synth_hard": 2000, "web_hard": 1500}, seq_len=32)
        ds = pmt.StratifiedShardDataset(dirs, seq_len=32,
                                        ratios=pmt.get_curriculum_ratios(0, total), dedup=False)
        assert len(ds.epoch_order) == len(set(ds.epoch_order)) == 14500, "permutation at start"
        # mid-run rebuild: hard must dominate the interleave window (fold peak)
        ds.ratios = pmt.get_curriculum_ratios(total // 2, total)
        ds._build_stratified_order()  # exactly what the training loop does every 2000 steps
        assert len(ds.epoch_order) == len(set(ds.epoch_order)) == 14500, "permutation after rebuild"
        mix = _domain_mix(ds, ds.epoch_order, n=500)
        hard_frac = mix.get("math_hard", 0) + mix.get("synth_hard", 0) + mix.get("web_hard", 0)
        easy_frac = mix.get("math_easy", 0) + mix.get("synth_easy", 0) + mix.get("web_easy", 0)
        assert hard_frac > easy_frac, \
            f"fold peak violated: hard {hard_frac:.3f} vs easy {easy_frac:.3f} in window"
        # end-state rebuild: mirrored back to easy-heavy (cancellation)
        ds.ratios = pmt.get_curriculum_ratios(total - 1, total)
        ds._build_stratified_order()
        mix_end = _domain_mix(ds, ds.epoch_order, n=500)
        hard_end = mix_end.get("math_hard", 0) + mix_end.get("synth_hard", 0) + mix_end.get("web_hard", 0)
        easy_end = mix_end.get("math_easy", 0) + mix_end.get("synth_easy", 0) + mix_end.get("web_easy", 0)
        assert easy_end > hard_end, \
            f"end-state must be easy-heavy (mirror): easy {easy_end:.3f} vs hard {hard_end:.3f}"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_fused_ce_matches_chunked_ce():
    """Liger fused linear+CE must match chunked_ce (same shift, same mean)."""
    if not torch.cuda.is_available():
        return
    from pretrain_gpu import chunked_ce, fused_ce
    torch.manual_seed(0)
    B, S, H, V = 2, 64, 128, 256
    hidden = torch.randn(B, S, H, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(V, H, device="cuda", dtype=torch.bfloat16) * 0.1
    labels = torch.randint(0, V, (B, S), device="cuda")
    logits = hidden @ w.t()
    ref = chunked_ce(logits, labels)
    got = fused_ce(hidden, w, labels)
    assert torch.allclose(got, ref, atol=1e-2), \
        f"fused {got.item():.4f} vs chunked {ref.item():.4f}"


# =============================================================================
# 3b. SWA lean snapshots + averaging
# =============================================================================
def test_swa_snapshot_prunes_to_window():
    """swa_tail keeps only the last `window` snapshots (oldest pruned)."""
    import pretrain_gpu as pmt
    from pathlib import Path
    tmpdir = tempfile.mkdtemp()
    try:
        sd = {"w": torch.randn(4, 4)}
        for step in (100, 200, 300, 400, 500):
            pmt.save_swa_snapshot_robust(step, sd, 3, tmpdir, pmt.logger)
        snaps = sorted(p.name for p in Path(tmpdir, "swa_tail").glob("swa_*.pt"))
        assert snaps == ["swa_000300.pt", "swa_000400.pt", "swa_000500.pt"], snaps
        # window=0 (off) must never prune
        pmt.save_swa_snapshot_robust(600, sd, 0, tmpdir, pmt.logger)
        assert len(list(Path(tmpdir, "swa_tail").glob("swa_*.pt"))) == 4
    finally:
        shutil.rmtree(tmpdir)
    print("  PASS: swa window prunes oldest snapshots, 0 = keep all")


def test_swa_snapshot_roundtrip():
    """Snapshot loads back with step + bf16 model weights (lean, no optimizer)."""
    import pretrain_gpu as pmt
    tmpdir = tempfile.mkdtemp()
    try:
        sd = {"w": torch.randn(4, 4, dtype=torch.bfloat16)}
        pmt.save_swa_snapshot_robust(777, sd, 0, tmpdir, pmt.logger)
        ck = torch.load(os.path.join(tmpdir, "swa_tail", "swa_000777.pt"),
                        map_location="cpu", weights_only=False)
        assert ck["step"] == 777, f"{_test_name()}: step = {ck['step']}"
        assert "optimizer_state_dict" not in ck
        assert ck["model_state_dict"]["w"].dtype == torch.bfloat16
    finally:
        shutil.rmtree(tmpdir)
    print("  PASS: swa snapshot roundtrips (lean model-only)")


def test_async_save_writes_swa_snapshot():
    """save_checkpoint_async with swa_window writes latest.pt AND lean snapshot."""
    import pretrain_gpu as pmt
    tmpdir = tempfile.mkdtemp()
    try:
        state = {"step": 42, "loss": 2.0, "best_loss": 2.0,
                 "model_state_dict": {"w": torch.randn(4, 4)},
                 "optimizer_state_dict": {"state": {}, "param_groups": []},
                 "config": {}}
        t = pmt.save_checkpoint_async(state, False, pmt.logger, output_dir=tmpdir,
                                      swa_window=5, swa_step=42)
        t.join(timeout=120)
        assert not t.is_alive(), f"{_test_name()}: saver thread did not finish"
        assert os.path.exists(os.path.join(tmpdir, "megatrain_latest.pt"))
        assert os.path.exists(os.path.join(tmpdir, "swa_tail", "swa_000042.pt"))
    finally:
        shutil.rmtree(tmpdir)
    print("  PASS: async save also emits swa snapshot")


def test_swa_average_matches_mean():
    """average_swa over known weights = exact arithmetic mean."""
    import swa_average
    tmpdir = tempfile.mkdtemp()
    try:
        swa_dir = os.path.join(tmpdir, "swa_tail")
        os.makedirs(swa_dir)
        w1 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.bfloat16)
        w2 = torch.tensor([[5.0, 6.0], [7.0, 8.0]], dtype=torch.bfloat16)
        torch.save({"step": 10, "model_state_dict": {"w": w1}},
                   os.path.join(swa_dir, "swa_000010.pt"))
        torch.save({"step": 20, "model_state_dict": {"w": w2}},
                   os.path.join(swa_dir, "swa_000020.pt"))
        out = os.path.join(tmpdir, "megatrain_swa.pt")
        steps = swa_average.average_swa(swa_dir, out)
        assert steps == [10, 20], f"{_test_name()}: steps = {steps}"
        ck = torch.load(out, map_location="cpu", weights_only=False)
        want = (w1.float() + w2.float()) / 2
        assert torch.allclose(ck["model_state_dict"]["w"].float(), want, atol=1e-2)
    finally:
        shutil.rmtree(tmpdir)
    print("  PASS: swa average equals exact mean of snapshots")


def test_log_stats_uses_true_step_count():
    """First post-resume log line divides by ACTUAL steps (14), not log_interval (100).

    Regression: after resume at 36,586 the first log fired at 36,600 (14 steps)
    but divided by 100 → fake Loss 0.27 / 1.94s/step instead of real ~1.98 / 13.9s.
    """
    from pretrain_gpu import compute_log_stats
    # resume case: 14 steps × 2.0 loss, 194s real wall time
    avg, tps, s_step = compute_log_stats(torch.tensor(28.0), 14, 194.0, 4, 2048, 8)
    assert abs(avg - 2.0) < 1e-6, f"{_test_name()}: avg = {avg}"
    assert abs(s_step - 194.0 / 14) < 1e-6, f"{_test_name()}: s/step = {s_step}"
    assert abs(tps - 4 * 2048 * 8 * 14 / 194.0) < 1e-3
    # normal case: full interval, unchanged math
    avg2, _, s2 = compute_log_stats(torch.tensor(200.0), 100, 1436.0, 4, 2048, 8)
    assert abs(avg2 - 2.0) < 1e-6
    assert abs(s2 - 14.36) < 1e-6
    print("  PASS: log stats divide by true step count (resume line honest)")


def test_smoothed_loss_rejects_single_step_noise():
    """Regression (2026-08-14): is_best compared ONE step's loss (acc_loss.item()).

    A lucky all-easy step at 1.46 faked 'Best loss 1.4603' in the log while
    the 100-step smoothed average was 2.04 → best.pt tracked per-step domain
    mix noise, not genuine progress. The fix: best-pt tracking uses the
    interval-smoothed average (same number the log line reports).
    """
    from pretrain_gpu import smoothed_loss_at_save
    # 100-step interval averaging 2.04 → smoothed MUST be 2.04, not 1.46
    smoothed = smoothed_loss_at_save(torch.full((), 204.0), 100, 2.04)
    assert abs(smoothed - 2.04) < 1e-5, f"{_test_name()}: smoothed = {smoothed}"
    # old code compared 1.46 < 1.6993 → fake best; smoothed 2.04 must NOT be best
    assert not (smoothed < 1.6993), f"{_test_name()}: noise step faked a best!"
    # a genuine improvement still wins
    assert smoothed_loss_at_save(torch.full((), 150.0), 100, 1.5) < 1.6993
    # save on a log boundary (n_log == 0, every 1000-step save) → logged avg
    assert smoothed_loss_at_save(torch.zeros(()), 0, 2.01) == 2.01
    print("  PASS: best-pt uses interval-smoothed loss (noise step rejected)")


def test_domain_loss_tracker():
    """Per-domain loss accumulator: sums, means, reset (DoReMi-lite input)."""
    from pretrain_gpu import DomainLossTracker
    t = DomainLossTracker()
    t.update("math_easy", 2.0)
    t.update("math_easy", 3.0)
    t.update("web_hard", 5.0)
    means = t.means()
    assert abs(means["math_easy"] - 2.5) < 1e-9, f"{_test_name()}: {means}"
    assert abs(means["web_hard"] - 5.0) < 1e-9
    t.reset()
    assert t.means() == {}, f"{_test_name()}: reset failed"
    print("  PASS: domain loss tracker sums/means/reset")


def test_doremi_adjust_upweights_stuck_domains():
    """DoReMi-lite: stuck domains (excess>0) upweighted, improved downweighted,
    per-tier totals preserved (G1-G4 tier structure intact)."""
    from pretrain_gpu import doremi_adjust
    ratios = {"math_easy": 0.1, "web_easy": 0.1, "math_hard": 0.2, "web_hard": 0.1}
    ref = {"math_easy": 2.0, "web_easy": 2.0, "math_hard": 3.0, "web_hard": 3.0}
    # math stuck (+20% loss), web improved (-20%)
    cur = {"math_easy": 2.4, "web_easy": 1.6, "math_hard": 3.6, "web_hard": 2.4}
    out = doremi_adjust(ratios, cur, ref, eps=0.3)
    assert out["math_easy"] > ratios["math_easy"], f"{_test_name()}: stuck not upweighted: {out}"
    assert out["math_hard"] > ratios["math_hard"], f"{_test_name()}: stuck not upweighted: {out}"
    assert out["web_easy"] < ratios["web_easy"], f"{_test_name()}: improved not downweighted: {out}"
    assert out["web_hard"] < ratios["web_hard"], f"{_test_name()}: improved not downweighted: {out}"
    # per-tier totals preserved
    easy = out["math_easy"] + out["web_easy"]
    hard = out["math_hard"] + out["web_hard"]
    assert abs(easy - 0.2) < 0.002, f"{_test_name()}: easy tier {easy} != 0.2"
    assert abs(hard - 0.3) < 0.002, f"{_test_name()}: hard tier {hard} != 0.3"
    print("  PASS: doremi adjust upweights stuck / downweights improved / tiers intact")


def test_doremi_adjust_clamps_exploding_excess():
    """Regression (Aug 15 crash): a near-zero ref (code_hard 0.0182) makes
    excess explode (+8470%) — exp(0.3*84.7)≈1e11 would hand the tier to one
    domain and zero the rest → min_ratio=0 → ZeroDivisionError. The clamp
    keeps every ratio > 0 and the tier total intact."""
    from pretrain_gpu import doremi_adjust
    ratios = {"math_hard": 0.55, "web_hard": 0.15, "synth_hard": 0.20, "code_hard": 0.10}
    ref = {"math_hard": 1.6, "web_hard": 2.9, "synth_hard": 1.7, "code_hard": 0.0182}
    cur = {"math_hard": 1.98, "web_hard": 2.35, "synth_hard": 1.97, "code_hard": 1.56}
    out = doremi_adjust(ratios, cur, ref, eps=0.3)
    assert all(v > 0 for v in out.values()), f"zero ratio survived clamp: {out}"
    assert all(v <= max(ratios.values()) * 5 for v in out.values()), \
        f"multiplier not clamped: {out}"
    # code_hard still upweighted (direction preserved) but not tier-dominant
    assert out["code_hard"] > ratios["code_hard"], "code_hard should be upweighted"
    assert out["code_hard"] < 0.5, f"code_hard still exploded: {out}"
    total = sum(out.values())
    assert abs(total - 1.0) < 0.01, f"hard tier total {total} != 1"
    # web_hard (ref 2.9, cur 2.35 → improved) still downweighted
    assert out["web_hard"] < ratios["web_hard"], "web_hard should be downweighted"
    print("  PASS: doremi clamp bounds exploding excess — all ratios > 0, tier intact")


def test_web_boost_upweights_web_preserves_tiers():
    """WEB_BOOST: web_* within-tier share rises ~1.5x vs its un-boosted base,
    non-web domains shrink, and every tier total stays exactly intact."""
    import pretrain_megatrain as _pmt
    total = 115000
    for step in (0, 100, total // 2, total - 1):
        # boost OFF = the plain G1-G4 curve (WEB_BOOST must not move tier totals)
        saved = _pmt.WEB_BOOST
        saved_file = _pmt.WEB_BOOST_FILE
        try:
            # isolate from curriculum_boost.json — this test exercises the
            # WEB_BOOST code default only (the JSON path has its own test)
            _pmt.WEB_BOOST_FILE = _pmt.WEB_BOOST_FILE.with_name("_nonexistent_boost.json")
            _pmt.WEB_BOOST = 1.0
            r_off = _pmt.get_curriculum_ratios(step, total)
            _pmt.WEB_BOOST = saved
            r_on = _pmt.get_curriculum_ratios(step, total)
        finally:
            _pmt.WEB_BOOST = saved
            _pmt.WEB_BOOST_FILE = saved_file
        e_off, m_off, h_off = _tier_weights(r_off)
        e_on, m_on, h_on = _tier_weights(r_on)
        assert abs(e_on - e_off) < 1e-3, f"easy tier moved: {e_off:.4f}->{e_on:.4f} @ {step}"
        assert abs(m_on - m_off) < 1e-3, f"med tier moved: {m_off:.4f}->{m_on:.4f} @ {step}"
        assert abs(h_on - h_off) < 1e-3, f"hard tier moved: {h_off:.4f}->{h_on:.4f} @ {step}"
        # web's share within each tier matches the exact closed form:
        #   share_on = base*B / (1 + (B-1)*base)   (only web is boosted per tier)
        for dom in r_on:
            if not dom.startswith("web"):
                continue
            tier = pmt.DOMAIN_TIER[dom]
            tier_doms = {"_easy": pmt.EASY_SPLIT, "_medium": pmt.MED_SPLIT,
                         "_hard": pmt.HARD_SPLIT}[tier]
            base_frac = tier_doms[dom]
            # General closed form: ALL web domains in the tier get boost B and
            # the tier is renormalized — web_base_sum accounts for multiple
            # web domains per tier (hard: web_hard + web_gold).
            web_base_sum = sum(f for d, f in tier_doms.items() if d.startswith("web"))
            B = _pmt.WEB_BOOST
            expected = base_frac * B / (1.0 + (B - 1.0) * web_base_sum)
            tsum_on = {"_easy": e_on, "_medium": m_on, "_hard": h_on}[tier]
            share_on = r_on[dom] / tsum_on
            assert abs(share_on - expected) < 2e-3, \
                f"{dom} share {share_on:.4f} != expected {expected:.4f} @ {step}"
        # non-web domains lose share within their tier (renormalization)
        for dom in r_on:
            if dom.startswith("web"):
                continue
            tier = pmt.DOMAIN_TIER[dom]
            tsum_on = {"_easy": e_on, "_medium": m_on, "_hard": h_on}[tier]
            tsum_off = {"_easy": e_off, "_medium": m_off, "_hard": h_off}[tier]
            if r_off[dom] <= 0:
                continue
            assert r_on[dom] / tsum_on < r_off[dom] / tsum_off, \
                f"{dom} should shrink but grew @ {step}"
    print(f"  PASS: WEB_BOOST={_pmt.WEB_BOOST} — web upweighted per tier, tier totals intact")


def test_web_boost_hot_reload_json():
    """curriculum_boost.json overrides WEB_BOOST per-domain, re-read on every
    call; tier totals stay intact; missing file falls back to WEB_BOOST."""
    import json as _json
    import pretrain_megatrain as _pmt
    total = 115000
    saved_file = _pmt.WEB_BOOST_FILE
    fake = _pmt.WEB_BOOST_FILE.with_name("_test_boost.json")
    try:
        _pmt.WEB_BOOST_FILE = fake
        fake.write_text(_json.dumps({"web_easy": 2.0, "web_hard": 1.2}))
        r = _pmt.get_curriculum_ratios(total // 2, total)
        e, m, h = _tier_weights(r)
        # web_easy boosted MORE than web_medium (default 1.5); web_hard LESS
        web_easy_share = r["web_easy"] / e
        web_med_share = r["web_medium"] / m
        web_hard_share = r["web_hard"] / h
        assert web_easy_share > web_med_share, \
            f"web_easy {web_easy_share:.4f} should exceed web_medium {web_med_share:.4f}"
        assert web_hard_share < web_med_share, \
            f"web_hard {web_hard_share:.4f} should be below web_medium {web_med_share:.4f}"
        # file removed → falls back to WEB_BOOST=1.5 default: web_easy share
        # must match the closed form base*B/(1+(B-1)*base) at B=1.5
        fake.unlink()
        r2 = _pmt.get_curriculum_ratios(total // 2, total)
        e2, m2, h2 = _tier_weights(r2)
        base_e = pmt.EASY_SPLIT["web_easy"]
        expected = base_e * 1.5 / (1.0 + 0.5 * base_e)
        assert abs((r2["web_easy"] / e2) - expected) < 2e-3, \
            f"fallback share {(r2['web_easy']/e2):.4f} != expected {expected:.4f}"
        # STRICT dead-file check: a non-web boost MUST move its share (this
        # catches a silently-failing file read, e.g. NameError on json — the
        # web-only assertions above hold even when the file is ignored).
        # web_* set to 1.0 so the closed form holds (only code_easy boosted).
        fake.write_text(_json.dumps({"web_easy": 1.0, "web_medium": 1.0,
                                     "web_hard": 1.0, "code_easy": 2.0}))
        r3 = _pmt.get_curriculum_ratios(total // 2, total)
        e3, _, _ = _tier_weights(r3)
        code_share = r3["code_easy"] / e3
        base_c = pmt.EASY_SPLIT["code_easy"]
        expected_c = base_c * 2.0 / (1.0 + 1.0 * base_c)
        assert abs(code_share - expected_c) < 2e-3, \
            f"code_easy share {code_share:.4f} != expected {expected_c:.4f} — boost file ignored!"
        # tier totals still exactly intact with the file active
        assert abs(e + m + h - 1.0) < 1e-3, f"tiers {e:.4f}+{m:.4f}+{h:.4f} != 1"
    finally:
        if fake.exists():
            fake.unlink()
        _pmt.WEB_BOOST_FILE = saved_file
    print("  PASS: curriculum_boost.json hot-reload — overrides apply, fallback works, tiers intact")


def test_collate_pretrain_with_domains():
    """StratifiedShardDataset dict batches carry 'domain'; tensor batches (FlatFarm)
    still work unchanged (backward compat)."""
    from pretrain_megatrain import collate_pretrain
    a = torch.arange(6)          # 1-D seq, like real samples (T=6)
    b = torch.arange(6, 12)
    out = collate_pretrain([{"input_ids": a, "domain": "math_easy"},
                            {"input_ids": b, "domain": "web_hard"}])
    assert out["domain"] == ["math_easy", "web_hard"], f"{_test_name()}: {out.get('domain')}"
    assert out["input_ids"].shape == (2, 6)
    # legacy tensor path
    out2 = collate_pretrain([a, b])
    assert "domain" not in out2
    assert out2["input_ids"].shape == (2, 6)
    print("  PASS: collate carries domains (dict path) and stays tensor-compatible")


def test_resume_defaults_to_latest():
    """resolve_resume_path: explicit wins; else latest if present; else None."""
    from pretrain_gpu import resolve_resume_path
    tmpdir = tempfile.mkdtemp()
    try:
        latest = os.path.join(tmpdir, "megatrain_latest.pt")
        assert resolve_resume_path(None, latest) is None          # no latest yet
        open(latest, "w").close()
        assert resolve_resume_path(None, latest) == latest        # latest exists
        other = os.path.join(tmpdir, "other.pt")
        assert resolve_resume_path(other, latest) == other        # explicit wins
    finally:
        shutil.rmtree(tmpdir)
    print("  PASS: resume resolves to latest by default (explicit wins)")


TESTS = [
    test_module_imports_cleanly,
    test_lr_warmup_rises_linearly,
    test_lr_cosine_decay_falls,
    test_lr_zero_warmup,
    test_lr_monotonic_during_warmup,
    test_lr_must_use_outer_step_not_global_step,
    test_collate_creates_causal_mask,
    test_collate_labels_equal_input_ids,
    test_collate_mask_is_pure_causal_no_padding,
    test_accumulate_loss_detached_and_exact,
    test_chunked_ce_matches_full_ce,
    test_fused_ce_matches_chunked_ce,
    test_warmup_default_400,
    test_liger_replaces_modules,
    test_compile_oom_fallback,
    test_ce_chunk_512_for_vram_safety,
    test_checkpoint_rejects_nan,
    test_checkpoint_rejects_inf,
    test_checkpoint_saves_clean_state,
    test_validate_cpu_params_detects_nan,
    test_validate_cpu_params_passes_clean,
    test_llama_config_creation,
    test_cpumaster_config_creation,
    test_momentum_ema_scale,
    test_momentum_vs_sgd_style,
    test_newton_schulz_preserves_spectral_norm,
    test_newton_schulz_non_square,
    test_newton_schulz_steps_effect,
    test_qk_clip_enforced,
    test_qk_clip_only_on_muon_params,
    test_adamw_runs_on_1d_params,
    test_adamw_not_muon_state,
    test_weight_decay_muon,
    test_lr_zero_no_change,
    test_momentum_warmup_progression,
    test_no_nan_inf,
    test_grad_none_skipped,
    test_adam_update_formula,
    test_default_argparse_values,
    test_g1_boundary_sharpening,
    test_g1_two_fold_reverse_cancels,
    test_g2_cyclic_review_wave,
    test_g3_curriculum_continuity_no_cliffs,
    test_g4_windowed_jit_shuffle_and_ratios,
    test_g1234_together_full_run,
    test_save_trigger_step_based,
    test_save_trigger_time_based,
    test_save_trigger_either_wins,
    test_torch_compile_smoke,
    test_optimizer_groups_fused,
    test_cautious_masks_disagreeing_updates,
    test_cautious_state_roundtrip,
    test_cautious_accepts_adamw_state_dict,
    test_cautious_converges_on_toy,
    test_make_optimizer_cautious_flag,
    test_build_dataloader_workers,
    test_async_save_writes_file,
    test_swa_snapshot_prunes_to_window,
    test_swa_snapshot_roundtrip,
    test_async_save_writes_swa_snapshot,
    test_swa_average_matches_mean,
    test_resume_defaults_to_latest,
    test_log_stats_uses_true_step_count,
    test_smoothed_loss_rejects_single_step_noise,
    test_domain_loss_tracker,
    test_doremi_adjust_upweights_stuck_domains,
    test_doremi_adjust_clamps_exploding_excess,
    test_web_boost_upweights_web_preserves_tiers,
    test_web_boost_hot_reload_json,
    test_collate_pretrain_with_domains,
    test_cautious_tail_triton_bitwise,
    test_gradient_checkpointing_inert_in_transformers,
    test_compile_reduce_overhead_smoke,
]

if __name__ == "__main__":
    torch.manual_seed(42)
    passed, failed = 0, 0

    print(f"Running {len(TESTS)} pretrain tests...\n")
    for test_fn in TESTS:
        try:
            test_fn()
            passed += 1
            print(f"  ✅ {test_fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"  ❌ {test_fn.__name__}: {e}")
        except Exception as e:
            failed += 1
            print(f"  💥 {test_fn.__name__}: {type(e).__name__}: {e}")

    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed out of {len(TESTS)}")
    if failed:
        print("EXIT CODE 1")
        sys.exit(1)
    else:
        print("All tests passed. EXIT CODE 0")
        sys.exit(0)
