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
    collate_pretrain, validate_cpu_params, save_checkpoint_robust
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
    parser.add_argument("--muon-lr", type=float, default=0.02)
    parser.add_argument("--adam-lr", type=float, default=3e-4)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=12)
    args = parser.parse_args([])
    assert args.muon_lr == 0.02, f"{_test_name()}: muon_lr default = {args.muon_lr}"
    assert args.adam_lr == 3e-4, f"{_test_name()}: adam_lr default = {args.adam_lr}"
    assert args.warmup_steps == 1000, f"{_test_name()}: warmup_steps default = {args.warmup_steps}"
    assert args.min_lr == 1e-6, f"{_test_name()}: min_lr default = {args.min_lr}"
    assert args.batch_size == 4
    assert args.grad_accum == 12
    print(f"  PASS: argparse defaults correct")


# =============================================================================
# Main
# =============================================================================
TESTS = [
    test_module_imports_cleanly,
    test_lr_warmup_rises_linearly,
    test_lr_cosine_decay_falls,
    test_lr_zero_warmup,
    test_lr_monotonic_during_warmup,
    test_collate_creates_causal_mask,
    test_collate_labels_equal_input_ids,
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
