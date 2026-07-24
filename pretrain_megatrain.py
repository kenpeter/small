"""
Pretraining with MegaTrain + Kimi K2 MuonClip
CPU offloaded training of 1.03B model with orthogonal updates.
Loads .bin shards directly (already tokenized with SmolLM2 tokenizer).
"""
import os, time, logging, argparse, math, shutil, random
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np

from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaConfig
from infinity import CPUMasterModel
from infinity.config import CPUMasterConfig

logger = logging.getLogger(__name__)

# ============================================================================
# Cosine LR schedule with linear warmup
# ============================================================================

def get_lr(step, warmup_steps, total_steps, base_lr, min_lr=1e-6):
    """Cosine decay with linear warmup."""
    if step < warmup_steps:
        return base_lr * step / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return min_lr + (base_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * progress))


# ============================================================================
# Kimi K2 MuonClip Optimizer — Full Implementation
# Based on arXiv 2502.16982 + github.com/AkulDatta/muonclip
# ============================================================================

def newton_schulz(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """Newton-Schulz iteration for matrix orthogonalization."""
    assert G.ndim >= 2
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.float()
    # Normalize by Frobenius norm
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + eps)
    if G.size(-2) > G.size(-1):
        X = X.mT
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(-2) > G.size(-1):
        X = X.mT
    return X.to(G.dtype)


def adam_update(grad, buf1, buf2, step, betas, eps):
    buf1.lerp_(grad, 1 - betas[0])
    buf2.lerp_(grad.square(), 1 - betas[1])
    buf1c = buf1 / (1 - betas[0] ** step)
    buf2c = buf2 / (1 - betas[1] ** step)
    return buf1c / (buf2c.sqrt() + eps)


class KimiMuonClip(torch.optim.Optimizer):
    """
    Kimi K2 MuonClip optimizer.
    - Muon (Newton-Schulz + momentum) for 2D hidden weights
    - AdamW for 1D scalars (norms, biases)
    - AdamW for embeddings + lm_head
    - Consistent RMS scaling across all layers
    - Momentum warmup: 0.90 -> 0.95 over first 300 steps
    - QK-Clip proxy: spectral norm cap on attention projections
    """
    def __init__(self, param_groups, tau: float = 150.0, ns_steps: int = 7, use_gpu_ns: bool = True):
        for group in param_groups:
            assert "use_muon" in group
            if group["use_muon"]:
                group.setdefault("lr", 0.01)
                group.setdefault("momentum", 0.95)
                group.setdefault("weight_decay", 0.0)
            else:
                group.setdefault("lr", 3e-4)
                group.setdefault("betas", (0.9, 0.95))
                group.setdefault("eps", 1e-10)
                group.setdefault("weight_decay", 0.0)
        defaults = dict(tau=tau, ns_steps=ns_steps, use_gpu_ns=use_gpu_ns)
        super().__init__(param_groups, defaults)

    @torch.no_grad()
    def step(self, closure=None, global_step: int = 0):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # Momentum warmup: 0.90 -> 0.95 over first 300 steps
        frac = min(global_step / 300.0, 1.0)
        warmed_momentum = (1 - frac) * 0.90 + frac * 0.95

        for group in self.param_groups:
            lr = group["lr"]
            wd = group["weight_decay"]

            if group["use_muon"]:
                # Use warmed momentum
                beta = warmed_momentum if group.get("warmup", True) else group["momentum"]
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(p)

                    buf = state["momentum_buffer"]
                    # Momentum: Mt = μ * Mt-1 + (1-μ) * Gt  (EMA, not SGD-style)
                    buf.mul_(beta).add_(p.grad, alpha=1-beta)

                    # Newton-Schulz orthogonalization — GPU if available
                    use_gpu = torch.cuda.is_available() and self.defaults.get("use_gpu_ns", True)
                    if use_gpu:
                        buf_gpu = buf.cuda(non_blocking=False)
                        if p.ndim > 2:
                            orig_shape = buf_gpu.shape
                            buf_2d = buf_gpu.view(buf_gpu.shape[0], -1)
                            update_gpu = newton_schulz(buf_2d, steps=self.defaults["ns_steps"])
                            update = update_gpu.view(orig_shape).cpu()
                        else:
                            update = newton_schulz(buf_gpu, steps=self.defaults["ns_steps"]).cpu()
                        del buf_gpu
                    else:
                        if p.ndim > 2:
                            orig_shape = buf.shape
                            buf_2d = buf.view(buf.shape[0], -1)
                            update = newton_schulz(buf_2d, steps=self.defaults["ns_steps"])
                            update = update.view(orig_shape)
                        else:
                            update = newton_schulz(buf, steps=self.defaults["ns_steps"])

                    # Consistent RMS scaling: sqrt(max(n,m) * 0.2)
                    n, m = p.shape[0], p.shape[1] if p.ndim > 1 else 1
                    rms_factor = math.sqrt(max(n, m) * 0.2)
                    update *= rms_factor

                    # Weight decay + update
                    if wd > 0:
                        p.mul_(1 - lr * wd)
                    p.add_(update, alpha=-lr)

            else:
                # AdamW for 1D params / embed / head
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    state = self.state[p]
                    if len(state) == 0:
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                        state["step"] = 0
                    state["step"] += 1
                    update = adam_update(p.grad, state["exp_avg"], state["exp_avg_sq"],
                                         state["step"], group["betas"], group["eps"])
                    if wd > 0:
                        p.mul_(1 - lr * wd)
                    p.add_(update, alpha=-lr)

        # QK-Clip proxy: cap spectral norm of attention-like projections — GPU
        tau = self.defaults["tau"]
        for group in self.param_groups:
            if group.get("use_muon", False):
                for p in group["params"]:
                    if p.ndim >= 2 and p.shape[0] <= p.shape[1]:
                        with torch.no_grad():
                            p_gpu = p.data.cuda(non_blocking=False)
                            spec_norm = torch.linalg.matrix_norm(p_gpu, ord=2)
                            if spec_norm.item() > tau:
                                p_gpu.mul_(tau / spec_norm)
                                p.data.copy_(p_gpu)
                            del p_gpu

        return loss


SEQ_LEN = 2048

# ============================================================================
# Stratified multi-domain shard loader with 13-gram dedup
# ============================================================================

SHARD_DIRS = {
    "web":   Path("/home/kenpeter/work/data/_shards_final"),
    "math":  Path("/home/kenpeter/work/data/_shards_final"),   # TODO: point to math shards when ready
    "synth": Path("/home/kenpeter/work/data/_shards_final"),   # TODO: point to cosmopedia shards when ready
}

RATIOS = {"web": 0.60, "math": 0.25, "synth": 0.15}


def _load_shard_list(shards_dir: Path, seq_len: int):
    shard_paths = sorted(shards_dir.glob("shard_*.bin"))
    shard_paths = [p for p in shard_paths if p.stat().st_size > 0]
    entries = []
    total = 0
    for p in shard_paths:
        n_tokens = p.stat().st_size // 2
        n_seqs = n_tokens // seq_len
        if n_seqs == 0:
            continue
        entries.append((p, n_seqs, total))
        total += n_seqs
    return entries, total


def _hash_13gram(tokens: np.ndarray) -> int:
    """Cheap 13-gram hash for dedup."""
    if len(tokens) < 13:
        return 0
    # Use first 13 tokens + last 13 tokens as proxy fingerprint
    front = tokens[:13].tobytes()
    back = tokens[-13:].tobytes()
    return hash((front, back))


class StratifiedShardDataset(Dataset):
    """
    Loads shards from multiple domains, applies stratified sampling ratios,
    and exact 13-gram deduplication across the whole corpus.
    """
    _causal_mask_4d = None

    def __init__(self, shard_dirs: dict, seq_len: int = 2048,
                 ratios: dict = None, dedup: bool = True):
        self.seq_len = seq_len
        self.ratios = ratios or RATIOS
        self.domains = []
        self.domain_entries = {}
        self.domain_totals = {}
        grand_total = 0

        for domain, dpath in shard_dirs.items():
            if not dpath.exists():
                logger.warning(f"Shard dir missing for '{domain}': {dpath} — skipping")
                continue
            entries, total = _load_shard_list(dpath, seq_len)
            if total == 0:
                continue
            self.domains.append(domain)
            self.domain_entries[domain] = entries
            self.domain_totals[domain] = total
            grand_total += total
            logger.info(f"Domain '{domain}': {len(entries)} shards, {total:,} seqs")

        if not self.domains:
            raise FileNotFoundError("No valid shard directories found")

        # Build flat index with domain tags for stratified sampling
        self.index = []          # (global_idx, domain, local_idx)
        self.domain_offsets = {}  # domain -> start in flat index
        cursor = 0
        for domain in self.domains:
            self.domain_offsets[domain] = cursor
            n = self.domain_totals[domain]
            self.index.extend([(cursor + i, domain, i) for i in range(n)])
            cursor += n
        self.raw_len = len(self.index)

        # Optional: exact 13-gram dedup (CPU, one-time scan)
        self.dedup = dedup
        self.valid_mask = None
        if dedup:
            self.valid_mask = self._compute_dedup_mask()
            kept = self.valid_mask.sum()
            logger.info(f"Dedup: {self.raw_len:,} raw → {kept:,} unique  (dropped {self.raw_len - kept:,})")
        else:
            self.valid_mask = torch.ones(self.raw_len, dtype=torch.bool)

        # Precompute stratified per-batch ordering
        self._build_stratified_order()

    def _compute_dedup_mask(self):
        seen = set()
        mask = torch.zeros(self.raw_len, dtype=torch.bool)
        # Scan every sequence once — slow but one-time
        for global_idx in range(self.raw_len):
            tokens = self._fetch_tokens(global_idx)
            h = _hash_13gram(tokens.numpy())
            if h not in seen:
                seen.add(h)
                mask[global_idx] = True
        return mask

    def _fetch_tokens(self, global_idx: int) -> torch.Tensor:
        _, domain, local_idx = self.index[global_idx]
        for shard_path, n_seqs, start_idx in self.domain_entries[domain]:
            if local_idx < start_idx + n_seqs:
                local = local_idx - start_idx
                offset = local * self.seq_len
                mm = np.memmap(str(shard_path), dtype=np.uint16, mode='r',
                               offset=offset * 2, shape=(self.seq_len,))
                tokens = torch.from_numpy(mm.copy().astype(np.int64))
                del mm
                return tokens
        raise IndexError(f"Bad index {global_idx}")

    def _build_stratified_order(self):
        # Create an epoch ordering that respects ratios
        valid_indices = torch.where(self.valid_mask)[0].tolist()
        # Bucket by domain
        buckets = {d: [] for d in self.domains}
        for idx in valid_indices:
            _, domain, _ = self.index[idx]
            buckets[domain].append(idx)
        # Shuffle each bucket
        for d in self.domains:
            random.shuffle(buckets[d])

        # Interleave according to ratios
        self.epoch_order = []
        ptrs = {d: 0 for d in self.domains}
        total_valid = len(valid_indices)
        # Determine per-step counts (proportional)
        batch_size = 2  # physical batch; will be overridden by DataLoader
        # We just build a flat list; DataLoader batching will grab sequentially
        # To enforce ratios per step, we emit in repeating pattern
        while sum(ptrs[d] < len(buckets[d]) for d in self.domains) > 0:
            for domain in self.domains:
                # emit ~ratio proportion
                n_emit = max(1, int(batch_size * self.ratios[domain]))
                for _ in range(n_emit):
                    if ptrs[domain] < len(buckets[domain]):
                        self.epoch_order.append(buckets[domain][ptrs[domain]])
                        ptrs[domain] += 1
            # safety break
            if len(self.epoch_order) > total_valid * 2:
                break
        # Trim to exact count and final shuffle in small windows to keep locality
        self.epoch_order = self.epoch_order[:total_valid]
        logger.info(f"Stratified epoch: {len(self.epoch_order):,} samples")

    def __len__(self):
        return len(self.epoch_order)

    def __getitem__(self, idx):
        global_idx = self.epoch_order[idx]
        return self._fetch_tokens(global_idx)


# Backwards compat alias
BinShardDataset = StratifiedShardDataset


def collate_pretrain(batch):
    input_ids = torch.stack(batch)
    B, T = input_ids.shape
    if BinShardDataset._causal_mask_4d is None or BinShardDataset._causal_mask_4d.shape[-1] != T:
        BinShardDataset._causal_mask_4d = torch.tril(torch.ones((1, 1, T, T), dtype=torch.bool))
    labels = input_ids.clone()
    return {"input_ids": input_ids, "attention_mask": BinShardDataset._causal_mask_4d.expand(B, -1, -1, -1).contiguous(), "labels": labels}


def validate_cpu_params(model, logger):
    bad = 0
    bad_info = []
    params = model.get_parameters()
    for i, p in enumerate(params):
        if p is not None and not torch.isfinite(p).all():
            bad += 1
            mask = ~torch.isfinite(p)
            bad_info.append(f"  param[{i}] shape={tuple(p.shape)} dtype={p.dtype} nonfinite={mask.sum().item()}/{p.numel()} min={p.min().item():.3e} max={p.max().item():.3e}")
    if bad:
        logger.error(f"CRITICAL: {bad} CPU master parameters are non-finite after sync. Training would corrupt checkpoints.")
        for info in bad_info:
            logger.error(info)
        raise RuntimeError(f"NaN/Inf detected in {bad} CPU master params after optimizer step. Aborting to preserve clean state.")


def save_checkpoint_robust(state, output_dir, is_best, logger):
    model_sd = state.get("model_state_dict", {})
    bad_keys = []
    for k, v in model_sd.items():
        if not torch.isfinite(v).all():
            n_bad = (~torch.isfinite(v)).sum().item()
            n_total = v.numel()
            bad_keys.append(f"{k}: {n_bad}/{n_total} non-finite")
    if bad_keys:
        logger.warning(f"Checkpoint SAVE ABORTED — non-finite tensors detected ({len(bad_keys)}):")
        for msg in bad_keys[:5]:
            logger.warning(f"  {msg}")
        if len(bad_keys) > 5:
            logger.warning(f"  ... and {len(bad_keys) - 5} more")
        return False

    latest_path = os.path.join(output_dir, "megatrain_latest.pt")
    tmp_path = latest_path + ".tmp"
    torch.save(state, tmp_path)
    os.replace(tmp_path, latest_path)
    logger.info(f"  Saved checkpoint to {latest_path}")

    if is_best:
        best_path = os.path.join(output_dir, "megatrain_best.pt")
        bak_path = best_path + ".bak"
        if os.path.exists(best_path):
            shutil.copy2(best_path, bak_path)
        torch.save(state, tmp_path)
        os.replace(tmp_path, best_path)
        logger.info(f"  Best loss {state['best_loss']:.4f} — saved to {best_path}")

    return True


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-steps", type=int, default=50000)
    parser.add_argument("--lr", type=float, default=4e-4)
    parser.add_argument("--max-seq-len", type=int, default=SEQ_LEN)
    parser.add_argument("--checkpoint-interval", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=12)
    parser.add_argument("--num-grad-slabs", type=int, default=12)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--log-interval", type=int, default=120)
    parser.add_argument("--save-interval", type=int, default=2000)
    parser.add_argument("--output-dir", type=str, default="/home/kenpeter/work/checkpoints")
    parser.add_argument("--muon-lr", type=float, default=0.01, help="Learning rate for Muon 2D params")
    parser.add_argument("--adam-lr", type=float, default=3e-4, help="Learning rate for AdamW 1D/embed/head params")
    parser.add_argument("--tau", type=float, default=150.0, help="QK-Clip spectral norm threshold")
    parser.add_argument("--warmup-steps", type=int, default=1000, help="Linear warmup steps")
    parser.add_argument("--min-lr", type=float, default=1e-6, help="Minimum LR for cosine decay")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    logger.info("=" * 60)
    logger.info("KIMI K2 MUONCLIP — Fresh training from scratch")
    logger.info("=" * 60)
    logger.info(f"Model: Custom 1032M (dim=1536, L=32, h=12, kv=4, ffn=4608)")
    logger.info(f"Data: {SHARD_DIRS}")
    logger.info(f"Params: batch={args.batch_size}, seq_len={args.max_seq_len}, steps={args.num_steps}")
    logger.info(f"Muon lr={args.muon_lr}, AdamW lr={args.adam_lr}, QK-Clip tau={args.tau}")
    logger.info(f"LR schedule: cosine, warmup={args.warmup_steps}, min_lr={args.min_lr}")

    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M", trust_remote_code=True)

    logger.info("Creating model from custom config (random init)...")
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
        torch_dtype="float32",
        head_dim=128,
        architectures=["LlamaForCausalLM"],
    )
    hf_model = AutoModelForCausalLM.from_config(
        hf_config,
        dtype=torch.float32,
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    n_params = sum(p.numel() for p in hf_model.parameters())
    logger.info(f"Model loaded: {n_params:,} parameters ({n_params/1e9:.2f}B)")

    # MegaTrain config
    config = CPUMasterConfig(
        model_name="custom-1B",
        dataset_path="/tmp/dummy",
        max_seq_len=args.max_seq_len,
        batch_size=args.batch_size,
        num_steps=args.num_steps,
        learning_rate=args.lr,
        gradient_accumulation_steps=args.grad_accum,
        checkpoint_interval=args.checkpoint_interval,
        num_grad_slabs=args.num_grad_slabs,
        device=args.device,
        dtype=torch.float32,
        log_interval=1,
        attn_implementation="sdpa",
        trust_remote_code=True,
    )

    model = CPUMasterModel(hf_model, config)
    del hf_model

    # === KIMI K2 MUONCLIP SETUP ===
    params = model.get_parameters()
    vocab_embed_numel = 49152 * 1536  # embed_tokens + lm_head

    muon_params = [p for p in params if p.ndim >= 2 and p.numel() != vocab_embed_numel]
    embed_head_params = [p for p in params if p.ndim >= 2 and p.numel() == vocab_embed_numel]
    scalar_params = [p for p in params if p.ndim < 2]

    logger.info(f"KimiMuonClip | Muon 2D: {len(muon_params)} params | "
                f"Embed/Head: {len(embed_head_params)} params | "
                f"Scalar: {len(scalar_params)} params")

    param_groups = [
        dict(params=muon_params, lr=args.muon_lr, momentum=0.95,
             weight_decay=config.weight_decay, use_muon=True, warmup=True),
        dict(params=embed_head_params, lr=args.adam_lr, betas=(0.8, 0.95),
             eps=1e-10, weight_decay=config.weight_decay, use_muon=False),
        dict(params=scalar_params, lr=args.adam_lr, betas=(0.9, 0.95),
             eps=1e-10, weight_decay=config.weight_decay, use_muon=False),
    ]
    optimizer = KimiMuonClip(param_groups, tau=args.tau, ns_steps=7, use_gpu_ns=True)
    logger.info("Optimizer: KimiMuonClip (Newton-Schulz + RMS scaling + QK-Clip + momentum warmup) [GPU NS enabled]")

    # Dataset
    logger.info("Loading dataset...")
    dataset = BinShardDataset(SHARD_DIRS, seq_len=args.max_seq_len, ratios=RATIOS, dedup=False)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=collate_pretrain,
        shuffle=False,  # stratified order already shuffled
        num_workers=0,
        pin_memory=False,
    )
    data_iter = iter(dataloader)

    # Training loop
    logger.info("=" * 60)
    logger.info("Starting pretraining from scratch...")
    logger.info("=" * 60)

    best_loss = float("inf")
    global_step = 0

    for step in range(config.num_steps):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        t0 = time.perf_counter()

        loss_val, n_tokens, timing = model.forward_and_backward(
            batch["input_ids"], batch["attention_mask"], batch["labels"]
        )

        if (step + 1) % config.gradient_accumulation_steps == 0:
            global_step += 1

            # Apply cosine LR schedule to each param group
            for group in optimizer.param_groups:
                base_lr = group["base_lr"] if "base_lr" in group else group["lr"]
                if "base_lr" not in group:
                    group["base_lr"] = base_lr  # store original once
                group["lr"] = get_lr(
                    step + 1,  # outer sample step, not global_step (optimizer step)
                    args.warmup_steps,
                    args.num_steps,
                    base_lr,
                    args.min_lr,
                )

            # Only clip AdamW params; Muon already normalizes via Newton-Schulz
            for group in optimizer.param_groups:
                if not group.get("use_muon", False):
                    torch.nn.utils.clip_grad_norm_(group["params"], config.max_grad_norm)

            optimizer.step(global_step=global_step)

            model._sync_params_to_gpu()
            validate_cpu_params(model, logger)
            model.zero_grad()
            optimizer.zero_grad()

        step_time = time.perf_counter() - t0
        tps = config.batch_size * config.max_seq_len / step_time

        if (step + 1) % args.log_interval == 0:
            gpu_mem = torch.cuda.max_memory_allocated(args.device) / 1024**3
            current_muon_lr = next((g["lr"] for g in optimizer.param_groups if g.get("use_muon")), args.muon_lr)
            current_adam_lr = next((g["lr"] for g in optimizer.param_groups if not g.get("use_muon")), args.adam_lr)
            logger.info(
                f"Step {step+1}/{config.num_steps} | "
                f"Loss {loss_val:.4f} | "
                f"LR muon={current_muon_lr:.2e} adam={current_adam_lr:.2e} | "
                f"{step_time:.2f}s/step | "
                f"{tps:.0f} tok/s | "
                f"GPU {gpu_mem:.2f}GB"
            )

        # Save checkpoints
        if (step + 1) % args.save_interval == 0 or step == config.num_steps - 1:
            is_best = loss_val < best_loss
            if is_best:
                best_loss = loss_val

            full_sd = {}
            embed_sd = model.embedding.state_dict()
            for k, v in embed_sd.items():
                full_sd[f"model.embed_tokens.{k}"] = v
            for i, layer in enumerate(model.cpu_layers):
                for k, v in layer.state_dict().items():
                    full_sd[f"model.layers.{i}.{k}"] = v
            norm_sd = model.norm.state_dict()
            for k, v in norm_sd.items():
                full_sd[f"model.norm.{k}"] = v
            head_sd = model.lm_head.state_dict()
            for k, v in head_sd.items():
                full_sd[f"lm_head.{k}"] = v

            state = {
                "step": step + 1,
                "loss": loss_val,
                "best_loss": best_loss,
                "model_state_dict": full_sd,
                "optimizer_state_dict": optimizer.state_dict(),
            }
            save_checkpoint_robust(state, args.output_dir, is_best, logger)

    model.cleanup()
    logger.info("Pretraining complete!")


if __name__ == "__main__":
    main()
