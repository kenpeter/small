"""
Pretraining with MegaTrain + Kimi K2 MuonClip
CPU offloaded training of 1.03B model with orthogonal updates.
Loads .bin shards directly (already tokenized with SmolLM2 tokenizer).
"""
import os, time, logging, argparse, math, shutil
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np

from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaConfig
from infinity import CPUMasterModel
from infinity.config import CPUMasterConfig

logger = logging.getLogger(__name__)

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
    - Momentum warmup: 0.85 -> 0.95 over first 300 steps
    - QK-Clip proxy: spectral norm cap on attention projections
    """
    def __init__(self, param_groups, tau: float = 100.0, ns_steps: int = 5):
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
        defaults = dict(tau=tau, ns_steps=ns_steps)
        super().__init__(param_groups, defaults)

    @torch.no_grad()
    def step(self, closure=None, global_step: int = 0):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # Momentum warmup: 0.85 -> 0.95 over first 300 steps
        frac = min(global_step / 300.0, 1.0)
        warmed_momentum = (1 - frac) * 0.85 + frac * 0.95

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
                    # Momentum: Mt = μ * Mt-1 + Gt
                    buf.mul_(beta).add_(p.grad)

                    # Newton-Schulz orthogonalization
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

        # QK-Clip proxy: cap spectral norm of attention-like projections
        tau = self.defaults["tau"]
        for group in self.param_groups:
            if group.get("use_muon", False):
                for p in group["params"]:
                    if p.ndim >= 2 and p.shape[0] <= p.shape[1]:
                        # Covers q_proj (1536,1536), k_proj (512,1536), v_proj (512,1536),
                        # o_proj (1536,1536), down_proj (1536,4608)
                        with torch.no_grad():
                            spec_norm = torch.linalg.matrix_norm(p.data, ord=2)
                            if spec_norm > tau:
                                p.data.mul_(tau / spec_norm)

        return loss


SHARDS_DIR = Path("/home/kenpeter/work/data/_shards_final")
SEQ_LEN = 2048


class BinShardDataset(Dataset):
    """Memory-maps .bin shards and yields sequences of SEQ_LEN tokens."""
    _causal_mask_4d = None
    def __init__(self, shards_dir, seq_len: int = 2048):
        shards_dir = Path(shards_dir)
        shard_paths = sorted(shards_dir.glob("shard_*.bin"))
        shard_paths = [p for p in shard_paths if p.stat().st_size > 0]
        if not shard_paths:
            raise FileNotFoundError(f"No .bin shards found in {shards_dir}")

        self.seq_len = seq_len
        self.shard_bounds = []
        self.total_seqs = 0

        for p in shard_paths:
            n_tokens = p.stat().st_size // 2
            n_seqs = n_tokens // seq_len
            if n_seqs == 0:
                continue
            self.shard_bounds.append((p, n_seqs, self.total_seqs))
            self.total_seqs += n_seqs

        logger.info(f"Loaded {len(shard_paths)} shards, {self.total_seqs:,} sequences ({self.total_seqs * seq_len:,} tokens)")

    def __len__(self):
        return self.total_seqs

    def __getitem__(self, idx):
        for shard_path, n_seqs, start_idx in self.shard_bounds:
            if idx < start_idx + n_seqs:
                local_idx = idx - start_idx
                offset = local_idx * self.seq_len
                mm = np.memmap(str(shard_path), dtype=np.uint16, mode='r',
                               offset=offset * 2, shape=(self.seq_len,))
                tokens = torch.from_numpy(mm.copy().astype(np.int64))
                del mm
                return tokens
        raise IndexError(f"Index {idx} out of range")


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
    parser.add_argument("--tau", type=float, default=100.0, help="QK-Clip spectral norm threshold")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    logger.info("=" * 60)
    logger.info("KIMI K2 MUONCLIP — Fresh training from scratch")
    logger.info("=" * 60)
    logger.info(f"Model: Custom 1032M (dim=1536, L=32, h=12, kv=4, ffn=4608)")
    logger.info(f"Data: {SHARDS_DIR}")
    logger.info(f"Params: batch={args.batch_size}, seq_len={args.max_seq_len}, steps={args.num_steps}")
    logger.info(f"Muon lr={args.muon_lr}, AdamW lr={args.adam_lr}, QK-Clip tau={args.tau}")

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
    optimizer = KimiMuonClip(param_groups, tau=args.tau, ns_steps=5)
    logger.info("Optimizer: KimiMuonClip (Newton-Schulz + RMS scaling + QK-Clip + momentum warmup)")

    # Dataset
    logger.info("Loading dataset...")
    dataset = BinShardDataset(SHARDS_DIR, seq_len=args.max_seq_len)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=collate_pretrain,
        shuffle=True,
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
            logger.info(
                f"Step {step+1}/{config.num_steps} | "
                f"Loss {loss_val:.4f} | "
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
