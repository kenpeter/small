"""
Pretraining with MegaTrain — CPU offloaded training of SmolLM2-1.7B.
Loads .bin shards directly (already tokenized with SmolLM2 tokenizer).
"""
import os, time, logging, argparse, math
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from infinity import CPUMasterModel
from infinity.config import CPUMasterConfig

logger = logging.getLogger(__name__)

SHARDS_DIR = Path("/home/kenpeter/work/data/_shards_final")
SEQ_LEN = 2048

class BinShardDataset(Dataset):
    """Memory-maps .bin shards and yields sequences of SEQ_LEN tokens."""
    def __init__(self, shards_dir, seq_len: int = 2048):
        shards_dir = Path(shards_dir)
        shard_paths = sorted(shards_dir.glob("shard_*.bin"))
        shard_paths = [p for p in shard_paths if p.stat().st_size > 0]
        if not shard_paths:
            raise FileNotFoundError(f"No .bin shards found in {shards_dir}")

        self.seq_len = seq_len
        self.shard_bounds = []
        self.total_seqs = 0

        # Pre-compute bounds and validate dimensions
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
        # Find which shard this index belongs to
        for shard_path, n_seqs, start_idx in self.shard_bounds:
            if idx < start_idx + n_seqs:
                local_idx = idx - start_idx
                offset = local_idx * self.seq_len
                # Memory-map and read one sequence
                mm = np.memmap(str(shard_path), dtype=np.uint16, mode='r',
                               offset=offset * 2, shape=(self.seq_len,))
                tokens = torch.from_numpy(mm.copy().astype(np.int64))
                del mm
                return tokens
        raise IndexError(f"Index {idx} out of range")

def collate_pretrain(batch):
    """Collate pretraining batch: labels = input_ids (all tokens train)."""
    input_ids = torch.stack(batch)  # (batch, seq_len)
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="HuggingFaceTB/SmolLM2-1.7B")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=4e-4)
    parser.add_argument("--max-seq-len", type=int, default=SEQ_LEN)
    parser.add_argument("--checkpoint-interval", type=int, default=4)
    parser.add_argument("--num-grad-slabs", type=int, default=12)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--output-dir", type=str, default="/home/kenpeter/work/checkpoints")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    logger.info("=" * 60)
    logger.info(f"Model: {args.model}")
    logger.info(f"Data: {SHARDS_DIR}")
    logger.info(f"Params: batch={args.batch_size}, seq_len={args.max_seq_len}, steps={args.num_steps}")

    # Load tokenizer (for reference, not used in training since data is pre-tokenized)
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M", trust_remote_code=True)

    # Load model
    logger.info("Loading model...")
    hf_model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    n_params = sum(p.numel() for p in hf_model.parameters())
    logger.info(f"Model loaded: {n_params:,} parameters ({n_params/1e9:.2f}B)")

    # MegaTrain config
    config = CPUMasterConfig(
        model_name=args.model,
        dataset_path="/tmp/dummy",  # dummy — we use our own dataset
        max_seq_len=args.max_seq_len,
        batch_size=args.batch_size,
        num_steps=args.num_steps,
        learning_rate=args.lr,
        checkpoint_interval=args.checkpoint_interval,
        num_grad_slabs=args.num_grad_slabs,
        device=args.device,
        dtype=torch.bfloat16,
        log_interval=1,
        attn_implementation="sdpa",
        trust_remote_code=True,
    )

    # Create CPU Master model
    model = CPUMasterModel(hf_model, config)
    del hf_model

    # Setup optimizer
    logger.info("Setting up DeepSpeed CPUAdam...")
    try:
        from deepspeed.ops.adam import DeepSpeedCPUAdam
        optimizer = DeepSpeedCPUAdam(
            model.get_parameters(),
            lr=config.learning_rate,
            betas=(config.beta1, config.beta2),
            eps=config.eps,
            weight_decay=config.weight_decay,
            adamw_mode=True
        )
    except ImportError:
        logger.warning("DeepSpeed CPUAdam not available, using PyTorch AdamW")
        optimizer = torch.optim.AdamW(
            model.get_parameters(),
            lr=config.learning_rate,
            betas=(config.beta1, config.beta2),
            eps=config.eps,
            weight_decay=config.weight_decay,
        )

    # Dataset
    logger.info("Loading dataset...")
    dataset = BinShardDataset(SHARDS_DIR, seq_len=args.max_seq_len)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=collate_pretrain,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    data_iter = iter(dataloader)

    # Training loop
    logger.info("=" * 60)
    logger.info("Starting pretraining...")
    logger.info("=" * 60)

    best_loss = float("inf")
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
            torch.nn.utils.clip_grad_norm_(model.get_parameters(), config.max_grad_norm)
            optimizer.step()
            model._sync_params_to_gpu()
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
        if (step + 1) % 500 == 0 or step == config.num_steps - 1:
            is_best = loss_val < best_loss
            if is_best:
                best_loss = loss_val

            # Reconstruct full state dict from CPUMasterModel components
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

            latest_path = os.path.join(args.output_dir, "megatrain_latest.pt")
            torch.save(state, latest_path)
            logger.info(f"  Saved checkpoint to {latest_path}")

            if is_best:
                best_path = os.path.join(args.output_dir, "megatrain_best.pt")
                torch.save(state, best_path)
                logger.info(f"  Best loss {best_loss:.4f} — saved to {best_path}")

    model.cleanup()
    logger.info("Pretraining complete!")

if __name__ == "__main__":
    main()
