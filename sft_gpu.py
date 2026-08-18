#!/usr/bin/env python3
"""SFT (Supervised Fine-Tuning) for small-1B.

Loads the pretrained 1B checkpoint (megatrain_best.pt), fine-tunes on pre-tokenized
instruction shards (_sft_final_shards, assistant-only labels = -100 for user/system),
saves stage-isolated checkpoints: sft_latest.pt + sft_best.pt (same dict format as
pretrain so it can resume/warm-start downstream stages).

Usage:
  ./venv/bin/python sft_gpu.py --steps 4000 --lr 2e-5 --save-every 500
"""
import argparse, glob, json, math, os, time
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, LlamaConfig

from pretrain_gpu import build_model, make_optimizer, chunked_ce, resolve_resume_path, save_checkpoint_async  # noqa

BASE_CKPT = Path("/home/kenpeter/work/checkpoints/megatrain_swa.pt")
SFT_SHARDS = Path("/mnt/file_drive/data/_sft_gold_shards")
OUT_DIR = Path("/home/kenpeter/work/checkpoints")
SEQ_LEN = 2048


class SFTDataset(Dataset):
    """Reads pre-tokenized .pt shards: list of {input_ids: list, labels: list, dataset: str}."""

    def __init__(self, shards_dir: Path):
        self.files = sorted(glob.glob(str(Path(shards_dir) / "*.pt")))
        assert self.files, f"no shards in {shards_dir}"
        self.examples = []
        for f in self.files:
            data = torch.load(f, map_location="cpu", weights_only=False)
            self.examples.extend(data)
        print(f"SFTDataset: {len(self.examples):,} examples from {len(self.files)} shards", flush=True)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        ex = self.examples[i]
        return {"input_ids": ex["input_ids"], "labels": ex["labels"], "dataset": ex.get("dataset", "")}


def collate(batch, seq_len=None):
    """Pad/truncate to seq_len. Truncation keeps the LAST seq_len tokens (answer stays)."""
    if seq_len is None:
        seq_len = SEQ_LEN
    ids, labels = [], []
    for b in batch:
        x, y = b["input_ids"], b["labels"]
        if len(x) > seq_len:
            x, y = x[-seq_len:], y[-seq_len:]
        ids.append(torch.tensor(x, dtype=torch.long))
        labels.append(torch.tensor(y, dtype=torch.long))
    input_ids = torch.nn.utils.rnn.pad_sequence(ids, batch_first=True, padding_value=0)
    labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
    return {"input_ids": input_ids, "labels": labels}


def cosine_lr(step, max_steps, warmup, base_lr, min_lr=1e-6):
    if step <= warmup:
        return base_lr * step / max(warmup, 1)
    p = min((step - warmup) / max(max_steps - warmup, 1), 1.0)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--max-seq-len", type=int, default=SEQ_LEN)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--init-from", type=str, default=str(BASE_CKPT),
                    help="pretrained ckpt to warm-start (model_state_dict)")
    ap.add_argument("--resume-from", type=str, default=None, help="sft_latest.pt to resume")
    ap.add_argument("--output-dir", type=str, default=str(OUT_DIR))
    ap.add_argument("--shards-dir", type=str, default=str(SFT_SHARDS))
    ap.add_argument("--cautious", action="store_true", default=True)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42)
    dtype = torch.bfloat16

    model = build_model(dtype).to(device)
    resume_path = args.resume_from or resolve_resume_path(None, str(Path(args.output_dir) / "sft_latest.pt"))
    start_step = 0
    if resume_path and Path(resume_path).exists():
        ck = torch.load(resume_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model_state_dict"])
        start_step = ck.get("step", 0)
        print(f"Resumed SFT from {resume_path} (step {start_step})", flush=True)
    else:
        ck = torch.load(args.init_from, map_location="cpu", weights_only=False)
        sd = ck["model_state_dict"] if "model_state_dict" in ck else ck
        model.load_state_dict(sd)
        print(f"Warm-started from {args.init_from} (pretrain best loss {ck.get('best_loss') if ck.get('best_loss') is not None else 'n/a'})", flush=True)

    optimizer = make_optimizer(model, base_lr=args.lr, cautious=args.cautious)
    ds = SFTDataset(Path(args.shards_dir))
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                    collate_fn=lambda b: collate(b, seq_len=args.max_seq_len),
                    num_workers=2, drop_last=True)
    data_iter = iter(dl)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")
    t0 = time.time()

    for step in range(start_step + 1, args.steps + 1):
        lr = cosine_lr(step, args.steps, args.warmup, args.lr)
        for pg in optimizer.param_groups:
            pg["lr"] = lr
        acc_loss, n_tok = 0.0, 0
        for _ in range(args.grad_accum):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dl)
                batch = next(data_iter)
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            if (labels != -100).sum() == 0:
                continue
            with torch.autocast(device_type="cuda", dtype=dtype):
                out = model(input_ids=input_ids)
                loss = chunked_ce(out.logits, labels) / args.grad_accum
            acc_loss += loss.item() * args.grad_accum
            n_tok += (labels != -100).sum().item()
            loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        if step % args.log_every == 0:
            dt = time.time() - t0
            print(f"step {step:6d} | loss {acc_loss:.4f} | lr {lr:.2e} | {dt:.0f}s | {n_tok/dt:,.0f} tok/s", flush=True)
            t0 = time.time()

        if step % args.save_every == 0 or step == args.steps:
            is_best = acc_loss < best_loss
            if is_best:
                best_loss = acc_loss
            state = {
                "step": step, "loss": acc_loss, "best_loss": best_loss,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": args.__dict__,
            }
            tmp = out_dir / "sft_latest.tmp"
            torch.save(state, tmp)
            os.replace(tmp, out_dir / "sft_latest.pt")
            if is_best:
                os.replace(str(out_dir / "sft_latest.pt"), str(out_dir / "sft_best.pt"))
                os.replace(str(out_dir / "sft_latest.tmp"), str(out_dir / "sft_latest.pt")) if False else None
                # best saved above; re-write latest copy
                torch.save(state, out_dir / "sft_latest.pt")
                print(f"  ⭐ sft_best.pt (loss {acc_loss:.4f}) @ step {step}", flush=True)
            else:
                print(f"  💾 sft_latest.pt (loss {acc_loss:.4f}) @ step {step}", flush=True)

    print("✅ SFT complete — sft_best.pt / sft_latest.pt written to", out_dir)


if __name__ == "__main__":
    main()