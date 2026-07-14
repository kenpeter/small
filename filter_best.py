"""Filter best 1.6B tokens by quality scoring."""
import os, sys, gc, json
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from train import TrainConfig, SmolLM2

DATA_DIR = Path("/home/kenpeter/work/data")
BEST_DIR = DATA_DIR / "best_1.6b"
BEST_DIR.mkdir(exist_ok=True)
CKPT_DIR = Path("/home/kenpeter/work/checkpoints")

TARGET = int(1.6e9)
SEQ_LEN = 2048
BATCH = 8

def load_model(device):
    cfg = TrainConfig()
    model = SmolLM2(cfg).to(device)
    ckpt = CKPT_DIR / "checkpoint_best.pt"
    if ckpt.exists():
        state = torch.load(ckpt, map_location=device, weights_only=False)
        model.load_state_dict(state['model_state_dict'])
        print(f"Loaded step {state.get('step', 'unknown')}")
    model.eval()
    return model

def score(seqs, model, device):
    batch = torch.tensor(seqs, dtype=torch.long, device=device)
    with torch.no_grad():
        output = model(batch)
        # Handle tuple output (logits, loss)
        logits = output[0] if isinstance(output, tuple) else output
        loss = torch.nn.functional.cross_entropy(
            logits[:, :-1, :].reshape(-1, logits.size(-1)),
            batch[:, 1:].reshape(-1),
            reduction='none'
        )
        return loss.view(batch.size(0), -1).mean(dim=1).cpu().numpy()

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    model = load_model(device)
    
    shards = sorted(DATA_DIR.glob("shard_*.bin"))
    print(f"Shards: {len(shards)}")
    
    scores = []
    for i, shard in enumerate(shards):
        print(f"[{i+1}/{len(shards)}] {shard.name}")
        tokens = np.memmap(str(shard), dtype=np.uint16, mode='r')
        n_seqs = len(tokens) // SEQ_LEN
        
        for j in range(0, n_seqs, BATCH):
            batch = []
            for k in range(j, min(j + BATCH, n_seqs)):
                batch.append(tokens[k * SEQ_LEN:(k + 1) * SEQ_LEN].tolist())
            while len(batch) < BATCH:
                batch.append([0] * SEQ_LEN)
            
            s = score(batch, model, device)[:len(batch)]
            for k, loss in enumerate(s):
                scores.append((shard.name, j + k, float(loss)))
            
            if (j // BATCH) % 100 == 0 and j > 0:
                print(f"  {j}/{n_seqs}")
        
        del tokens
        gc.collect()
    
    print(f"Scored {len(scores):,} sequences")
    scores.sort(key=lambda x: x[2])
    
    keep_n = TARGET // SEQ_LEN
    top = scores[:keep_n]
    print(f"Keep top {len(top):,}")
    
    # Group by shard
    by_shard = {}
    for name, idx, _ in top:
        by_shard.setdefault(name, []).append(idx)
    
    # Extract
    out_buf = []
    out_idx = 0
    for name, idxs in sorted(by_shard.items()):
        tokens = np.memmap(str(DATA_DIR / name), dtype=np.uint16, mode='r')
        for idx in sorted(idxs):
            out_buf.extend(tokens[idx * SEQ_LEN:(idx + 1) * SEQ_LEN])
            while len(out_buf) >= SEQ_LEN:
                np.array(out_buf[:SEQ_LEN], dtype=np.uint16).tofile(
                    BEST_DIR / f"shard_{out_idx:06d}.bin"
                )
                out_idx += 1
                out_buf = out_buf[SEQ_LEN:]
        del tokens
        gc.collect()
    
    if out_buf:
        np.array(out_buf, dtype=np.uint16).tofile(BEST_DIR / f"shard_{out_idx:06d}.bin")
        out_idx += 1
    
    json.dump({
        "source": len(shards),
        "scored": len(scores),
        "kept": len(top),
        "tokens": len(top) * SEQ_LEN,
        "shards": out_idx
    }, open(BEST_DIR / "best_info.json", "w"), indent=2)
    
    print(f"Done! {len(top):,} seqs = {len(top)*SEQ_LEN:,} tokens")

if __name__ == "__main__":
    main()
