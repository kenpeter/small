"""Per-domain loss comparison: megatrain_best.pt vs megatrain_swa.pt.
Reads raw uint16 .bin shards (2 bytes/token, contiguous 2048-token seqs —
same format as pretrain_megatrain.py StratifiedShardDataset), computes CE
loss per domain under each checkpoint. Side-by-side table.
"""
import argparse, glob, numpy as np, torch
import torch.nn.functional as F
from transformers import LlamaConfig, LlamaForCausalLM

DOMAINS = ["_shards_math_easy", "_shards_web_easy", "_shards_gold", "_shards_code_easy",
           "_shards_synth_easy", "_shards_reformat_easy", "_shards_web_hard", "_shards_math_hard"]
DATA_ROOT = "/home/kenpeter/work/data"
SEQ_LEN = 2048
SEQS_PER_DOMAIN = 64
MAX_STEPS = 512  # cap tokens fed per seq to avoid OOM (512 ctx window per loss call)

def load_sd(path):
    sd = torch.load(path, map_location="cpu", weights_only=False)["model_state_dict"]
    if any(k.startswith("_orig_mod.") for k in sd):
        sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    return sd

def build_model(sd):
    cfg = LlamaConfig(vocab_size=49152, hidden_size=1536, intermediate_size=4608,
                      num_hidden_layers=32, num_attention_heads=12, num_key_value_heads=4,
                      max_position_embeddings=8192, rope_theta=10000.0, rms_norm_eps=1e-5,
                      hidden_act="silu", tie_word_embeddings=False, attention_bias=False,
                      mlp_bias=False, head_dim=128)
    m = LlamaForCausalLM(cfg).to("cuda").eval().to(torch.bfloat16)
    missing, unexpected = m.load_state_dict(sd, strict=False)
    if missing:
        print(f"  WARN missing: {missing[:3]} ({len(missing)})")
    return m

def domain_seqs(domain):
    files = sorted(glob.glob(f"{DATA_ROOT}/{domain}/*.bin"))
    files = [p for p in files if __import__('os').path.getsize(p) > 0]
    files = files[:8]  # cap file reads for speed
    seqs = []
    for p in files:
        arr = np.fromfile(p, dtype=np.uint16)
        n = len(arr) // SEQ_LEN
        for i in range(n):
            seqs.append(arr[i * SEQ_LEN:(i + 1) * SEQ_LEN].tolist())
            if len(seqs) >= SEQS_PER_DOMAIN:
                return seqs
    return seqs

@torch.no_grad()
def domain_loss(model, seqs):
    tot, cnt = 0.0, 0
    for s in seqs:
        for start in range(0, min(len(s), 2048) - 1, MAX_STEPS):
            chunk = s[start:start + MAX_STEPS + 1]
            if len(chunk) < 2:
                continue
            ids = torch.tensor(chunk, dtype=torch.long, device="cuda").unsqueeze(0)
            logits = model(ids).logits[0, :-1].float()
            nll = F.cross_entropy(logits, ids[0, 1:])
            tot += nll.item(); cnt += 1
    return tot / cnt

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", default=[
        "/home/kenpeter/work/checkpoints/megatrain_best.pt",
        "/home/kenpeter/work/checkpoints/megatrain_swa.pt"])
    args = ap.parse_args()

    models = {}
    for ck in args.ckpts:
        name = ck.split("/")[-1]
        print(f"Loading {name}...", flush=True)
        models[name] = build_model(load_sd(ck))
    print(f"\n{'domain':<18} " + "  ".join(f"{n.split('.')[0]:>15}" for n in models) + "   Δ(best-swa)", flush=True)
    print("-" * 72, flush=True)
    for d in DOMAINS:
        seqs = domain_seqs(d)
        if not seqs:
            print(f"{d:<18} (no data)", flush=True)
            continue
        vals = {}
        for name, m in models.items():
            vals[name] = domain_loss(m, seqs)
        names = [n.split('.')[0] for n in args.ckpts]
        b, s = vals[args.ckpts[0].split('/')[-1]], vals[args.ckpts[1].split('/')[-1]]
        row = f"{d:<18} " + "  ".join(f"{vals[args.ckpts[i].split('/')[-1]]:>15.4f}" for i in range(len(args.ckpts)))
        print(row + f"   {b-s:+.4f}", flush=True)