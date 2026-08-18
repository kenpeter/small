#!/usr/bin/env python3
"""AlpaGasus-style loss audit v3 — STRATIFIED SAMPLE.
Samples up to N examples per dataset group, computes per-example CE over
assistant tokens (batched), reports per-dataset loss distributions to set
the AlpaGasus drop threshold WITHOUT a 38h full pass."""
import glob, json, os, time
import numpy as np
import torch
import torch.nn.functional as F
from transformers import LlamaConfig, LlamaForCausalLM
from tqdm import tqdm

SHARDS = sorted(glob.glob("/mnt/file_drive/data/_sft_final_shards/*.pt"))
OUT = "/home/kenpeter/work/sft_loss_audit.json"
BATCH = 32
MAX_TOK = 2048
CHUNK = 512
PER_DATASET = 60000  # target sample per dataset group

def load_sd():
    sd = torch.load("/home/kenpeter/work/checkpoints/megatrain_swa.pt",
                    map_location="cpu", weights_only=False)["model_state_dict"]
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
    m.load_state_dict(sd, strict=False)
    return m

@torch.no_grad()
def batch_loss(model, batch_ids, batch_labels):
    maxlen = max(len(x) for x in batch_ids)
    bs = len(batch_ids)
    out = [0.0] * bs
    cnt = [0] * bs
    for s in range(0, maxlen - 1, CHUNK):
        cur_ids, cur_lab, idxs = [], [], []
        for i, (ids, labs) in enumerate(zip(batch_ids, batch_labels)):
            seg = ids[s:s + CHUNK + 1]
            lab = labs[s:s + CHUNK + 1]
            if len(seg) < 2:
                continue
            if (np.array(lab[1:]) != -100).sum() == 0:
                continue
            cur_ids.append(seg); cur_lab.append(lab); idxs.append(i)
        if not cur_ids:
            continue
        pad_ids = torch.full((len(cur_ids), CHUNK + 1), 0, dtype=torch.long)
        pad_lab = torch.full((len(cur_ids), CHUNK + 1), -100, dtype=torch.long)
        for j, (seg, lab) in enumerate(zip(cur_ids, cur_lab)):
            pad_ids[j, :len(seg)] = torch.tensor(seg)
            pad_lab[j, :len(seg)] = torch.tensor(lab)
        pad_ids, pad_lab = pad_ids.to("cuda"), pad_lab.to("cuda")
        logits = model(pad_ids).logits[:, :-1].float()
        nll = F.cross_entropy(logits.reshape(-1, logits.size(-1)), pad_lab[:, 1:].reshape(-1),
                              reduction="none").reshape(len(cur_ids), -1)
        mask = (pad_lab[:, 1:] != -100)
        for j, i in enumerate(idxs):
            mj = mask[j]
            out[i] += nll[j][mj].sum().item()
            cnt[i] += int(mj.sum().item())
    return [out[i] / cnt[i] if cnt[i] else None for i in range(bs)]

def main():
    model = build_model(load_sd())
    # group by each example's dataset field (file names all start 'shard_')
    groups = {}
    for f in SHARDS:
        try:
            probe = torch.load(f, map_location="cpu", weights_only=False)
            k = probe[0].get("dataset", os.path.basename(f)) if probe else os.path.basename(f)
            groups.setdefault(k, []).append(f)
            del probe
        except Exception:
            groups.setdefault(os.path.basename(f).split("_")[1], []).append(f)
    print(f"dataset groups: { {k: len(v) for k, v in groups.items()} }", flush=True)
    report = {"per_dataset": {}, "overall": {}}
    all_losses = []
    t0 = time.time()
    for dskey, files in groups.items():
        losses = []
        buf_ids, buf_lab = [], []
        def flush():
            nonlocal buf_ids, buf_lab
            if not buf_ids:
                return
            ls = batch_loss(model, buf_ids, buf_lab)
            losses.extend(l for l in ls if l is not None)
            buf_ids, buf_lab = [], []
        for f in files:
            data = torch.load(f, map_location="cpu", weights_only=False)
            for ex in data:
                if len(losses) + len(buf_ids) >= PER_DATASET:
                    break
                buf_ids.append(ex["input_ids"][:MAX_TOK])
                buf_lab.append(ex["labels"][:MAX_TOK])
                if len(buf_ids) >= BATCH:
                    flush()
            if len(losses) + len(buf_ids) >= PER_DATASET:
                flush()
                break
            del data
        flush()
        losses = np.array(losses)
        pcts = {f"p{p}": float(np.percentile(losses, p)) for p in [10, 25, 50, 75, 90, 95, 99]}
        report["per_dataset"][dskey] = {
            "n": int(len(losses)), "mean": float(losses.mean()), "pcts": pcts}
        all_losses.extend(losses.tolist())
        print(f"{dskey}: n={len(losses):,} mean={losses.mean():.4f} "
              f"p50={pcts['p50']:.3f} p90={pcts['p90']:.3f} p99={pcts['p99']:.3f}", flush=True)
        json.dump(report, open(OUT, "w"), indent=1)
    a = np.array(all_losses)
    report["overall"] = {
        "n": int(len(a)), "mean": float(a.mean()),
        "pcts": {f"p{p}": float(np.percentile(a, p)) for p in [50, 75, 90, 95, 99]},
        "drop_30pct_threshold": float(np.percentile(a, 70)),
        "drop_50pct_threshold": float(np.percentile(a, 50))}
    json.dump(report, open(OUT, "w"), indent=1)
    print(f"\nSaved {OUT} in {(time.time()-t0)/60:.1f} min")
    print(f"Overall n={len(a):,} mean={a.mean():.4f} p50={np.percentile(a,50):.3f} "
          f"p90={np.percentile(a,90):.3f} p99={np.percentile(a,99):.3f}")

if __name__ == "__main__":
    main()