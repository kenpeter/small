#!/usr/bin/env python3
"""DEITA-style Qwen scoring of a sampled SFT subset.
Scores N examples per dataset for quality + complexity (batch 10/call, JSON out),
then correlates with local audit losses to find the loss band that separates
Qwen-good from Qwen-bad → extrapolate the threshold to all 6M.
Requires vLLM Qwen3-8B-AWQ on 127.0.0.1:8000 (offline mode).
"""
import argparse, json, os, random, re, time
import urllib.request
import numpy as np

API = "http://127.0.0.1:8000/v1/chat/completions"
KEY = "token-abc123"
SHARDS = sorted(__import__("glob").glob("/mnt/file_drive/data/_sft_final_shards/*.pt"))
OUT = "/home/kenpeter/work/sft_qwen_scores.jsonl"
BATCH_CALLS = 10          # examples per API call (keep ≤ context)
PER_DATASET = 2000        # sample per dataset

PROMPT = """You are a data-quality judge. Rate each instruction-response pair below.

For each item output JSON: {"scores": [{"quality": 1-10, "complexity": 1-10}, ...]}

- quality: is the response correct, coherent, follows the instruction? 10 = perfect, 1 = garbage.
- complexity: how hard/valuable is the task? 10 = complex multi-step reasoning, 1 = trivial filler.

Be strict. Low-quality or trivial items must get LOW scores.

=== ITEMS ===
{items}
=== END ===
Return ONLY the JSON."""


def extract_answer(text):
    """Decode assistant-only text from tokenized ids? No — we score from RAW? We only have
    tokenized shards. So decode token ids back to text for the judge."""
    pass  # handled by caller passing raw strings


def call_qwen(items_text):
    payload = {
        "model": "Qwen/Qwen3-8B-AWQ",
        "messages": [{"role": "user", "content": PROMPT.format(items="\n".join(items_text))}],
        "temperature": 0.0, "max_tokens": 1024,
    }
    req = urllib.request.Request(API, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json",
                                          "Authorization": f"Bearer {KEY}"})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                body = json.loads(r.read())
            content = body["choices"][0]["message"]["content"]
            content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)
            m = re.search(r"\{.*\}", content, flags=re.DOTALL)
            if not m:
                return None
            return json.loads(m.group(0))
        except Exception as e:
            if attempt == 3:
                return None
            time.sleep(5 * (attempt + 1))
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-dataset", type=int, default=PER_DATASET)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    # group files by the dataset FIELD of first example (matches audit keys)
    groups = {}
    for f in SHARDS:
        try:
            probe = torch.load(f, map_location="cpu", weights_only=False)
            k = probe[0].get("dataset", os.path.basename(f).split("_")[1]) if probe else os.path.basename(f)
            groups.setdefault(k, []).append(f)
            del probe
        except Exception:
            groups.setdefault(os.path.basename(f).split("_")[1], []).append(f)
    print(f"groups: { {k: len(v) for k, v in groups.items()} }", flush=True)

    import torch
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M", trust_remote_code=True)

    total_calls = 0
    t0 = time.time()
    with open(OUT, "w") as wf:
        for dskey, files in groups.items():
            # sample examples across files, RECORDING (file, pos) provenance
            pool, prov = [], []
            for f in files:
                data = torch.load(f, map_location="cpu", weights_only=False)
                for pos, ex in enumerate(data):
                    pool.append(ex)
                    prov.append((dskey, os.path.basename(f), pos))
                    if len(pool) >= args.per_dataset:
                        break
                if len(pool) >= args.per_dataset:
                    break
                del data
            rng.shuffle(pool)
            # shuffle prov in lockstep
            perm = list(range(len(pool)))
            rng.shuffle(perm)
            pool = [pool[i] for i in perm]
            prov = [prov[i] for i in perm]
            pool = pool[:args.per_dataset]
            prov = prov[:args.per_dataset]
            print(f"{dskey}: scoring {len(pool)} examples", flush=True)
            # build item strings: prompt context + assistant answer (decoded)
            items_text = []
            for ex in pool:
                # find assistant span: labels != -100
                labels = ex["labels"]
                mask = [i for i, l in enumerate(labels) if l != -100]
                if not mask:
                    continue
                a0, a1 = mask[0], mask[-1] + 1
                ask = tok.decode(ex["input_ids"][:max(a0, 0)], skip_special_tokens=True)
                ans = tok.decode(ex["input_ids"][a0:a1], skip_special_tokens=True)
                items_text.append(f"Q: {ask[-400:]}\nA: {ans[:400]}")
            # batch calls
            for i in range(0, len(items_text), BATCH_CALLS):
                batch = items_text[i:i + BATCH_CALLS]
                res = call_qwen(batch)
                total_calls += 1
                if res and "scores" in res:
                    sc = res["scores"]
                    for j, s in enumerate(sc[:len(batch)]):
                        if i + j < len(pool):
                            dk, fn, pos = prov[i + j]
                            rec = {"dataset": dk, "file": fn, "pos": pos,
                                   "quality": s.get("quality"), "complexity": s.get("complexity")}
                            wf.write(json.dumps(rec) + "\n")
                            wf.flush()
                if total_calls % 10 == 0:
                    el = time.time() - t0
                    print(f"  {total_calls} calls, {el/60:.0f} min, "
                          f"{total_calls/el:.2f} calls/s", flush=True)
            print(f"{dskey} done", flush=True)
    print(f"\nSaved {OUT} — {total_calls} calls in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()