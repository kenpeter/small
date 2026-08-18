#!/usr/bin/env python3
"""SFT gold set — LIMA-style (~1000 best-of-best prompt→answer pairs).

Sources: high_quality_leetcode (high_quality_cot reference) + finemath score>=2.
Format: {prompt, response, dataset} — SFT shards get assistant-only labels.

Usage: ./venv/bin/python sft_gold_gen.py --max 1000 --out data/_sft_gold/gold.jsonl
Requires vLLM Qwen3-8B-AWQ on 127.0.0.1:8000 (offline mode).
"""
import argparse, json, random, re, sys, time, urllib.request
from pathlib import Path

MODEL = "Qwen/Qwen3-8B-AWQ"
API_BASE = "http://127.0.0.1:8000/v1"
API_KEY = "token-abc123"
LEETCODE = "/mnt/file_drive/data/high_quality_leetcode/train.jsonl"
MATH_DIR = "/mnt/file_drive/data/finemath-3plus"

def strip_think(text):
    return re.sub(r"<\|im_start\|> think.*?<\|im_end\|>| think.*? response", "", text, flags=re.DOTALL).strip()

def call_vllm(prompt, max_tokens=2048, temperature=0.2, retries=3):
    payload = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }).encode()
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                f"{API_BASE}/chat/completions", data=payload,
                headers={"Content-Type": "application/json",
                         "Authorization": f"Bearer {API_KEY}"})
            resp = json.loads(urllib.request.urlopen(req, timeout=180).read())
            return strip_think(resp["choices"][0]["message"]["content"].strip())
        except Exception as e:
            if attempt == retries - 1:
                raise
            time.sleep(5 * (attempt + 1))

def load_leetcode(max_q):
    qs = []
    with open(LEETCODE) as f:
        for line in f:
            d = json.loads(line)
            if not d.get("problem_description") or not d.get("high_quality_cot"):
                continue
            qs.append({
                "prompt": d["problem_description"][:3000],
                "response": d["high_quality_cot"][:4000],  # reference answer (already high quality)
                "dataset": "sft_gold_code",
            })
    random.shuffle(qs)
    return qs[:max_q]

def load_math(max_q, min_score=2):
    import pyarrow.parquet as pq
    from glob import glob
    files = sorted(glob(f"{MATH_DIR}/*.parquet"))
    random.shuffle(files)
    qs = []
    for fp in files[:40]:
        try:
            t = pq.read_table(fp, columns=["text", "score", "int_score"])
        except Exception:
            continue
        rows = t.to_pylist()
        random.shuffle(rows)
        for r in rows:
            text = (r.get("text") or "").strip()
            sc = r.get("score") or 0
            if sc < min_score or len(text) < 150 or len(text) > 2500:
                continue
            if not re.search(r"[?=:]", text):
                continue
            qs.append({"prompt": text[:2000], "response": "", "dataset": "sft_gold_math"})
            if len(qs) >= max_q:
                return qs
    return qs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max", type=int, default=1000)
    ap.add_argument("--math-frac", type=float, default=0.5)
    ap.add_argument("--out", default="data/_sft_gold/gold.jsonl")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--skip-file", default="data/_sft_gold/.progress.json")
    args = ap.parse_args()

    n_math = int(args.max * args.math_frac)
    n_code = args.max - n_math
    print(f"plan: {n_code} code + {n_math} math = {args.max}", flush=True)

    # resume support
    done = set()
    if Path(args.skip_file).exists():
        done = set(json.load(open(args.skip_file)).get("done", []))
        print(f"resume: {len(done)} already done", flush=True)

    code = load_leetcode(n_code)
    math = load_math(n_math)
    # math needs Qwen to write the solution (response empty)
    pending = [(q, i) for i, q in enumerate(code + math) if str(i) not in done]
    print(f"pending: {len(pending)} (code w/ ref answer: {len(code)}, math need solve: {len(math)})", flush=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    import concurrent.futures as cf

    def solve(q, i):
        if q["response"]:
            return q, i  # code: reference answer already good
        # math: Qwen writes a clean step-by-step solution (temp 0.2, deterministic)
        prompt = ("Solve this math problem step by step. Show all work, be precise.\n\n"
                  f"Problem:\n{q['prompt']}\n\nSolution:")
        try:
            ans = call_vllm(prompt, max_tokens=2048, temperature=0.2)
            if len(ans) < 20:
                return None
            q["response"] = ans
            return q, i
        except Exception as e:
            print(f"  fail {i}: {e}", flush=True)
            return None

    results = []
    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(solve, q, i): i for q, i in pending}
        for fut in cf.as_completed(futs):
            r = fut.result()
            if r:
                q, i = r
                results.append(q)
                done.add(str(i))
                with open(args.skip_file, "w") as f:
                    json.dump({"done": sorted(done, key=int)}, f)
                with open(out_path, "a") as f:
                    f.write(json.dumps(q) + "\n")
                if len(results) % 50 == 0:
                    print(f"  {len(results)} done, "
                          f"{len(results)/(time.time()-start):.2f} docs/s", flush=True)

    print(f"DONE: {len(results)} examples → {out_path}", flush=True)

if __name__ == "__main__":
    start = time.time()
    main()