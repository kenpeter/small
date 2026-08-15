#!/usr/bin/env python3
"""
Gold-set generator: for each high-quality question, generate
  1) 3 distinct solution methods (same problem, 3 approaches)
  2) 5 similar variant questions (same type, different numbers/context)
via local vLLM (Qwen3-8B-AWQ). Resume-able, incremental output.

Usage:
  python3 gold_gen.py --leetcode data/high_quality_leetcode/train.jsonl \
      --math-dir /mnt/file_drive/data/finemath-3plus \
      --max-questions 400 --out data/_gold/gold_set.jsonl
"""

import sys, os, json, time, re, random, argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import urllib.request

API_BASE = "http://127.0.0.1:8000/v1"
API_KEY = "token-abc123"
MODEL = "Qwen/Qwen3-8B-AWQ"

# ─── Qwen3-safe prompts (direct, no role-play, no meta-commentary) ───

PROMPT_METHODS = (
    "Solve the following problem in THREE completely different ways. "
    "Number them METHOD 1, METHOD 2, METHOD 3. Each method must use a "
    "genuinely different approach (different algorithm, different reasoning "
    "path, or different mathematical technique), not just reworded steps. "
    "Show the full working for each method. "
    "START IMMEDIATELY with 'METHOD 1'. Do NOT include any planning, thinking, "
    "or meta-commentary.\n\nPROBLEM:\n{question}"
)

PROMPT_VARIANTS = (
    "Create FIVE similar questions to the problem below. Each variant must: "
    "- use the SAME solution technique / method type as the original\n"
    "- change the numbers, names, objects, or context\n"
    "- be solvable with the same approach\n"
    "- be a complete, standalone question\n"
    "Number them VARIANT 1 through VARIANT 5. "
    "START IMMEDIATELY with 'VARIANT 1'. Do NOT include any planning, thinking, "
    "or meta-commentary.\n\nORIGINAL PROBLEM:\n{question}"
)

PROMPT_MATH_SOLVE = (
    "Solve the following math problem step by step, clearly showing the "
    "reasoning. START IMMEDIATELY with the solution. Do NOT include any "
    "planning, thinking, or meta-commentary.\n\nPROBLEM:\n{question}"
)


def strip_think(text):
    """Remove Qwen3 <think>...</think> reasoning blocks."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def call_vllm(prompt, max_tokens=2048, temperature=0.4, retries=3):
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


def load_leetcode(path, max_q):
    """Load LeetCode problems; prefer Medium/Hard for 'best of best'."""
    questions = []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            if not d.get("problem_description") or not d.get("high_quality_cot"):
                continue
            questions.append({
                "source": "leetcode",
                "domain": "code",
                "difficulty": d.get("difficulty", "?"),
                "tags": d.get("tags", []),
                "question": d["problem_description"][:3000],
                "solution": d["high_quality_cot"][:4000],
            })
    random.shuffle(questions)
    return questions[:max_q]


def load_math(parquet_dir, max_q, min_score=2):
    """Load math questions from finemath parquets with quality score >= min_score."""
    import pyarrow.parquet as pq
    from glob import glob
    files = sorted(glob(f"{parquet_dir}/*.parquet"))
    random.shuffle(files)
    questions = []
    for fp in files[:40]:  # sample 40 files, then random rows within
        try:
            t = pq.read_table(fp, columns=["text", "score", "int_score"])
        except Exception:
            continue
        rows = t.to_pylist()
        random.shuffle(rows)
        for r in rows:
            text = (r.get("text") or "").strip()
            sc = r.get("score") or 0
            if sc < min_score or len(text) < 200 or len(text) > 3000:
                continue
            if not re.search(r"[?=:]", text):  # needs question-like content
                continue
            questions.append({
                "source": "finemath",
                "domain": "math",
                "difficulty": "hard" if sc >= 3 else "medium",
                "tags": [],
                "question": text[:2500],
                "solution": "",
            })
            if len(questions) >= max_q:
                return questions
    return questions


def build_doc(q, methods_text, variants_text, solve_text=""):
    """Assemble one gold training document from a question + generations."""
    parts = [f"# {q['domain'].upper()} QUESTION", q["question"], ""]
    if q.get("difficulty"):
        parts.insert(1, f"Difficulty: {q['difficulty']}")

    if methods_text:
        parts.append("## THREE SOLUTION METHODS")
        parts.append(methods_text)
        parts.append("")

    if solve_text:
        parts.append("## SOLUTION")
        parts.append(solve_text)
        parts.append("")

    if variants_text:
        parts.append("## SIMILAR QUESTIONS")
        parts.append(variants_text)

    return "\n".join(parts)


def _q_fp(q):
    """Fingerprint a question (first 200 chars normalized)."""
    return re.sub(r"\s+", " ", q["question"])[:200]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--leetcode", default="data/high_quality_leetcode/train.jsonl")
    ap.add_argument("--math-dir", default="/mnt/file_drive/data/finemath-3plus")
    ap.add_argument("--max-questions", type=int, default=400)
    ap.add_argument("--out", default="data/_gold/gold_set.jsonl")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--math-frac", type=float, default=0.4)
    ap.add_argument("--skip-file", default=None,
                    help="Existing gold jsonl — questions whose fingerprint "
                         "already appears there are skipped (dedup across runs)")
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_path.with_suffix(".progress.json")

    # Dedup across runs: fingerprint set of previously generated questions
    seen_fps = set()
    if args.skip_file and Path(args.skip_file).exists():
        with open(args.skip_file) as f:
            for line in f:
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                # Existing docs embed the question; fingerprint the doc's
                # first section (up to the first '##' header), stripping the
                # "# CODE QUESTION" / "Difficulty:" prefix lines so the fp
                # matches _q_fp() on the raw source question.
                txt = d.get("text", "")
                head = txt.split("##")[0]
                lines = [ln for ln in head.splitlines()
                         if ln.strip() and not ln.strip().startswith("#")
                         and not ln.strip().startswith("Difficulty:")]
                head = "\n".join(lines)
                seen_fps.add(re.sub(r"\s+", " ", head).strip()[:200])
        print(f"Dedup: {len(seen_fps)} previously generated docs loaded", flush=True)

    # Load sources
    n_math = int(args.max_questions * args.math_frac)
    n_code = args.max_questions - n_math
    print(f"Loading sources: {n_code} code + {n_math} math...", flush=True)
    qs = load_leetcode(args.leetcode, n_code * 3)  # oversample, dedup below
    qs += load_math(args.math_dir, n_math * 3)
    random.shuffle(qs)
    # Dedup against prior runs + within this batch
    batch_seen = set()
    filtered = []
    for q in qs:
        fp = _q_fp(q)
        if fp in seen_fps or fp in batch_seen:
            continue
        batch_seen.add(fp)
        filtered.append(q)
        if len(filtered) >= args.max_questions:
            break
    qs = filtered
    print(f"Loaded {len(qs)} gold questions (after dedup)", flush=True)
    # Resume support
    done_ids = set()
    if ckpt_path.exists():
        done_ids = set(json.loads(ckpt_path.read_text())["done"])
        print(f"Resuming: {len(done_ids)} already done", flush=True)

    def process(q, idx):
        if str(idx) in done_ids:
            return None
        try:
            methods = call_vllm(PROMPT_METHODS.format(question=q["question"]),
                                max_tokens=2500, temperature=0.4)
            variants = call_vllm(PROMPT_VARIANTS.format(question=q["question"]),
                                 max_tokens=1800, temperature=0.7)
            solve = ""
            if q["domain"] == "math":
                solve = call_vllm(PROMPT_MATH_SOLVE.format(question=q["question"]),
                                  max_tokens=1200, temperature=0.2)
            doc = build_doc(q, methods, variants, solve)
            return {"idx": idx, "domain": q["domain"], "difficulty": q.get("difficulty"),
                    "tags": q.get("tags", []), "text": doc,
                    "meta": {"source": q["source"]}}
        except Exception as e:
            print(f"  ERROR idx {idx}: {e}", flush=True)
            return {"idx": idx, "error": str(e)}

    t0 = time.time()
    with open(out_path, "a") as fout:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(process, q, i): i for i, q in enumerate(qs)}
            done = 0
            for fut in as_completed(futures):
                idx = futures[fut]
                res = fut.result()
                if res is None:
                    continue
                if "error" in res:
                    continue
                fout.write(json.dumps(res) + "\n")
                fout.flush()
                done_ids.add(idx)
                done += 1
                if done % 10 == 0:
                    ckpt_path.write_text(json.dumps({"done": sorted(done_ids)}))
                    rate = done / (time.time() - t0)
                    print(f"  [{done}/{len(qs)}] {rate:.2f} docs/s, "
                          f"ETA {(len(qs)-done)/rate/60:.1f} min", flush=True)

    ckpt_path.write_text(json.dumps({"done": sorted(done_ids)}))
    print(f"Done. {len(done_ids)} docs -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
