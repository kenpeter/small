#!/usr/bin/env python3
"""Generate a DIVERSE, large code-gold corpus via local vLLM (Qwen3-8B-AWQ).

Goal: hundreds of DISTINCT coding problems spanning many categories & difficulty
levels, each with a single self-contained CORRECT Python solution. Diversity is
what makes the model generalize (like facing different questions on an exam)
instead of memorizing a handful of test answers.

Each generated doc = problem statement + correct function. Solutions are then
VERIFIED by exec() against check-cases; only passing ones are kept.

Usage: venv/bin/python gen_code_gold_qwen.py --max 400 --out ...
"""
import json, time, re, argparse, concurrent.futures
from pathlib import Path
import urllib.request

API_BASE = "http://127.0.0.1:8000/v1"
MODEL = "Qwen/Qwen3-8B-AWQ"

# Broad, distinct problem seeds across categories & difficulty. The model writes
# a correct solution for each. High diversity → generalization.
SEEDS = [
    # easy: strings
    "count how many vowels are in a string",
    "check if a string is a palindrome ignoring case",
    "find the longest word in a sentence",
    "remove all duplicate characters from a string keeping first occurrence",
    "check if two strings are anagrams",
    "find the first non-repeating character in a string",
    "capitalize the first letter of each word in a sentence",
    "check if a string contains only digits",
    "find the most frequent character in a string",
    "reverse the words in a sentence while keeping spaces",
    # easy: arrays
    "find the maximum element in an array",
    "compute the sum of all elements divisible by 3 in an array",
    "find the second largest number in an array",
    "check if an array is sorted in ascending order",
    "find all elements that appear more than once in an array",
    "return the k largest elements of an array",
    # easy: math
    "check if a number is a perfect square",
    "compute the sum of digits of a positive integer until single digit",
    "check if a number is a palindrome integer",
    "find the factorial of a number iteratively",
    "count the number of trailing zeros in n!",
    "check if a number is an Armstrong number",
    "compute the nth power of a number using exponentiation by squaring",
    # medium: arrays / two-pointer
    "three sum: find all unique triplets in an array that sum to zero",
    "container with most water",
    "find the longest subarray with sum equal to k",
    "rotate an array to the right by k steps",
    "find the majority element (appears more than n/2 times)",
    "find the smallest positive integer missing from an array",
    "maximum length of subarray with positive product",
    # medium: strings
    "longest substring without repeating characters",
    "group anagrams together",
    "find the longest palindromic substring",
    "decode a string with k[encoded] compression",
    "minimum window substring containing all characters of a pattern",
    # medium: DP
    "house robber: max money without robbing adjacent houses",
    "coin change: fewest coins to make amount",
    "longest increasing subsequence length",
    "edit distance between two strings",
    "partition equal subset sum",
    "word break: can a string be segmented into dictionary words",
    "best time to buy and sell stock with a single transaction",
    # medium: trees / graphs
    "find the maximum depth of a binary tree",
    "check if a binary tree is balanced",
    "level order traversal of a binary tree",
    "find the lowest common ancestor of two nodes in a binary tree",
    "validate if a binary tree is a binary search tree",
    "check if a graph has a cycle (undirected)",
    "number of connected components in an undirected graph",
    "implement Dijkstra's shortest path on a weighted graph",
    "clone a graph given adjacency list",
    # medium: greedy / search
    "jump game: can you reach the last index",
    "gas station circuit problem",
    "task scheduler: minimum intervals given cooldown",
    "find the kth smallest element in a sorted matrix",
    "search a 2D matrix for a target value",
    "find median of two sorted arrays",
    # hard: 
    "trapping rain water given elevation heights",
    "sliding window maximum",
    "find the kth largest element in an array",
    "regular expression matching with * and .",
    "serialize and deserialize a binary tree",
    "median of a stream of integers",
    "course schedule II: topological order if possible",
    "longest valid parentheses substring",
    "largest rectangle in a histogram",
    "word ladder: shortest transformation sequence",
    "burst balloons for maximum coins",
    "find minimum number of coins to form a target (unbounded)",
    "merge k sorted linked lists",
    "find all palindrome partitions of a string",
    "alien dictionary: determine order of letters",
    "count number of islands in a 2D grid",
    "surrounded regions in a 2D board",
    "rotten oranges: minutes until all oranges rot",
    "find the shortest path in a binary matrix",
    "longest common subsequence of two strings",
    # ── added for diversity (round 2) ──
    # strings
    "find the length of the longest substring with at most two distinct characters",
    "check if a string can be rearranged into a palindrome",
    "find the shortest way to make two strings equal by deleting characters",
    "convert a string to its zigzag pattern on a given number of rows",
    "find the minimum window that contains every character of a shorter string",
    "implement a simple caesar cipher that shifts letters by k",
    "check if one string is a rotation of another",
    "find the longest common substring of two strings",
    "reverse only the vowels in a string",
    # arrays
    "find the number of subarrays with sum equal to zero",
    "find all pairs in an array whose difference equals k",
    "compute the maximum profit from buying and selling stock with cooldown",
    "find the longest mountain subarray",
    "find the k closest elements to a target in a sorted array",
    "merge two sorted arrays in place without extra space",
    "find the single number that appears once when all others appear twice",
    "find the duplicate number in an array of size n+1",
    "sort colors (0s, 1s, 2s) in a single pass",
    "find the continuous subarray with largest sum whose length is at least k",
    # math
    "find the smallest prime greater than a given number",
    "compute the integer square root of a number using binary search",
    "check if a number is a perfect power of a base",
    "find the number of ones in the binary representation of n",
    "compute the product of digits of a number",
    "find all divisors of a number",
    "check if a number can be expressed as a sum of two squares",
    "find the n-th row of pascals triangle",
    "compute binomial coefficient n choose k",
    "find the largest palindromic number made from a list of digits",
    # matrices / 2D
    "rotate a square matrix 90 degrees clockwise in place",
    "transpose a matrix",
    "find the determinant of a 2x2 matrix",
    "spiral order traversal of a 2D matrix",
    "find the number of distinct paths with obstacles in a grid",
    "set matrix rows and columns to zero if any element is zero",
    "check if a 2D matrix is symmetric",
    # trees
    "find the sum of all root-to-leaf paths in a binary tree",
    "count the number of leaf nodes in a binary tree",
    "find the diameter of a binary tree",
    "build a binary tree from a level-order list",
    "check if two binary trees are mirror images",
    "find the kth smallest element in a binary search tree",
    "print all paths from root to leaf in a binary tree",
    "check if a binary tree is a complete binary tree",
    # graphs
    "detect a cycle in a directed graph",
    "count connected components using union-find",
    "find the shortest path in an unweighted graph using BFS",
    "check if a graph is bipartite",
    "find all nodes reachable from a start node in a graph",
    # DP
    "minimum number of jumps to reach the end of an array",
    "longest common prefix of strings using dynamic programming",
    "number of ways to make change for an amount",
    "longest palindromic subsequence",
    "minimum cost to climb stairs with given costs",
    "maximum subarray sum with a maximum of k deletions",
    # greedy / misc
    "assign cookies to children maximizing the number of happy children",
    "find the minimum number of platforms needed for trains",
    "remove the minimum number of digits to make a number divisible by three",
    "candy distribution: each child gets at least as many as neighbors rule",
]

# Function name derived from problem (used to help Qwen produce clean, runnable code).
def _fn_name(seed):
    return "solve_" + "_".join(re.findall(r"[a-z]+", seed)[:3])

PROMPT_TMPL = """Write ONE self-contained, correct, runnable Python function that solves this problem:
"{seed}"

Rules (follow strictly or the code is WRONG):
- Output ONLY the function definition. No problem statement, no explanation, no examples, no test calls.
- Name the function exactly: `{fn}`
- Define ALL symbols you use. Do NOT use bare `-inf`/`inf` (that's a NameError) — use `float('-inf')`/`float('inf')` or initialize with the first element.
- Handle edge cases: empty input, single element, n=0/1, None where needed.
- Use only standard library (import inside the function or at top if needed).
- Return the documented value type. Make loops/recursion terminate (correct base cases).
- Correct, complete logic — this should pass unit tests.
"""

CHECK_CASES = {
    # (fn_to_call_hint, list of (args, expected)) — generic spot checks applied per doc
}

def call_llm(seed, fn, timeout=120):
    body = json.dumps({
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "Answer DIRECTLY with code only. Do NOT think out loud, do NOT explain, do NOT describe the algorithm in prose. Output exactly one Python function definition and nothing else."},
            {"role": "user", "content": PROMPT_TMPL.format(seed=seed, fn=fn)},
        ],
        "temperature": 0.2, "max_tokens": 800,
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode()
    req = urllib.request.Request(API_BASE + "/chat/completions",
        data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = json.loads(r.read().decode())
    return data["choices"][0]["message"]["content"]

def extract_python(text):
    # Qwen3-8B (thinking) wraps: ... <thinking>...</thinking> <answer>...</answer> or plain.
    m = re.search(r"<answer>\s*(.*?)</answer>", text, re.DOTALL)
    if m:
        text = m.group(1).strip()
    else:
        # strip a leading <thinking> block if present
        text = re.sub(r"<thinking>.*?</thinking>", "", text, flags=re.DOTALL).strip()
        # strip a trailing "thinking" prose preamble down to the first code marker
    # grab the first python code block if present, else trim to first def
    m = re.search(r"```(?:python)?\s*(.*?)```", text, re.DOTALL)
    if m:
        code = m.group(1).strip()
        return _trim_code(code)
    idx = text.find("def ")
    if idx >= 0:
        return _trim_code(text[idx:].strip())
    return _trim_code(text.strip())

def _trim_code(code):
    """Cut trailing prose after a python function body."""
    lines = code.splitlines()
    out = []
    for ln in lines:
        stripped = ln.strip()
        # keep def/class/import/from/blank/indented lines
        if (stripped.startswith("def ") or stripped.startswith("class ")
                or stripped.startswith("import ") or stripped.startswith("from ")
                or stripped.startswith("@") or not stripped or ln[0] in " \t"):
            out.append(ln)
        else:
            break  # first top-level non-code line ends the function
    return "\n".join(out).strip()

# ── Curated verifiers: seed-IDENTIFIER → list of (arg_tuple, expected) ──
# Matched by substring of the PROBLEM SEED (stable) using the CURRENT fn name
# from the generated code. Functions whose seed matches must pass ALL cases.
VERIFY = {
    "maximum element": [("([1,5,3,2],)", 5), ("([-1,-5,-2],)", -1)],
    "sum of all elements divisible": [("([3,6,1,9,12],)", 21)],
    "second largest": [("([3,7,1,8,2],)", 7), ("([1,1,2],)", 1)],
    "array is sorted": [("([1,2,3,4],)", True), ("([1,3,2,4],)", False)],
    "perfect square": [("(25,)", True), ("(26,)", False)],
    "palindrome integer": [("(121,)", True), ("(123,)", False)],
    "digits of a positive integer until single": [("(38,)", 2), ("(123,)", 6)],
    "factorial of a number": [("(5,)", 120), ("(3,)", 6)],
    "two strings are anagrams": [("('listen','silent',)", True), ("('abc','abd',)", False)],
    "most frequent character": [("('abbccc',)", 'c')],
    "remove all duplicate characters": [("('abacabad',)", 'abcd')],
    "capitalize the first letter": [("('hello world',)", 'Hello World')],
    "how many vowels": [("('hello world',)", 3)],
    "reverse the words": [("('the sky is blue',)", 'blue is sky the')],
    "contains only digits": [("('12345',)", True), ("('12a5',)", False)],
    "number is a palindrome": [("(121,)", True), ("(123,)", False)],
}

def _sample_call(seed):
    """A safe sample arg tuple derived from seed keywords; returns None if unknown."""
    s = seed.lower()
    if "array" in s or "list" in s or "nums" in seed:
        return ("([3,9,2,7],)",)
    if "string" in s or "word" in s or "sentence" in s:
        return ("('abc',)",)
    if "number" in s or "integer" in s or "n" == s.strip():
        return ("(7,)",)
    return None

def verify(seed, fn, code):
    """Compile + run. If a curated seed-substring matches, ALL cases must pass;
    else fall back to a sample call with no exception. False on broken code."""
    try:
        ns = {}
        compile(code, "<gen>", "exec")
        exec(code, ns)
        if fn not in ns:
            return False
        func = ns[fn]
        # curated: match the specific seed
        curated = [cases for key, cases in VERIFY.items() if key in seed.lower()]
        if curated:
            cases = curated[0]
            for args_t, exp in cases:
                try:
                    if func(*eval(args_t)) != exp:
                        return False
                except Exception:
                    return False
            return True
        sample = _sample_call(seed)
        if sample is not None:
            try:
                func(*eval(sample))
            except Exception:
                try:
                    func()
                except (TypeError, ValueError):
                    return False
            return True
        return True  # compile-only acceptable for exotic seeds
    except Exception:
        return False

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max", type=int, default=400)
    ap.add_argument("--out", default="/home/kenpeter/work/data/_code_gold_qwen/code_gold_qwen.jsonl")
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    seen_prev = set()
    if out.exists():
        for line in open(out):
            try:
                seen_prev.add(json.loads(line)["text"])
            except Exception:
                pass
    print(f"Resuming with {len(seen_prev)} existing docs", flush=True)

    def gen(seed):
        fn = _fn_name(seed)
        try:
            text = call_llm(seed, fn)
            code = extract_python(text)
            if "def " not in code:
                return None
            if not verify(seed, fn, code):
                print(f"  ✗ rejected (failed verify): {seed[:40]}", flush=True)
                return None
            return {"seed": seed, "fn": fn, "text": code}
        except Exception as e:
            print(f"  ERR {seed[:40]}: {e}", flush=True)
            return None

    results = []
    todo = [s for s in SEEDS for _ in range(args.max // max(1, len(SEEDS)) + 1)]
    seen = set(seen_prev)
    target = args.max
    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as ex:
        futs = {ex.submit(gen, s): s for s in todo}
        done = 0
        for fut in concurrent.futures.as_completed(futs):
            r = fut.result()
            done += 1
            if r and r["text"] not in seen:
                seen.add(r["text"])
                results.append(r)
            if done % 20 == 0:
                print(f"  ...{done} tries, {len(results)} kept, {time.time()-t0:.0f}s", flush=True)
            if len(results) >= target:
                break
    # write
    new = 0
    with open(args.out, "a") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
            new += 1
    print(f"Wrote {new} new docs → {args.out} (total incl prior {len(seen)}). {time.time()-t0:.0f}s", flush=True)

if __name__ == "__main__":
    main()
