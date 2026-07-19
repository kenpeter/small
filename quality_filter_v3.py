#!/usr/bin/env python3
"""
quality_filter_v3.py — Fast, best-of-breed data quality pipeline.

Pipeline: parquet → heuristic filter → URL dedup → exact dedup → prefix dedup
          → perplexity scoring (optional) → mixture → tokenized .bin shards

Key design: multiprocessing for CPU-bound filtering, single-process for dedup,
            optional GPU perplexity scoring, streaming tokenization.
"""
import os, sys, json, gc, hashlib, re, math, time, struct
import string
from pathlib import Path
from collections import defaultdict
from multiprocessing import Pool, cpu_count
from functools import partial

import numpy as np

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ─── Paths ──────────────────────────────────────────────────────────
STAGING_DIR = Path("/home/kenpeter/work/data/_staging_multi")
FILTERED_DIR = Path("/home/kenpeter/work/data/_filtered_best")
SHARDS_DIR = Path("/home/kenpeter/work/data/_shards_final")
CKPT_DIR = Path("/home/kenpeter/work/checkpoints")

FILTERED_DIR.mkdir(parents=True, exist_ok=True)
SHARDS_DIR.mkdir(parents=True, exist_ok=True)

# ─── Dataset Config ──────────────────────────────────────────────────
DATASETS = {
    "fineweb-edu": {
        "text_col": "text", "url_col": "url", "is_code": False, "is_math": False,
        "mixture": 0.45, "min_words": 50, "max_words": 50000,
        "lang_score_min": 0.8, "quality_score_min": 3.0,
    },
    "finemath-3plus": {
        "text_col": "text", "url_col": "url", "is_code": False, "is_math": True,
        "mixture": 0.20, "min_words": 30, "max_words": 100000,
        "lang_score_min": 0.5, "quality_score_min": 0.0,
    },
    "cosmopedia": {
        "text_col": "text", "url_col": "", "is_code": False, "is_math": False,
        "mixture": 0.12, "min_words": 50, "max_words": 50000,
    },
    "open-web-math": {
        "text_col": "text", "url_col": "url", "is_code": False, "is_math": True,
        "mixture": 0.10, "min_words": 30, "max_words": 100000,
    },
    "finemath": {
        "text_col": "text", "url_col": "url", "is_code": False, "is_math": True,
        "mixture": 0.08, "min_words": 30, "max_words": 100000,
        "lang_score_min": 0.5, "quality_score_min": 0.0,
    },
    "stack-python": {
        "text_col": "content", "url_col": "", "is_code": True, "is_math": False,
        "mixture": 0.05, "min_words": 10, "max_words": 100000,
    },
}

STOP_WORDS = {
    'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'i',
    'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at',
    'this', 'but', 'his', 'by', 'from', 'they', 'we', 'say', 'her', 'she',
    'or', 'an', 'will', 'my', 'one', 'all', 'would', 'there', 'their',
    'is', 'are', 'was', 'were', 'been', 'has', 'had', 'can', 'may', 'also',
    'about', 'up', 'out', 'if', 'than', 'some', 'very', 'just', 'like',
}

# ─── Quality Filters ────────────────────────────────────────────────
def quality_filter(text: str, cfg: dict) -> tuple:
    """Returns (True, 'pass') or (False, reason_code)."""
    if not text or not isinstance(text, str):
        return False, "empty"
    text = text.strip()
    if not text:
        return False, "empty"

    n_words = len(text.split())
    if n_words < cfg["min_words"]:
        return False, f"too_short:{n_words}"
    if n_words > cfg["max_words"]:
        return False, f"too_long:{n_words}"

    total_chars = len(text)
    
    # Fast char repetition check (no Counter)
    if total_chars > 0:
        char_max = 0
        char_counts = {}
        for c in text:
            if c not in ' \n\t\r':
                char_counts[c] = char_counts.get(c, 0) + 1
        if char_counts:
            max_count = max(char_counts.values())
            total_non_ws = sum(char_counts.values())
            if max_count / max(total_non_ws, 1) > 0.25:
                return False, "char_repeat"
            # Check single-char dominance
            vals = sorted(char_counts.values(), reverse=True)
            if len(vals) >= 2 and vals[0] > vals[1] * 5 and vals[1] > 5:
                return False, "char_dom"

    newline_ratio = text.count('\n') / max(total_chars, 1)
    if newline_ratio > 0.35:
        return False, f"nl:{newline_ratio:.2f}"

    # URL density check (fast: count occurrences of 'http')
    url_count = text.count('http://') + text.count('https://')
    if url_count > 20:
        return False, f"urls:{url_count}"

    # Per-type filtering
    if cfg.get("is_code"):
        return _filter_code(text, n_words)
    elif cfg.get("is_math"):
        return _filter_math(text, n_words)
    else:
        return _filter_text(text, n_words)

def _filter_text(text: str, n_words: int) -> tuple:
    words = text.split()
    
    avg_wlen = sum(len(w) for w in words) / max(n_words, 1)
    if avg_wlen < 2.5 or avg_wlen > 12:
        return False, f"avg_wlen:{avg_wlen:.1f}"
    
    sym_words = sum(1 for w in words if any(c in string.punctuation for c in w))
    if sym_words / max(n_words, 1) > 0.40:
        return False, f"sym_words:{sym_words/n_words:.2f}"

    stops = len(set(w.lower() for w in words) & STOP_WORDS)
    if stops < 3:
        return False, f"stops:{stops}"
    
    alpha_words = sum(1 for w in words if any(c.isalpha() for c in w))
    if alpha_words / max(n_words, 1) < 0.15:
        return False, f"alpha_ratio:{alpha_words/n_words:.2f}"
    
    lines = text.split('\n')
    n_lines = len(lines)
    if n_lines < 3:
        return False, "few_lines"
    
    # Bullet lines
    bullet_count = sum(1 for l in lines if l.strip() and l.strip()[0] in '-*•·')
    if bullet_count / max(n_lines, 1) > 0.60:
        return False, f"bullets:{bullet_count/n_lines:.2f}"
    
    # Lines ending with punctuation
    punct_end = sum(1 for l in lines if l.strip() and l.strip()[-1] in '.!?;')
    if punct_end / max(n_lines, 1) < 0.08:
        return False, f"punct_end:{punct_end/n_lines:.2f}"
    
    # Short lines
    short_lines = sum(1 for l in lines if l.strip() and len(l.strip()) < 20)
    if short_lines / max(n_lines, 1) > 0.75:
        return False, f"short_lines:{short_lines/n_lines:.2f}"
    
    return True, "pass"

def _filter_math(text: str, n_words: int) -> tuple:
    words = text.split()
    
    has_numbers = bool(re.search(r'\d', text))
    has_math_sym = bool(re.search(r'[=+\-*/^∫∑√πθελ]', text))
    if not has_numbers and not has_math_sym:
        return False, "no_math"
    
    avg_wlen = sum(len(w) for w in words) / max(n_words, 1)
    if avg_wlen < 2.0 or avg_wlen > 15:
        return False, f"avg_wlen:{avg_wlen:.1f}"
    
    alpha_words = sum(1 for w in words if any(c.isalpha() for c in w))
    if alpha_words / max(n_words, 1) < 0.08:
        return False, f"alpha:{alpha_words/n_words:.2f}"
    
    return True, "pass"

def _filter_code(text: str, n_words: int) -> tuple:
    lines = text.split('\n')
    if len(lines) < 5:
        return False, "few_lines"
    
    imports = sum(1 for l in lines if l.strip().startswith(('import ', 'from ', '#include', 'using ', 'def ', 'class ', 'function ', 'public ', 'private ', 'fn ', 'pub ')))
    if imports == 0:
        keywords = {'if', 'else', 'for', 'while', 'return', 'try', 'catch', 'print', 'var', 'let', 'const', 'fn', 'def', 'end', 'do'}
        if not any(k in text.lower() for k in keywords):
            return False, "no_code_keywords"
    
    comment_lines = sum(1 for l in lines if l.strip().startswith(('#', '//', '/*', '*', '<!--', '\"\"\"', "'''")))
    if comment_lines / max(len(lines), 1) > 0.75:
        return False, f"comments:{comment_lines/len(lines):.2f}"
    
    return True, "pass"


# ─── Deduplication ──────────────────────────────────────────────────
class Deduplicator:
    __slots__ = ('url_seen', 'text_seen', 'prefix_seen')
    def __init__(self):
        self.url_seen = set()
        self.text_seen = set()
        self.prefix_seen = set()

    def check_url(self, url: str) -> bool:
        if not url:
            return False
        norm = url.strip().rstrip('/').lower()
        if norm in self.url_seen:
            return True
        self.url_seen.add(norm)
        return False

    def check_text(self, text: str) -> bool:
        h = hashlib.md5(text.encode('utf-8')).hexdigest()
        if h in self.text_seen:
            return True
        self.text_seen.add(h)
        return False

    def check_prefix(self, text: str) -> bool:
        prefix = text[:100]
        if prefix in self.prefix_seen:
            return True
        self.prefix_seen.add(prefix)
        return False


# ─── Parallel Pipeline ────────────────────────────────────────────
def filter_file_worker(args):
    """Worker: filter one parquet file, write to temp JSONL. Returns temp path + doc count."""
    fpath, cfg, tmp_dir = args
    import pyarrow.parquet as pq
    import tempfile
    
    tmp_fd, tmp_path = tempfile.mkstemp(suffix='.jsonl', dir=str(tmp_dir), prefix=f'w_{os.getpid()}_')
    os.close(tmp_fd)
    
    count = 0
    try:
        pf = pq.ParquetFile(str(fpath))
        cols = [cfg["text_col"]]
        if cfg.get("url_col"):
            cols.append(cfg["url_col"])
        with open(tmp_path, 'w', encoding='utf-8') as fout:
            for batch in pf.iter_batches(batch_size=5000, columns=cols):
                for row in batch.to_pylist():
                    text = row.get(cfg["text_col"], '') or ''
                    if not isinstance(text, str):
                        text = str(text)
                    ok, _ = quality_filter(text, cfg)
                    if not ok:
                        continue
                    url = str(row.get(cfg.get("url_col", ""), '')) if cfg.get("url_col") else ''
                    fout.write(json.dumps({"text": text, "url": url}) + '\n')
                    count += 1
    except Exception:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass
        return None, 0
    
    if count == 0:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass
        return None, 0
    return tmp_path, count


def process_dataset_stage1(ds_name: str, cfg: dict, dedup: Deduplicator, n_workers: int = 3):
    """
    Stage 1: parallel file processing. Workers write to temp files (low memory).
    Main process reads temp files → dedup → JSONL.
    """
    print(f"\n  [{ds_name}] Config: min_words={cfg['min_words']}, max_words={cfg['max_words']}, "
          f"is_math={cfg.get('is_math', False)}, is_code={cfg.get('is_code', False)}")
    
    src_dir = STAGING_DIR / ds_name
    if not src_dir.exists():
        print(f"    SKIP: no directory")
        return 0
    
    parquet_files = sorted(src_dir.glob("*.parquet"))
    if not parquet_files:
        parquet_files = sorted(src_dir.glob("**/*.parquet"))
    if not parquet_files:
        print(f"    SKIP: no parquet files")
        return 0
    
    n_files = len(parquet_files)
    out_dir = FILTERED_DIR / ds_name
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = FILTERED_DIR / f"_tmp_{ds_name}"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    
    N_WORKERS = n_workers
    file_args = [(f, cfg, tmp_dir) for f in parquet_files]
    
    total_candidates = 0
    total_kept = 0
    total_dup_url = total_dup_text = total_dup_prefix = 0
    n_shards = 0
    current_shard = []
    MAX_PER_SHARD = 25000
    t0 = time.time()
    
    from concurrent.futures import ProcessPoolExecutor, as_completed
    
    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        futures = {pool.submit(filter_file_worker, fa): i for i, fa in enumerate(file_args)}
        done_count = 0
        
        for future in as_completed(futures):
            idx = futures[future]
            fi = idx + 1
            ft0 = time.time()
            fname = parquet_files[idx].name
            
            try:
                tmp_path, n_candidates = future.result()
            except Exception as e:
                print(f"    [{fi}/{n_files}] {fname}: ERROR {e}", flush=True)
                continue
            
            if tmp_path is None or n_candidates == 0:
                print(f"    [{fi}/{n_files}] {fname}: 0 candidates", flush=True)
                done_count += 1
                continue
            
            total_candidates += n_candidates
            
            # Read temp file → dedup → write
            file_kept = file_dup_url = file_dup_text = file_dup_prefix = 0
            with open(tmp_path, 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line)
                    text = data["text"]
                    url = data.get("url", "")
                    
                    if url and dedup.check_url(url):
                        file_dup_url += 1
                        continue
                    if dedup.check_text(text):
                        file_dup_text += 1
                        continue
                    if dedup.check_prefix(text):
                        file_dup_prefix += 1
                        continue
                    
                    current_shard.append(text)
                    file_kept += 1
                    
                    if len(current_shard) >= MAX_PER_SHARD:
                        _write_jsonl_shard(out_dir, n_shards, current_shard)
                        n_shards += 1
                        current_shard = []
            
            # Clean up temp file
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
            
            total_kept += file_kept
            total_dup_url += file_dup_url
            total_dup_text += file_dup_text
            total_dup_prefix += file_dup_prefix
            
            elapsed = time.time() - ft0
            done_count += 1
            print(f"    [{fi}/{n_files}] {fname}: {n_candidates:,} candidates, {file_kept:,} kept "
                  f"(url={file_dup_url}, text={file_dup_text}, prefix={file_dup_prefix}) "
                  f"[{elapsed:.0f}s]", flush=True)
    
    # Flush final shard
    if current_shard:
        _write_jsonl_shard(out_dir, n_shards, current_shard)
        n_shards += 1
    
    # Cleanup tmp dir
    try:
        for f in tmp_dir.iterdir():
            f.unlink()
        tmp_dir.rmdir()
    except Exception:
        pass
    
    total_time = time.time() - t0
    print(f"    [{ds_name}] Done: {total_candidates:,} candidates, {total_kept:,} kept "
          f"({total_kept/max(total_candidates,1)*100:.1f}%), "
          f"dup: url={total_dup_url:,} text={total_dup_text:,} prefix={total_dup_prefix:,} "
          f"[{total_time:.0f}s]", flush=True)
    print(f"    Output: {n_shards} JSONL shards in {out_dir}", flush=True)
    
    return total_kept


def _write_jsonl_shard(out_dir: Path, shard_idx: int, texts: list):
    """Write texts to a JSONL shard file."""
    out_path = out_dir / f"shard_{shard_idx:06d}.jsonl"
    with open(out_path, 'w', encoding='utf-8') as f:
        for t in texts:
            f.write(json.dumps({"text": t}, ensure_ascii=False) + '\n')
    size_mb = out_path.stat().st_size / (1024*1024)
    print(f"    → {out_path.name}: {len(texts):,} docs, {size_mb:.1f}MB", flush=True)


def write_filtered_jsonl(ds_name: str, texts: list, max_per_shard=50000):
    """Write to JSONL shards."""
    out_dir = FILTERED_DIR / ds_name
    out_dir.mkdir(parents=True, exist_ok=True)
    
    n_shards = 0
    for i in range(0, len(texts), max_per_shard):
        chunk = texts[i:i+max_per_shard]
        out_path = out_dir / f"shard_{n_shards:06d}.jsonl"
        with open(out_path, 'w', encoding='utf-8') as f:
            for t in chunk:
                f.write(json.dumps({"text": t}, ensure_ascii=False) + '\n')
        size_mb = out_path.stat().st_size / (1024*1024)
        print(f"    Wrote {out_path.name}: {len(chunk):,} docs, {size_mb:.1f}MB", flush=True)
        n_shards += 1
    return n_shards


# ─── Perplexity Scoring (Stage 2) ──────────────────────────────────
def run_perplexity_scoring(ds_name: str, checkpoint_path: str, keep_ratio: float = 0.60):
    """Score docs by model perplexity, keep best K%. Returns kept texts."""
    import torch
    from transformers import AutoTokenizer
    sys.path.insert(0, str(Path(__file__).parent))
    from train import TrainConfig, SmolLM2
    
    ds_dir = FILTERED_DIR / ds_name
    if not ds_dir.exists():
        return 0
    
    jsonl_files = sorted(ds_dir.glob("*.jsonl"))
    texts = []
    for jf in jsonl_files:
        with open(jf, 'r') as f:
            for line in f:
                texts.append(json.loads(line)["text"])
    
    print(f"    {ds_name}: loaded {len(texts):,} docs for scoring")
    if len(texts) < 1000:
        return len(texts)  # too few to bother
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"    Loading model on {device}...", flush=True)
    cfg = TrainConfig()
    model = SmolLM2(cfg).to(device)
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_state = state["model_state_dict"]
    unwanted = "_orig_mod."
    for k, v in list(model_state.items()):
        if k.startswith(unwanted):
            model_state[k[len(unwanted):]] = model_state.pop(k)
    model.load_state_dict(model_state)
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M", trust_remote_code=True)
    seq_len = 2048
    
    all_scores = []
    batch_size = min(16, len(texts) // 10 + 1)
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        batch_losses = []
        
        for text in batch:
            ids = tokenizer.encode(text, add_special_tokens=False, truncation=True, max_length=seq_len)
            if len(ids) < 20:
                batch_losses.append(10.0)
                continue
            tokens = torch.tensor([ids], dtype=torch.long, device=device)
            with torch.no_grad():
                try:
                    _, loss = model(tokens[:, :-1], tokens[:, 1:])
                    batch_losses.append(loss.item())
                except Exception:
                    batch_losses.append(10.0)
        
        for loss in batch_losses:
            all_scores.append(loss)
        
        if (i // batch_size) % 100 == 0:
            print(f"    Scored {i+len(batch)}/{len(texts)}", end='\r', flush=True)
    
    print(f"    Scoring done: avg loss={np.mean(all_scores):.4f}", flush=True)
    
    # Keep best K%
    scored = list(zip(all_scores, texts))
    scored.sort(key=lambda x: x[0])
    keep_n = max(int(len(scored) * keep_ratio), 100)
    kept = [t for _, t in scored[:keep_n]]
    cutoff = scored[keep_n-1][0] if keep_n > 0 else 0
    
    print(f"    Kept {len(kept):,}/{len(scored):,} ({keep_ratio*100:.0f}%), loss cutoff ≤{cutoff:.4f}", flush=True)
    
    # Rewrite
    for f in jsonl_files:
        f.unlink()
    write_filtered_jsonl(ds_name, kept)
    
    del model, texts, scored, kept
    torch.cuda.empty_cache()
    gc.collect()
    
    return len(kept)


# ─── Mixture Enforcement ───────────────────────────────────────────
def enforce_mixture(target_tokens: int = 15_000_000_000):
    """Downsample per-dataset to achieve target mixture ratios."""
    from transformers import AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M", trust_remote_code=True)
    
    # Load all texts
    all_texts = {}
    doc_stats = {}
    for ds_name, cfg in DATASETS.items():
        ds_dir = FILTERED_DIR / ds_name
        if not ds_dir.exists():
            continue
        texts = []
        for jf in sorted(ds_dir.glob("*.jsonl")):
            with open(jf, 'r') as f:
                for line in f:
                    texts.append(json.loads(line)["text"])
        if not texts:
            continue
        
        # Estimate avg tokens per doc from sample
        sample = texts[:min(500, len(texts))]
        sample_tokens = sum(len(tokenizer.encode(t, add_special_tokens=False, truncation=True, max_length=2048)) for t in sample)
        avg_tok = sample_tokens / max(len(sample), 1)
        est_tokens = int(len(texts) * avg_tok)
        doc_stats[ds_name] = {"docs": len(texts), "avg_tok": avg_tok, "est_tokens": est_tokens}
        all_texts[ds_name] = texts
    
    print(f"\n  [Mixture] Current estimated totals:")
    total_est = sum(s["est_tokens"] for s in doc_stats.values())
    for ds_name, s in sorted(doc_stats.items(), key=lambda x: -DATASETS.get(x[0], {}).get("mixture", 0)):
        pct = s["est_tokens"] / max(total_est, 1) * 100
        print(f"    {ds_name}: {s['docs']:,} docs, ~{s['est_tokens']/1e9:.2f}B tokens ({pct:.1f}%)")
    
    print(f"    TOTAL: ~{total_est/1e9:.2f}B tokens")
    print(f"    Target: {target_tokens/1e9:.1f}B tokens")
    
    # Calculate target docs per dataset
    import random
    random.seed(42)
    
    result = {}
    for ds_name, texts in all_texts.items():
        cfg = DATASETS.get(ds_name, {})
        target_pct = cfg.get("mixture", 0)
        if target_pct == 0:
            print(f"    {ds_name}: mixture=0, skipping")
            continue
        
        target_tok = int(target_tokens * target_pct)
        stats = doc_stats[ds_name]
        target_docs = min(int(target_tok / max(stats["avg_tok"], 1)), len(texts))
        
        if target_docs < len(texts):
            selected = random.sample(texts, target_docs)
            print(f"    {ds_name}: {len(texts):,} → {target_docs:,} docs ({target_pct*100:.0f}% mix, ~{target_tok/1e9:.2f}B tok)")
        else:
            selected = texts
            print(f"    {ds_name}: {len(texts):,} docs (keep all, {target_pct*100:.0f}% mix)")
        
        result[ds_name] = selected
    
    # Rewrite
    for ds_name, texts in result.items():
        ds_dir = FILTERED_DIR / ds_name
        for f in ds_dir.glob("*.jsonl"):
            f.unlink()
        write_filtered_jsonl(ds_name, texts)
    
    return result


# ─── Tokenization to .bin (Stage 4) ───────────────────────────────
def tokenize_final(datasets_to_tokenize: list, target_tokens: int = 15_000_000_000):
    """Tokenize filtered JSONL to .bin shards for training."""
    from transformers import AutoTokenizer
    
    print(f"\n  [Tokenize] Target: {target_tokens/1e9:.1f}B tokens")
    
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M", trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    TOKENS_PER_SHARD = int(256_000_000)
    MAX_CHARS = 50000
    BATCH_SIZE = 2000
    
    buffer = []
    total_tokens = 0
    total_docs = 0
    shard_idx = 0
    
    for ds_name in datasets_to_tokenize:
        ds_dir = FILTERED_DIR / ds_name
        if not ds_dir.exists():
            continue
        
        jsonl_files = sorted(ds_dir.glob("*.jsonl"))
        if not jsonl_files:
            continue
        
        print(f"    [{ds_name}] {len(jsonl_files)} files")
        
        for jf in jsonl_files:
            with open(jf, 'r', encoding='utf-8') as f:
                batch_texts = []
                for line in f:
                    data = json.loads(line)
                    text = data.get('text', '')[:MAX_CHARS]
                    if len(text) < 20:
                        continue
                    batch_texts.append(text)
                    
                    if len(batch_texts) >= BATCH_SIZE:
                        encoded = tokenizer(batch_texts, add_special_tokens=False, truncation=False)
                        for ids in encoded["input_ids"]:
                            arr = np.array(ids, dtype=np.uint16)
                            buffer.append(arr)
                            total_tokens += len(arr)
                            total_docs += 1
                        
                        buffer = _flush_buffer(buffer, TOKENS_PER_SHARD)
                        new_shards = len(list(SHARDS_DIR.glob("shard_*.bin")))
                        if new_shards > shard_idx:
                            shard_idx = new_shards
                        batch_texts = []
                        
                        if total_tokens >= target_tokens:
                            break
            
            if batch_texts:
                encoded = tokenizer(batch_texts, add_special_tokens=False, truncation=False)
                for ids in encoded["input_ids"]:
                    arr = np.array(ids, dtype=np.uint16)
                    buffer.append(arr)
                    total_tokens += len(arr)
                    total_docs += 1
                buffer = _flush_buffer(buffer, TOKENS_PER_SHARD)
                new_shards = len(list(SHARDS_DIR.glob("shard_*.bin")))
                if new_shards > shard_idx:
                    shard_idx = new_shards
            
            print(f"      {jf.name}: {total_tokens:,} tok / {total_docs:,} docs", end='\r', flush=True)
            
            if total_tokens >= target_tokens:
                break
        
        if total_tokens >= target_tokens:
            break
    
    # Flush final
    if buffer:
        final = np.concatenate(buffer)
        out_path = SHARDS_DIR / f"shard_{shard_idx:06d}.bin"
        final.tofile(str(out_path))
        print(f"\n    Final shard: {out_path.name} ({len(final):,} tok)", flush=True)
        shard_idx += 1
        total_tokens += len(final)
    
    print(f"\n    ✅ Done: {shard_idx} shards, {total_tokens:,} tokens ({total_tokens/1e9:.3f}B)", flush=True)
    
    # Verify
    verify_shards()
    return total_tokens


def _flush_buffer(buffer, tok_per_shard):
    """Flush accumulated token arrays to shard files."""
    while buffer and sum(len(a) for a in buffer) >= tok_per_shard:
        all_tok = np.concatenate(buffer)
        if len(all_tok) >= tok_per_shard:
            shard_idx = len(list(SHARDS_DIR.glob("shard_*.bin")))
            out_path = SHARDS_DIR / f"shard_{shard_idx:06d}.bin"
            all_tok[:tok_per_shard].tofile(str(out_path))
            size_mb = out_path.stat().st_size / (1024*1024)
            print(f"\n      → {out_path.name}: {tok_per_shard:,} tok ({size_mb:.0f}MB)", flush=True)
            remainder = [all_tok[tok_per_shard:]] if len(all_tok) > tok_per_shard else []
            buffer[:] = remainder
        else:
            break
    return buffer


def verify_shards():
    """Verify shard integrity."""
    shards = sorted(SHARDS_DIR.glob("shard_*.bin"))
    total_bytes = sum(s.stat().st_size for s in shards)
    print(f"    📊 {len(shards)} shards, {total_bytes/1024**3:.2f} GB, ~{total_bytes//2:,} tokens")


# ─── Entry ─────────────────────────────────────────────────────────
def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Data Quality v3 Pipeline")
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS.keys()),
                        help="Datasets to process")
    parser.add_argument("--stages", type=int, nargs="+", default=[1, 2, 3, 4],
                        help="Stages to run: 1=heuristic+dedup, 2=perplexity, 3=mixture, 4=tokenize")
    parser.add_argument("--target-tokens", type=int, default=15_000_000_000,
                        help="Target total tokens")
    parser.add_argument("--checkpoint", default=str(CKPT_DIR / "pretrained_best.pt"),
                        help="Model checkpoint for perplexity scoring")
    parser.add_argument("--keep-ratio", type=float, default=0.60,
                        help="Top fraction to keep by perplexity")
    parser.add_argument("--workers", type=int, default=3,
                        help="Workers per dataset (default 3, use ~3 for parallel runs)")
    args = parser.parse_args()
    
    stages = set(args.stages)
    
    # Stage 1: Heuristic filter + dedup
    if 1 in stages:
        print("=" * 60)
        print("STAGE 1: Heuristic Filtering + Deduplication")
        print("=" * 60)
        
        dedup = Deduplicator()
        
        for ds_name in args.datasets:
            cfg = DATASETS.get(ds_name)
            if not cfg:
                print(f"  [SKIP] {ds_name}: unknown")
                continue
            
            # Check if already done
            ds_dir = FILTERED_DIR / ds_name
            if ds_dir.exists() and list(ds_dir.glob("*.jsonl")):
                print(f"  [SKIP] {ds_name}: already filtered (delete {ds_dir} to redo)")
                continue
            
            n_kept = process_dataset_stage1(ds_name, cfg, dedup, args.workers)
            if n_kept:
                print(f"  ✅ {ds_name}: {n_kept:,} docs kept")
            else:
                print(f"  ⚠ {ds_name}: 0 docs passed")
    
    # Stage 2: Perplexity scoring (GPU)
    if 2 in stages:
        print("\n" + "=" * 60)
        print("STAGE 2: Perplexity Scoring (GPU)")
        print("=" * 60)
        
        for ds_name in args.datasets:
            ds_dir = FILTERED_DIR / ds_name
            if not ds_dir.exists():
                continue
            n_kept = run_perplexity_scoring(ds_name, args.checkpoint, args.keep_ratio)
            if n_kept:
                print(f"  ✅ {ds_name}: {n_kept:,} docs after perplexity filter")
    
    # Stage 3: Mixture enforcement
    if 3 in stages:
        print("\n" + "=" * 60)
        print("STAGE 3: Mixture Enforcement")
        print("=" * 60)
        enforce_mixture(args.target_tokens)
    
    # Stage 4: Tokenization
    if 4 in stages:
        print("\n" + "=" * 60)
        print("STAGE 4: Tokenization → .bin Shards")
        print("=" * 60)
        tokenize_final(args.datasets, args.target_tokens)
    
    print("\n✅ Pipeline complete!")


if __name__ == "__main__":
    main()
