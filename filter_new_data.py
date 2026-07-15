#!/usr/bin/env python3
"""
Quality filter raw parquet downloads before tokenization.
Uses Gopher + FineWeb + C4 heuristics with RELAXED thresholds.
Processes files in parallel for speed.
"""
import os
import sys
import json
import string
import glob
import multiprocessing as mp
from pathlib import Path
from collections import Counter
from datetime import datetime

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# === CONFIG ===
STAGING_ROOT = Path("/home/kenpeter/work/data/_staging_multi")
OUTPUT_ROOT = Path("/home/kenpeter/work/data/_filtered_new")
N_WORKERS = 6

# Text column names per dataset
TEXT_COLS = {
    "stack-python": "content",
    "finemath-3plus": "text",
    "finemath": "text",
    "cosmopedia": "text",
    "open-web-math": "text",
    "fineweb-edu": "text",
}

# === RELAXED QUALITY FILTER ===
stop_words = {"the", "be", "to", "of", "and", "that", "have", "with",
              "it", "for", "not", "on", "with", "he", "as", "you", "do", "at"}

def quality_filter(text: str, dataset_name: str = "") -> tuple[bool, str]:
    """Return (pass, reason). Uses RELAXED thresholds. Code datasets skip punctuation/line checks."""
    if not text or not isinstance(text, str):
        return False, "empty"

    words = text.split()
    n_words = len(words)

    # CODE-SPECIFIC RELAXATION: skip checks that kill code
    is_code = "stack-python" in dataset_name or "code" in dataset_name.lower()
    
    if is_code:
        # CODE ONLY: very lenient thresholds
        if n_words < 20:
            return False, "gopher:too_few_words"
        if n_words > 100_000:
            return False, "gopher:too_many_words"
        # Skip avg_word_len, symbol_words, stop_words, bullets for code
    else:
        # Normal text checks
        if n_words < 50:
            return False, "gopher:too_few_words"
        if n_words > 100_000:
            return False, "gopher:too_many_words"

        # Gopher: avg word length
        avg_word_len = sum(len(w) for w in words) / max(n_words, 1)
        if avg_word_len < 2.5 or avg_word_len > 12:
            return False, "gopher:avg_word_len"

        # Gopher: symbol words (RELAXED to 30%)
        symbol_words = sum(1 for w in words if any(c in string.punctuation for c in w))
        if symbol_words / max(n_words, 1) > 0.30:
            return False, "gopher:too_many_symbols"

        # Gopher: stop words
        unique_stops = len(set(w.lower() for w in words) & stop_words)
        if unique_stops < 2:
            return False, "gopher:too_few_stop_words"

        # Lines
        lines = text.split('\n')
        n_lines = len(lines)
        if n_lines == 0:
            return False, "gopher:no_lines"

        # Gopher: bullet lines (RELAXED to 70%)
        bullet_lines = sum(1 for l in lines if l.strip().startswith(('-', '*', '•', '·')))
        if bullet_lines / n_lines > 0.70:
            return False, "gopher:too_many_bullets"

        # FineWeb: punctuation ending (RELAXED to 8%)
        punct_ending = sum(1 for l in lines if l.strip() and l.strip()[-1] in '.!?;')
        if punct_ending / n_lines < 0.08:
            return False, "fineweb:low_punct"

        # FineWeb: short lines (RELAXED to 80%)
        short_lines = sum(1 for l in lines if len(l) < 30)
        if short_lines / n_lines > 0.80:
            return False, "fineweb:too_many_short_lines"

    # FineWeb: char repetition (CRITICAL: exclude whitespace!) — applies to ALL
    if len(text) > 0:
        char_counts = Counter(text)
        filtered = {c: n for c, n in char_counts.items() if c not in ' \n\t\r'}
        if filtered:
            max_dup = max(filtered.values()) / len(text)
            if max_dup > 0.12:
                return False, "fineweb:char_repetition"

    # C4: bad words — applies to ALL
    bad_words = {'http', 'https', 'www', 'html'}
    if any(bw in text.lower() for bw in bad_words):
        return False, "c4:bad_words"

    return True, "pass"


def process_one_file(args):
    """Worker: read one parquet, filter, append to dataset JSONL."""
    parquet_path, dataset_name, text_col = args

    import pyarrow.parquet as pq

    out_dir = OUTPUT_ROOT / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{parquet_path.stem}.jsonl"

    # Skip if already done
    if out_path.exists() and out_path.stat().st_size > 100:
        return {"file": str(parquet_path), "skipped": True}

    kept = 0
    total = 0
    reasons = Counter()

    try:
        with open(out_path, "w", encoding="utf-8") as fout:
            pf = pq.ParquetFile(str(parquet_path))
            for batch in pf.iter_batches(batch_size=1000, columns=[text_col]):
                for row in batch.to_pylist():
                    total += 1
                    text = row.get(text_col, "")
                    if not isinstance(text, str):
                        text = str(text) if text is not None else ""

                    ok, reason = quality_filter(text, dataset_name)
                    if ok:
                        kept += 1
                        fout.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
                    else:
                        reasons[reason] += 1
    except Exception as e:
        return {"file": str(parquet_path), "error": str(e), "kept": kept, "total": total, "reasons": dict(reasons)}

    return {"file": str(parquet_path), "kept": kept, "total": total, "reasons": dict(reasons), "skipped": False}


def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    # Build task list
    tasks = []
    for dataset_name, text_col in TEXT_COLS.items():
        src_dir = STAGING_ROOT / dataset_name
        if not src_dir.exists():
            continue
        files = sorted(src_dir.glob("*.parquet"))
        print(f"[{dataset_name}] {len(files)} parquet files")
        for f in files:
            tasks.append((f, dataset_name, text_col))

    print(f"\nTotal tasks: {len(tasks)}")
    print(f"Workers: {N_WORKERS}")
    print(f"Start: {datetime.now().isoformat()}\n")
    sys.stdout.flush()

    # Process in parallel
    all_results = []
    with mp.Pool(N_WORKERS) as pool:
        for i, res in enumerate(pool.imap_unordered(process_one_file, tasks), 1):
            all_results.append(res)
            if i % 10 == 0 or i == len(tasks):
                print(f"  {i}/{len(tasks)} done")

    # Summary
    print(f"\n=== FILTER SUMMARY ===")
    dataset_stats = {}
    for r in all_results:
        ds = Path(r["file"]).parent.name
        if ds not in dataset_stats:
            dataset_stats[ds] = {"kept": 0, "total": 0, "files": 0, "errors": 0, "skipped": 0}
        dataset_stats[ds]["files"] += 1
        if r.get("skipped"):
            dataset_stats[ds]["skipped"] += 1
        if r.get("error"):
            dataset_stats[ds]["errors"] += 1
            print(f"  ERROR {ds}: {r['error']}")
            continue
        dataset_stats[ds]["kept"] += r.get("kept", 0)
        dataset_stats[ds]["total"] += r.get("total", 0)

    for ds, s in sorted(dataset_stats.items()):
        rate = s["kept"] / max(s["total"], 1) * 100
        print(f"  {ds}: {s['kept']:,}/{s['total']:,} kept ({rate:.1f}%) | {s['files']} files | {s['errors']} errors | {s['skipped']} skipped")

    print(f"\nEnd: {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()
