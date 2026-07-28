#!/usr/bin/env python3
"""Preprocess all LeetCode datasets into a unified parquet format for filter_all.py.
Extracts code + problem text, writes to _raw_original/leetcode_code/ as parquet with 'content' column."""
import os, sys, json, gc
from pathlib import Path
import pyarrow.parquet as pq
import pyarrow as pa

RAW_DIR = Path("/home/kenpeter/work/data/_raw_original/leetcode_code")
RAW_DIR.mkdir(parents=True, exist_ok=True)

def log(msg):
    print(f"[{os.uname()[1].split('.')[0]}] {msg}", flush=True)

# ─── Source datasets ─────────────────────────────────────────────
SOURCES = {
    # Largest one: 313 MB, has level field + code in multiple langs
    "yt_cot": {
        "path": "/mnt/file_drive/data/LeetCode_YT_CC_CoT_Summary/data",
        "glob": "*.parquet",
        "type": "parquet",
        "extract": lambda row: _extract_yt_cot(row),
    },
    # Has difficulty + completion
    "high_quality": {
        "path": "/mnt/file_drive/data/high_quality_leetcode/train.jsonl",
        "type": "jsonl",
        "extract": lambda row: _extract_hq_leetcode(row),
    },
    # Has conversation format with code
    "limyeri": {
        "path": "/mnt/file_drive/data/LimYeri_LeetCode/train.parquet",
        "type": "parquet",
        "extract": lambda row: _extract_limyeri(row),
    },
    # Has python/java code solutions
    "greengerong": {
        "path": "/mnt/file_drive/data/greengerong_LeetCode/leetcode-train.jsonl",
        "type": "jsonl",
        "extract": lambda row: _extract_greengerong(row),
    },
    # Has java+python parquet
    "denct": {
        "path": "/mnt/file_drive/data/DenCT_LeetCode/leetcode-java-python.parquet",
        "type": "parquet",
        "extract": lambda row: _extract_denct(row),
    },
}

def _extract_yt_cot(row):
    """Extract code from LeetCode_YT_CC_CoT_Summary."""
    code = ""
    for lang in ["python", "java", "c++", "javascript"]:
        if row.get(lang):
            code += f"\n// {lang}\n" + row[lang] + "\n"
    question = row.get("question_content", "") or ""
    title = row.get("title", "") or ""
    level = row.get("level", "") or ""
    text = f"Problem: {title} (Difficulty: {level})\n\n{question}\n\nSolutions:\n{code}"
    return text.strip() if len(text.strip()) > 20 else None

def _extract_hq_leetcode(row):
    """Extract from high_quality_leetcode."""
    desc = row.get("problem_description", "") or ""
    completion = row.get("completion", "") or ""
    difficulty = row.get("difficulty", "") or ""
    tags = row.get("tags", []) or []
    tag_str = ", ".join(tags) if isinstance(tags, list) else str(tags)
    text = f"Problem: {row.get('task_id','')} (Difficulty: {difficulty}, Tags: {tag_str})\n\n{desc}\n\nSolution:\n{completion}"
    return text.strip() if len(text.strip()) > 20 else None

def _extract_limyeri(row):
    """Extract from LimYeri (conversation format)."""
    title = row.get("title", "") or ""
    convs = row.get("conversations", []) or []
    text_parts = [f"Problem: {title}"]
    for c in convs:
        if isinstance(c, dict) and c.get("value"):
            text_parts.append(c["value"])
    text = "\n\n".join(text_parts)
    return text.strip() if len(text.strip()) > 20 else None

def _extract_greengerong(row):
    """Extract from greengerong."""
    text = json.dumps(row)
    return text if len(text) > 20 else None

def _extract_denct(row):
    """Extract from DenCT."""
    java = row.get("java", "") or ""
    python = row.get("python", "") or ""
    text = ""
    if python: text += f"Python:\n{python}\n\n"
    if java: text += f"Java:\n{java}"
    return text.strip() if len(text.strip()) > 20 else None

# ─── Main ────────────────────────────────────────────────────────
all_texts = []
total_rows = 0

for name, src in SOURCES.items():
    log(f"Processing {name}...")
    try:
        if src["type"] == "parquet":
            if src["path"].endswith(".parquet"):
                files = [Path(src["path"])]
            else:
                files = sorted(Path(src["path"]).glob(src.get("glob", "*.parquet")))
            
            for f in files:
                if not f.exists():
                    log(f"  {f} not found, skipping")
                    continue
                pf = pq.ParquetFile(str(f))
                nrows = pf.metadata.num_rows
                log(f"  {f.name}: {nrows} rows")
                for batch in pf.iter_batches(batch_size=500):
                    for row in batch.to_pylist():
                        total_rows += 1
                        text = src["extract"](row)
                        if text:
                            all_texts.append(text)
                    gc.collect()
        
        elif src["type"] == "jsonl":
            p = Path(src["path"])
            if not p.exists():
                log(f"  {p} not found, skipping")
                continue
            with open(p) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    total_rows += 1
                    row = json.loads(line)
                    text = src["extract"](row)
                    if text:
                        all_texts.append(text)
    
    except Exception as e:
        log(f"  ERROR: {e}")
        continue

log(f"\nTotal rows read: {total_rows}")
log(f"Total texts extracted: {len(all_texts)}")

# Write as parquet chunks (filter_all.py reads these)
if all_texts:
    chunk_size = 20000
    for i in range(0, len(all_texts), chunk_size):
        chunk = all_texts[i:i+chunk_size]
        table = pa.table({"content": pa.array(chunk, type=pa.large_string())})
        out_file = RAW_DIR / f"train-{i:05d}.parquet"
        pq.write_table(table, str(out_file))
        log(f"Wrote {out_file.name}: {len(chunk)} texts, {sum(len(t) for t in chunk)//1024} KB")
    
    log(f"\nDone! {len(all_texts)} texts written to {RAW_DIR}")
else:
    log("No texts extracted!")
