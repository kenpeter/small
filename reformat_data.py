#!/usr/bin/env python3
"""
Kimi K2-style multi-format data reformatting.
Loads raw docs, rewrites each into N format styles via vLLM, saves as .jsonl.

Format styles:
  1. textbook   — structured educational explanation
  2. qa         — Q&A pairs covering key concepts
  3. lecture    — spoken lecture / presentation style
  4. tutorial   — step-by-step how-to guide
  5. knowledge_card — compact fact/knowledge cards

Usage:
  python3 reformat_data.py --input <parquet_or_jsonl> --output <dir> --formats 1,2,3,4,5 --max-docs 100
"""

import sys, os, json, time, gc, re, copy, random
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse

# ─── Format Templates ───────────────────────────────────────────────────────

FORMAT_TEMPLATES = {
    "textbook": {
        "name": "Textbook Explanation",
        "system": "Write a textbook section. "
                  "Use formal academic language with definitions, explanations, and examples. "
                  "START IMMEDIATELY with the textbook content. Do NOT include any planning, "
                  "thinking, reasoning, or meta-commentary. Do NOT start with 'Okay', "
                  "'Let me', 'First', 'I need', 'The user', 'Alright', 'Well', or 'So'. "
                  "Just write the textbook section directly without any introduction.",
        "user": "{text}",
    },
    "qa": {
        "name": "Q&A Pairs",
        "system": "Create 3-5 Q&A pairs about the text below. "
                  "Format:\nQ: <question>\nA: <answer>\n\n"
                  "START IMMEDIATELY with Q:. "
                  "Do NOT include any planning, thinking, reasoning, or meta-commentary. "
                  "Do NOT start with 'Okay', 'Let me', 'First', 'I need', 'The user', "
                  "'Alright', 'Well', or 'So'.",
        "user": "{text}",
    },
    "lecture": {
        "name": "Lecture / Presentation",
        "system": "You are an engaging lecturer presenting material to students. "
                  "Rewrite the given text as a spoken lecture or presentation script. "
                  "Use natural spoken language, rhetorical questions, and clear signposting "
                  "(e.g., 'Let's turn now to...', 'The key insight here is...'). "
                  "Output ONLY the lecture content, no commentary.",
        "user": "Turn this into a lecture script:\n\n{text}",
    },
    "tutorial": {
        "name": "Tutorial / How-to Guide",
        "system": "You are a technical writer creating a step-by-step tutorial. "
                  "Rewrite the given content as a practical how-to guide. "
                  "Break down the process into numbered steps, include prerequisites, "
                  "tips, and common pitfalls where relevant. "
                  "Output ONLY the tutorial content, no commentary.",
        "user": "Create a step-by-step tutorial from this:\n\n{text}",
    },
    "knowledge_card": {
        "name": "Knowledge Cards",
        "system": "You are creating concise knowledge cards for spaced-repetition learning. "
                  "Extract the key facts from the given text and present them as "
                  "compact knowledge cards. Each card has a single concept and its explanation. "
                  "Format as:\n\n📌 Card 1: <concept>\n   {text}\n\n📌 Card 2: <concept>\n   {text}\n\n"
                  "Output ONLY the knowledge cards, no introduction.",
        "user": "Extract knowledge cards from:\n\n{text}",
    },
}

@dataclass
class ReformattedDoc:
    source_idx: int
    source_text: str
    source_domain: str
    format_type: str
    format_name: str
    rewritten_text: str
    model: str = ""
    tokens_in: int = 0
    tokens_out: int = 0
    timestamp: float = 0.0

    def to_dict(self):
        return {
            "source_idx": self.source_idx,
            "source_domain": self.source_domain,
            "format_type": self.format_type,
            "format_name": self.format_name,
            "model": self.model,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "timestamp": self.timestamp,
            "text": self.rewritten_text,
        }


class ReformatPipeline:
    """Multi-format document reformatting pipeline using vLLM."""

    def __init__(
        self,
        api_base: str = "http://localhost:8000/v1",
        model: str = "Qwen/Qwen2.5-7B-Instruct-AWQ",
        output_dir: str = "/home/kenpeter/work/data/_reformatted",
        formats: list = None,
        max_docs: int = 100,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        num_workers: int = 4,
        resume: bool = True,
    ):
        self.api_base = api_base.rstrip("/")
        self.model = model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.formats = formats or list(FORMAT_TEMPLATES.keys())
        self.max_docs = max_docs
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.num_workers = num_workers
        self.resume = resume
        self.completed_file = self.output_dir / "_completed_indices.json"
        self.completed_pairs: set = set()  # (doc_idx, format) tuples
        self.total_api_calls = 0
        self.total_api_time = 0.0

        if resume and self.completed_file.exists():
            try:
                data = json.loads(self.completed_file.read_text())
                self.completed_pairs = set(tuple(p) for p in data.get("completed_pairs", []))
                self.total_api_calls = data.get("total_api_calls", 0)
                self.total_api_time = data.get("total_api_time", 0.0)
                print(f"[resume] {len(self.completed_pairs)} (doc,format) pairs already processed")
            except Exception:
                self.completed_pairs = set()

        # Test connection
        self._check_vllm_connection()

    def _check_vllm_connection(self):
        import urllib.request
        try:
            req = urllib.request.Request(f"{self.api_base}/models")
            req.add_header("Authorization", "Bearer token-abc123")
            resp = urllib.request.urlopen(req, timeout=10)
            data = json.loads(resp.read())
            models = [m["id"] for m in data.get("data", [])]
            print(f"[vllm] Connected. Available models: {models}")
        except Exception as e:
            print(f"[vllm] Connection warning: {e}")
            print(f"[vllm] Make sure vLLM server is running at {self.api_base}")

    def _call_vllm(self, system_prompt: str, user_prompt: str) -> tuple:
        """Call vLLM API. Returns (text, tokens_in, tokens_out)."""
        import urllib.request
        t0 = time.time()

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        payload = json.dumps({
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": 0.9,
        }).encode()

        req = urllib.request.Request(
            f"{self.api_base}/chat/completions",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer token-abc123",
            },
        )

        try:
            resp = urllib.request.urlopen(req, timeout=120)
            result = json.loads(resp.read())
            choice = result["choices"][0]
            text = choice["message"]["content"].strip()
            # Strip Qwen3 think tags
            text = re.sub(r'</?think>', '', text, flags=re.IGNORECASE)
            # Aggressive preamble removal: strip leading meta-commentary sentences
            # Match sentences starting with preamble words at the beginning of text
            text = re.sub(
                r'^((?:Okay|Ok|Alright|Well|So|First|Now)[,!\s].*?'
                r'|Let me.*?'
                r'I (?:need|will|shall|can|am|start).*?'
                r'The (?:user|text|provided|given|original|article).*?'
                r'As (?:an|a) (?:AI|assistant|expert|educator).*?'
                r'Looking at|Reading|Understanding|Analyzing|Examining|Starting to)\n+|'
                r'^[^Q\n]*?(?=Q[.:]\s)',
                '', text, flags=re.DOTALL
            )
            # Multi-round strip: keep removing preamble sentences
            for _ in range(5):
                new_text = re.sub(
                    r'^\s*(?:(?:Okay|Ok|Alright|Well|So|First|Now)[,!\s].*?(?:[.!?]\s|$)'
                    r'|Let me.*?(?:[.!?]\s|$)'
                    r'I (?:need|will|shall|can|am|start).*?(?:[.!?]\s|$)'
                    r'The (?:user|text|provided|given|original).*?(?:[.!?]\s|$)'
                    r')\s*',
                    '', text
                )
                if new_text == text:
                    break
                text = new_text
            text = text.strip()
            usage = result.get("usage", {})
            tokens_in = usage.get("prompt_tokens", 0)
            tokens_out = usage.get("completion_tokens", 0)
            elapsed = time.time() - t0
            self.total_api_calls += 1
            self.total_api_time += elapsed
            return text, tokens_in, tokens_out
        except Exception as e:
            print(f"  ⚠ API error: {e}")
            return "", 0, 0

    def process_doc(self, doc: dict, doc_idx: int, domain: str) -> list[ReformattedDoc]:
        """Rewrite a single document into all configured formats."""
        text = doc.get("text", "")
        if not isinstance(text, str) or len(text.strip()) < 50:
            return []

        # Truncate very long docs to avoid OOM
        if len(text) > 8000:
            text = text[:8000]

        results = []
        for fmt in self.formats:
            template = FORMAT_TEMPLATES[fmt]
            user_prompt = template["user"].format(text=text)

            rewritten, tok_in, tok_out = self._call_vllm(
                template["system"], user_prompt
            )

            if rewritten:
                results.append(ReformattedDoc(
                    source_idx=doc_idx,
                    source_text=text[:200],  # preview
                    source_domain=domain,
                    format_type=fmt,
                    format_name=template["name"],
                    rewritten_text=rewritten,
                    model=self.model,
                    tokens_in=tok_in,
                    tokens_out=tok_out,
                    timestamp=time.time(),
                ))
            else:
                print(f"  ⚠ [{doc_idx}/{fmt}] empty response, skipping")

        return results

    def process_batch(self, docs: list[dict], domain: str = "web"):
        """Process a batch of documents through all format styles using thread pool for parallel API calls."""
        print(f"\n{'='*60}")
        print(f"Processing {len(docs)} docs from '{domain}' domain")
        print(f"Formats: {', '.join(self.formats)}")
        print(f"Workers: {self.num_workers}")
        print(f"Output: {self.output_dir}")
        print(f"{'='*60}\n")

        all_results = []
        processed = 0
        skipped_resume = 0

        # Build task list: (doc, doc_idx, format) tuples
        tasks = []
        for idx, doc in enumerate(docs):
            global_idx = idx
            for fmt in self.formats:
                if self.resume and (global_idx, fmt) in self.completed_pairs:
                    skipped_resume += 1
                    continue
                tasks.append((doc, global_idx, domain, fmt))

        # Process formats for all docs with thread pool
        from concurrent.futures import ThreadPoolExecutor, as_completed
        t_start = time.time()

        def process_task(task):
            doc, doc_idx, dom, fmt = task
            text = doc.get("text", "")
            if not isinstance(text, str) or len(text.strip()) < 50:
                return None, doc_idx, fmt
            if len(text) > 8000:
                text = text[:8000]

            template = FORMAT_TEMPLATES[fmt]
            user_prompt = template["user"].format(text=text)
            rewritten, tok_in, tok_out = self._call_vllm(template["system"], user_prompt)
            if not rewritten:
                return None, doc_idx, fmt

            return ReformattedDoc(
                source_idx=doc_idx,
                source_text=text[:200],
                source_domain=dom,
                format_type=fmt,
                format_name=template["name"],
                rewritten_text=rewritten,
                model=self.model,
                tokens_in=tok_in,
                tokens_out=tok_out,
                timestamp=time.time(),
            ), doc_idx, fmt

        with ThreadPoolExecutor(max_workers=self.num_workers) as pool:
            futures = {pool.submit(process_task, t): t for t in tasks}
            done_count = 0
            for future in as_completed(futures):
                done_count += 1
                result, doc_idx, fmt = future.result()
                if result:
                    all_results.append(result)

                # Track completed pairs
                self.completed_pairs.add((doc_idx, fmt))

                # Incremental save
                if len(all_results) >= 50:
                    self._flush(all_results)

                # Save completed indices every 5 docs
                if done_count % 25 == 0:
                    self._save_completed()

                if done_count % 10 == 0 or done_count == len(tasks):
                    elapsed = time.time() - t_start
                    rate = done_count / max(elapsed, 0.1)
                    print(f"  [{done_count}/{len(tasks)} tasks] kept={len(all_results)} reformatted | "
                          f"{rate:.1f} tasks/s | {elapsed:.0f}s elapsed")

        # Final flush
        self._flush(all_results)
        self._save_completed()

        self._print_summary(done_count, processed, skipped_resume, all_results)
        return all_results

    def _print_summary(self, done_count, processed, skipped_resume, all_results):
        elapsed = time.time()
        print(f"\n{'='*60}")
        print(f"Done: {done_count} tasks, {len(all_results)} reformatted outputs")
        print(f"API calls: {self.total_api_calls} in {self.total_api_time:.0f}s")
        print(f"Skipped (resume): {skipped_resume}")
        print(f"{'='*60}")

        from collections import Counter
        fmt_counts = Counter(r.format_type for r in all_results)
        print(f"\nPer format:")
        for fmt in self.formats:
            c = fmt_counts.get(fmt, 0)
            out_path = self.output_dir / f"{fmt}.jsonl"
            sz = out_path.stat().st_size if out_path.exists() else 0
            print(f"  {fmt}: {c} docs, {sz/1024:.0f} KB")

        avg_tokens_out = {}
        for r in all_results:
            avg_tokens_out[r.format_type] = avg_tokens_out.get(r.format_type, 0) + r.tokens_out
        for fmt in self.formats:
            c = fmt_counts.get(fmt, 0)
            if c > 0:
                avg = avg_tokens_out[fmt] / c
                print(f"  {fmt}: avg_out_tokens={avg:.0f}")

    def _flush(self, results: list):
        """Write accumulated results to .jsonl file."""
        if not results:
            return

        # Group by format
        by_format = {}
        for r in results:
            by_format.setdefault(r.format_type, []).append(r)

        for fmt, items in by_format.items():
            out_path = self.output_dir / f"{fmt}.jsonl"
            with open(out_path, "a") as f:
                for item in items:
                    f.write(json.dumps(item.to_dict(), ensure_ascii=False) + "\n")
            print(f"  📝 {fmt}: +{len(items)} -> {out_path.name} (total: {out_path.stat().st_size/1024:.0f} KB)")

        results.clear()

    def _save_completed(self):
        """Save completed (doc,format) pairs for resume."""
        data = {
            "completed_pairs": sorted(self.completed_pairs),
            "total_api_calls": self.total_api_calls,
            "total_api_time": self.total_api_time,
        }
        self.completed_file.write_text(json.dumps(data, indent=2))

    def get_output_stats(self):
        """Show current output stats."""
        total = 0
        for fmt in self.formats:
            path = self.output_dir / f"{fmt}.jsonl"
            if path.exists():
                n = sum(1 for _ in open(path))
                sz = path.stat().st_size
                print(f"  {fmt}: {n} docs, {sz/1024:.0f} KB")
                total += n
        print(f"  Total: {total} reformatted docs")
        return total


def load_input(source: str, max_docs: int = None) -> list[dict]:
    """Load input documents from parquet or jsonl."""
    path = Path(source)
    docs = []

    if path.suffix == ".parquet":
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(str(path))
        n_rows = pf.metadata.num_rows
        if max_docs:
            n_rows = min(n_rows, max_docs)
        print(f"Loading {n_rows} rows from {path.name}...")
        for batch in pf.iter_batches(batch_size=min(1000, n_rows), columns=["text"]):
            for row in batch.to_pylist():
                if len(docs) >= n_rows:
                    break
                if isinstance(row.get("text"), str) and len(row["text"].strip()) > 50:
                    docs.append(row)
            if len(docs) >= n_rows:
                break

    elif path.suffix == ".jsonl":
        with open(path) as f:
            for line in f:
                if max_docs and len(docs) >= max_docs:
                    break
                row = json.loads(line)
                if isinstance(row.get("text"), str) and len(row["text"].strip()) > 50:
                    docs.append(row)

    else:
        print(f"Unsupported input format: {path.suffix}")
        print("Supported: .parquet, .jsonl")
        sys.exit(1)

    print(f"Loaded {len(docs)} documents (>=50 chars each)")
    return docs


def main():
    parser = argparse.ArgumentParser(description="Kimi K2-style multi-format data reformatting")
    parser.add_argument("--input", type=str, default="",
                        help="Input parquet or jsonl file")
    parser.add_argument("--domain", type=str, default="web",
                        help="Domain label (web, math, code, synth)")
    parser.add_argument("--output", type=str,
                        default="/home/kenpeter/work/data/_reformatted",
                        help="Output directory for .jsonl files")
    parser.add_argument("--formats", type=str, default="textbook,qa,lecture,tutorial,knowledge_card",
                        help="Comma-separated format types")
    parser.add_argument("--max-docs", type=int, default=100,
                        help="Maximum source documents to process")
    parser.add_argument("--max-tokens", type=int, default=2048,
                        help="Max output tokens per reformatting call")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Generation temperature")
    parser.add_argument("--workers", type=int, default=4,
                        help="Parallel API workers")
    parser.add_argument("--api-base", type=str, default="http://localhost:8000/v1",
                        help="vLLM server URL")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct-AWQ",
                        help="Model name")
    parser.add_argument("--no-resume", action="store_true",
                        help="Ignore previous progress")
    parser.add_argument("--stats", action="store_true",
                        help="Show output stats only")

    args = parser.parse_args()
    formats = [f.strip() for f in args.formats.split(",") if f.strip()]

    pipeline = ReformatPipeline(
        api_base=args.api_base,
        model=args.model,
        output_dir=args.output,
        formats=formats,
        max_docs=args.max_docs,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        num_workers=args.workers,
        resume=not args.no_resume,
    )

    if args.stats:
        pipeline.get_output_stats()
        return

    if not args.input:
        # Use a small sample from our data
        print("No input specified. Checking for available data samples...")
        base = Path("/home/kenpeter/work/data")
        candidates = list(base.glob("_raw_original/*/*.parquet"))
        if candidates:
            example = candidates[0]
            print(f"Using: {example} ({example.stat().st_size/1024**3:.1f} GB)")
            args.input = str(example)
        else:
            print("No data found. Use --input to specify a parquet or jsonl file.")
            sys.exit(1)

    docs = load_input(args.input, args.max_docs)
    if not docs:
        print("No valid documents loaded. Exiting.")
        return

    pipeline.process_batch(docs, domain=args.domain)
    pipeline.get_output_stats()


if __name__ == "__main__":
    main()
