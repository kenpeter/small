#!/usr/bin/env python3
"""Scan recent session context and suggest gene activations.

Usage:
    python3 evolver_scan.py --last-minutes 30
    python3 evolver_scan.py --file /tmp/pretrain.log --pattern "ERROR|CRITICAL|NaN"
"""
import sys, os, json, argparse, re, subprocess, datetime

GEP_DIR = os.path.expanduser("~/.hermes/profiles/agent-1/.evolver/gep")
GENES_PATH = os.path.join(GEP_DIR, "genes.json")
EVENTS_PATH = os.path.join(GEP_DIR, "events.jsonl")


def load_genes():
    with open(GENES_PATH) as f:
        return json.load(f)


def scan_signals_from_logs(log_file, patterns):
    """Scan a log file for error patterns."""
    signals = []
    if not os.path.exists(log_file):
        return signals
    with open(log_file) as f:
        for line in f:
            for pat in patterns:
                if re.search(pat, line, re.I):
                    signals.append({"source": log_file, "pattern": pat, "line": line.strip()[:200]})
    return signals


def scan_signals_from_memory():
    """Read agent memory for recent corrections."""
    signals = []
    # Check recent events.jsonl for rejections
    if os.path.exists(EVENTS_PATH):
        with open(EVENTS_PATH) as f:
            lines = f.readlines()[-20:]
        for line in lines:
            try:
                evt = json.loads(line)
                if evt.get("result") in ["rejected", "failed", "violated"]:
                    signals.append({"source": "events.jsonl", "pattern": "rejection", "line": f"{evt['gene_id']}: {evt.get('context','')}"})
            except json.JSONDecodeError:
                pass
    return signals


def match_genes(genes_data, signals):
    """Match signals against gene triggers."""
    suggestions = []
    for gene in genes_data["genes"]:
        for sig in signals:
            trigger = gene.get("trigger", "").lower()
            line = sig.get("line", "").lower()
            # Simple keyword matching
            keywords = set(re.findall(r'\w+', trigger))
            line_words = set(re.findall(r'\w+', line))
            overlap = keywords & line_words
            if len(overlap) >= 2 or any(kw in line for kw in trigger.split() if len(kw) > 4):
                suggestions.append({"gene": gene, "signal": sig, "confidence": len(overlap)})
                break
    return suggestions


def cmd_scan(args):
    genes_data = load_genes()
    signals = []

    # Scan log files
    if args.file:
        patterns = args.pattern.split(",") if args.pattern else ["ERROR", "CRITICAL", "FAIL", "NaN", "Inf", "abort"]
        signals.extend(scan_signals_from_logs(args.file, patterns))

    # Scan memory/events
    signals.extend(scan_signals_from_memory())

    if not signals:
        print("No signals detected in recent context.")
        return

    print(f"Detected {len(signals)} signal(s):\n")
    for s in signals[:5]:
        print(f"  [{s['source']}] {s['pattern']}: {s['line'][:120]}")
    print()

    suggestions = match_genes(genes_data, signals)
    if not suggestions:
        print("No matching genes. Consider creating a new gene for this pattern.")
        return

    print(f"Suggested gene activation(s):\n")
    for sug in suggestions:
        g = sug["gene"]
        print(f"  🧬 {g['id']} (confidence: {sug['confidence']})")
        print(f"     Name: {g['name']}")
        print(f"     Action: {g['action'][:100]}{'...' if len(g['action']) > 100 else ''}")
        print(f"     Sacred: {'YES' if g.get('sacred') else 'no'}  |  Activations: {g.get('activations',0)}")
        if g.get("sacred"):
            print(f"     ⚠️  This is a SACRED rule — must activate")
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", default="/tmp/pretrain.log", help="Log file to scan")
    parser.add_argument("--pattern", default="ERROR,CRITICAL,FAIL,NaN,Inf,abort", help="Comma-separated patterns")
    parser.add_argument("--last-minutes", type=int, default=30, help="Time window (not yet implemented)")
    args = parser.parse_args()
    cmd_scan(args)
