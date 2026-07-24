#!/usr/bin/env python3
"""Promote validated genes to immutable capsules.

Usage:
    python3 evolver_capsule.py promote sdcard-test-gate
    python3 evolver_capsule.py list
    python3 evolver_capsule.py verify sdcard-test-gate
"""
import sys, json, os, argparse, datetime, hashlib

GEP_DIR = os.path.expanduser("~/.hermes/profiles/agent-1/.evolver/gep")
GENES_PATH = os.path.join(GEP_DIR, "genes.json")
CAPSULES_PATH = os.path.join(GEP_DIR, "capsules.json")
EVENTS_PATH = os.path.join(GEP_DIR, "events.jsonl")


def load_json(path):
    with open(path) as f:
        return json.load(f)


def save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def log_event(gene_id, evt_type, context, result):
    t = datetime.datetime.utcnow().isoformat() + "Z"
    line = json.dumps({"t": t, "type": evt_type, "gene_id": gene_id, "context": context, "result": result})
    with open(EVENTS_PATH, "a") as f:
        f.write(line + "\n")


def checksum(gene):
    """Simple checksum of gene content for immutability tracking."""
    s = json.dumps(gene, sort_keys=True)
    return hashlib.sha256(s.encode()).hexdigest()[:16]


def cmd_promote(args):
    genes_data = load_json(GENES_PATH)
    caps_data = load_json(CAPSULES_PATH)
    gene = next((g for g in genes_data["genes"] if g["id"] == args.gene_id), None)
    if not gene:
        print(f"ERROR: gene '{args.gene_id}' not found")
        sys.exit(1)
    if gene.get("activations", 0) < 5 and not args.force:
        print(f"ERROR: gene '{args.gene_id}' has only {gene.get('activations',0)} activations. Need >= 5 or --force.")
        sys.exit(1)

    cs = checksum(gene)
    capsule = {
        "id": gene["id"],
        "name": gene["name"],
        "category": gene.get("category", "workflow"),
        "strategy": gene.get("strategy", "balanced"),
        "action": gene["action"],
        "sacred": gene.get("sacred", False),
        "checksum": cs,
        "frozen_at": datetime.datetime.utcnow().isoformat() + "Z",
        "activations_at_freeze": gene.get("activations", 0),
        "source": gene.get("source", ""),
    }
    caps_data["capsules"].append(capsule)
    caps_data["frozen"].append(gene["id"])
    save_json(CAPSULES_PATH, caps_data)
    log_event(args.gene_id, "promote", f"promoted to capsule, checksum={cs}", "ok")
    print(f"Promoted '{args.gene_id}' to capsule (checksum: {cs})")


def cmd_list():
    caps_data = load_json(CAPSULES_PATH)
    if not caps_data["capsules"]:
        print("No capsules yet.")
        return
    print(f"{'ID':<30} {'Name':<40} {'Checksum':<18} {'Frozen At':<25}")
    print("-" * 115)
    for c in caps_data["capsules"]:
        print(f"{c['id']:<30} {c['name']:<40} {c['checksum']:<18} {c['frozen_at']:<25}")


def cmd_verify(args):
    caps_data = load_json(CAPSULES_PATH)
    genes_data = load_json(GENES_PATH)
    cap = next((c for c in caps_data["capsules"] if c["id"] == args.gene_id), None)
    gene = next((g for g in genes_data["genes"] if g["id"] == args.gene_id), None)
    if not cap:
        print(f"ERROR: capsule '{args.gene_id}' not found")
        sys.exit(1)
    if not gene:
        print(f"WARNING: gene '{args.gene_id}' no longer exists in genes.json")
        sys.exit(1)
    current_cs = checksum(gene)
    if current_cs == cap["checksum"]:
        print(f"VERIFIED: '{args.gene_id}' unchanged since freeze ({current_cs})")
    else:
        print(f"MISMATCH: '{args.gene_id}' has changed!")
        print(f"  Frozen checksum: {cap['checksum']}")
        print(f"  Current checksum: {current_cs}")
        print(f"  Gene was modified after capsule creation. Consider re-promoting.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd")

    p_promote = sub.add_parser("promote", help="Promote a gene to immutable capsule")
    p_promote.add_argument("gene_id")
    p_promote.add_argument("--force", action="store_true", help="Promote even with <5 activations")

    sub.add_parser("list", help="List all capsules")

    p_verify = sub.add_parser("verify", help="Verify gene hasn't changed since freeze")
    p_verify.add_argument("gene_id")

    args = parser.parse_args()
    if args.cmd == "promote":
        cmd_promote(args)
    elif args.cmd == "list":
        cmd_list()
    elif args.cmd == "verify":
        cmd_verify(args)
    else:
        parser.print_help()
