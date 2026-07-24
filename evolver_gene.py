#!/usr/bin/env python3
"""Create and list GEP genes for agent self-evolution.

Usage:
    python3 evolver_gene.py list
    python3 evolver_gene.py create --id "sdcard-foo" --name "Foo" --trigger "when..." --action "do..."
    python3 evolver_gene.py activate sdcard-test-gate --context "user asked to patch optimizer"
"""
import sys, json, os, argparse, datetime

GEP_DIR = os.path.expanduser("~/.hermes/profiles/agent-1/.evolver/gep")
GENES_PATH = os.path.join(GEP_DIR, "genes.json")
EVENTS_PATH = os.path.join(GEP_DIR, "events.jsonl")


def load_genes():
    with open(GENES_PATH) as f:
        return json.load(f)


def save_genes(data):
    with open(GENES_PATH, "w") as f:
        json.dump(data, f, indent=2)


def log_event(gene_id, evt_type, context, result):
    t = datetime.datetime.utcnow().isoformat() + "Z"
    line = json.dumps({"t": t, "type": evt_type, "gene_id": gene_id, "context": context, "result": result})
    with open(EVENTS_PATH, "a") as f:
        f.write(line + "\n")


def cmd_list():
    data = load_genes()
    print(f"{'ID':<30} {'Name':<40} {'Sacred':<7} {'Strategy':<10} {'Activations':<12}")
    print("-" * 105)
    for g in data["genes"]:
        print(f"{g['id']:<30} {g['name']:<40} {'YES' if g.get('sacred') else 'no':<7} {g.get('strategy','balanced'):<10} {g.get('activations',0):<12}")


def cmd_create(args):
    data = load_genes()
    genes = data["genes"]
    if any(g["id"] == args.id for g in genes):
        print(f"ERROR: gene '{args.id}' already exists")
        sys.exit(1)
    gene = {
        "id": args.id,
        "name": args.name,
        "category": args.category or "workflow",
        "strategy": args.strategy or "balanced",
        "dna": "🧬",
        "trigger": args.trigger,
        "action": args.action,
        "sacred": args.sacred,
        "source": args.source or "manual",
        "created": datetime.datetime.utcnow().isoformat()[:10],
        "activations": 0,
    }
    genes.append(gene)
    save_genes(data)
    log_event(args.id, "create", f"created via CLI", "ok")
    print(f"Created gene: {args.id} ({args.name})")


def cmd_activate(args):
    data = load_genes()
    gene = next((g for g in data["genes"] if g["id"] == args.gene_id), None)
    if not gene:
        print(f"ERROR: gene '{args.gene_id}' not found")
        sys.exit(1)
    gene["activations"] = gene.get("activations", 0) + 1
    save_genes(data)
    log_event(args.gene_id, "activation", args.context or "manual", args.result or "ok")
    print(f"Activated gene: {args.gene_id} (total activations: {gene['activations']})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd")

    sub.add_parser("list", help="List all genes")

    p_create = sub.add_parser("create", help="Create a new gene")
    p_create.add_argument("--id", required=True)
    p_create.add_argument("--name", required=True)
    p_create.add_argument("--trigger", required=True)
    p_create.add_argument("--action", required=True)
    p_create.add_argument("--category", default="workflow")
    p_create.add_argument("--strategy", default="balanced")
    p_create.add_argument("--sacred", action="store_true")
    p_create.add_argument("--source", default="manual")

    p_act = sub.add_parser("activate", help="Record a gene activation")
    p_act.add_argument("gene_id")
    p_act.add_argument("--context", default="")
    p_act.add_argument("--result", default="ok")

    args = parser.parse_args()
    if args.cmd == "list":
        cmd_list()
    elif args.cmd == "create":
        cmd_create(args)
    elif args.cmd == "activate":
        cmd_activate(args)
    else:
        parser.print_help()
