#!/usr/bin/env python3
"""Search GitHub repositories. No API key needed for basic search.

Usage:
    python3 search_github.py "muon optimizer" --max 5
    python3 search_github.py "muon optimizer" language:python --max 5 --sort stars
    python3 search_github.py --user KellerJordan --max 10
"""
import sys
import urllib.request
import urllib.parse
import json


def search_repos(query, max_results=5, sort="stars", order="desc", language=None, user=None):
    parts = [query] if query else []
    if language:
        parts.append(f"language:{language}")
    if user:
        parts.append(f"user:{user}")
    q = "+".join(parts)

    url = (
        f"https://api.github.com/search/repositories?"
        f"q={urllib.parse.quote(q)}"
        f"&sort={sort}&order={order}&per_page={max_results}"
    )

    req = urllib.request.Request(url, headers={
        'User-Agent': 'HermesAgent/1.0',
        'Accept': 'application/vnd.github.v3+json'
    })

    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        if e.code == 403:
            print("GitHub API rate limit hit (60 req/hr for unauthenticated). Try again in an hour.")
        else:
            print(f"GitHub API error: {e.code} {e.reason}")
        sys.exit(1)

    items = data.get("items", [])
    total = data.get("total_count", 0)

    if not items:
        print("No repositories found.")
        return

    print(f"Found {total:,} repos (showing {len(items)})\n")
    for i, r in enumerate(items):
        stars = r.get("stargazers_count", 0)
        forks = r.get("forks_count", 0)
        lang = r.get("language", "N/A")
        desc = (r.get("description") or "No description")[:120]
        updated = r.get("updated_at", "N/A")[:10]
        print(f"{i+1}. {r['full_name']}  ⭐ {stars:,}  🍴 {forks:,}  [{lang}]  (updated {updated})")
        print(f"   {desc}")
        print(f"   https://github.com/{r['full_name']}")
        print()


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args or args[0] in ("-h", "--help"):
        print(__doc__)
        sys.exit(0)

    query = None
    max_results = 5
    sort = "stars"
    language = None
    user = None

    i = 0
    while i < len(args):
        if args[i] == "--max" and i + 1 < len(args):
            max_results = int(args[i + 1]); i += 2
        elif args[i] == "--sort" and i + 1 < len(args):
            sort = args[i + 1]; i += 2
        elif args[i] == "--user" and i + 1 < len(args):
            user = args[i + 1]; i += 2
        elif args[i].startswith("language:"):
            language = args[i].split(":", 1)[1]; i += 1
        else:
            query = " ".join(args[i:]) if query is None else query + " " + args[i]
            i += 1

    search_repos(query, max_results, sort, language=language, user=user)
