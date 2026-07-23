#!/usr/bin/env python3
"""Search the web via DuckDuckGo Lite. No API key needed.

Usage:
    python3 search_web.py "muon optimizer explained" --max 5
    python3 search_web.py "site:arxiv.org muon optimizer" --max 3
"""
import sys
import urllib.request
import urllib.parse
import re


def search_ddg(query, max_results=5):
    # Use DuckDuckGo Lite (less JavaScript, easier to scrape)
    q = urllib.parse.quote_plus(query)
    url = f"https://lite.duckduckgo.com/lite/?q={q}"

    req = urllib.request.Request(url, headers={
        'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html',
    })

    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            html = resp.read().decode('utf-8', errors='ignore')
    except Exception as e:
        print(f"Search failed: {type(e).__name__}: {e}")
        sys.exit(1)

    # Parse results from DuckDuckGo Lite HTML
    results = []
    # Each result is: <a class="result-link" href="...">title</a>
    # followed by <td class="result-snippet">snippet</td>
    links = re.findall(r'<a[^>]+class="result-link"[^>]+href="([^"]+)"[^>]*>(.*?)</a>', html, re.DOTALL)
    snippets = re.findall(r'<td[^>]+class="result-snippet"[^>]*>(.*?)</td>', html, re.DOTALL)

    for i, (href, title_raw) in enumerate(links):
        if i >= max_results:
            break
        # Clean HTML tags from title
        title = re.sub(r'<[^>]+>', '', title_raw).strip()
        # Resolve redirects (DDG uses redirect URLs)
        if href.startswith('/'):
            href = 'https://lite.duckduckgo.com' + href
        elif not href.startswith('http'):
            continue
        # Try to extract real URL from DDG redirect
        real_url_match = re.search(r'uddg=([^&]+)', href)
        if real_url_match:
            href = urllib.parse.unquote(real_url_match.group(1))

        snippet = re.sub(r'<[^>]+>', '', snippets[i]).strip() if i < len(snippets) else ""
        results.append((title, href, snippet))

    if not results:
        print("No results found.")
        return

    print(f"Found {len(results)} result(s)\n")
    for i, (title, url, snippet) in enumerate(results):
        print(f"{i+1}. {title}")
        print(f"   {url}")
        if snippet:
            print(f"   {snippet[:200]}{'...' if len(snippet) > 200 else ''}")
        print()


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args or args[0] in ("-h", "--help"):
        print(__doc__)
        sys.exit(0)

    query = None
    max_results = 5

    i = 0
    while i < len(args):
        if args[i] == "--max" and i + 1 < len(args):
            max_results = int(args[i + 1]); i += 2
        else:
            query = " ".join(args[i:]) if query is None else query + " " + args[i]
            i += 1

    search_ddg(query, max_results)
