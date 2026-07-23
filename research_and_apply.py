#!/usr/bin/env python3
"""
Research a topic and map findings to the training codebase.

Usage:
    python3 research_and_apply.py "muon optimizer" --max-papers 3 --max-repos 3
    python3 research_and_apply.py "cosine learning rate schedule" --extract-only
    python3 research_and_apply.py "attention residual connections" --full-paper

Pipeline:
    1. Search arXiv (latest papers)
    2. Search Semantic Scholar (citations, impact)
    3. Search GitHub (code implementations)
    4. Extract paper text (jina.ai or PDF)
    5. Search web (blogs, tutorials)
    6. Synthesize: hyperparameters, architecture changes, code patterns
    7. Map to pretrain_megatrain.py (compatibility check)
    8. Output actionable patch suggestions
"""
import sys, os, json, urllib.request, urllib.parse, re, tempfile, shutil, subprocess

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MAX_PAPERS = 3
MAX_REPOS = 3
MAX_WEB = 3

def run(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True)

# ---------------------------------------------------------------------------
# 1. arXiv Search
# ---------------------------------------------------------------------------
def search_arxiv(query, max_results=3):
    q = urllib.parse.quote(query)
    url = f"https://export.arxiv.org/api/query?search_query=all:{q}&max_results={max_results}&sortBy=submittedDate&sortOrder=descending"
    req = urllib.request.Request(url, headers={'User-Agent': 'HermesAgent/1.0'})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = resp.read()
    except Exception as e:
        return []
    import xml.etree.ElementTree as ET
    ns = {'a': 'http://www.w3.org/2005/Atom'}
    root = ET.fromstring(data)
    entries = []
    for entry in root.findall('a:entry', ns):
        title = entry.find('a:title', ns).text.strip().replace('\n', ' ')
        raw_id = entry.find('a:id', ns).text.strip()
        arxiv_id = raw_id.split('/abs/')[-1].split('v')[0]
        published = entry.find('a:published', ns).text[:10]
        authors = ', '.join(a.find('a:name', ns).text for a in entry.findall('a:author', ns))
        summary = entry.find('a:summary', ns).text.strip().replace('\n', ' ')[:300]
        entries.append({
            'title': title, 'id': arxiv_id, 'published': published,
            'authors': authors, 'abstract': summary,
            'pdf': f"https://arxiv.org/pdf/{arxiv_id}.pdf",
            'abs': f"https://arxiv.org/abs/{arxiv_id}",
        })
    return entries

# ---------------------------------------------------------------------------
# 2. Semantic Scholar (citations + open access)
# ---------------------------------------------------------------------------
def search_semantic(query, max_results=3):
    q = urllib.parse.quote(query)
    url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={q}&limit={max_results}&fields=title,authors,year,citationCount,referenceCount,influentialCitationCount,externalIds,openAccessPdf"
    req = urllib.request.Request(url, headers={'User-Agent': 'HermesAgent/1.0'})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())
    except Exception as e:
        return []
    papers = []
    for p in data.get('data', []):
        eids = p.get('externalIds', {})
        papers.append({
            'title': p.get('title', ''),
            'year': p.get('year', ''),
            'citations': p.get('citationCount', 0),
            'influential': p.get('influentialCitationCount', 0),
            'arxiv': eids.get('ArXiv', ''),
            'open_access': p.get('openAccessPdf', {}).get('url', ''),
        })
    return papers

# ---------------------------------------------------------------------------
# 3. GitHub Search
# ---------------------------------------------------------------------------
def search_github(query, max_results=3):
    q = urllib.parse.quote(query)
    url = f"https://api.github.com/search/repositories?q={q}&sort=stars&order=desc&per_page={max_results}"
    req = urllib.request.Request(url, headers={'User-Agent': 'HermesAgent/1.0', 'Accept': 'application/vnd.github.v3+json'})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())
    except Exception as e:
        return []
    repos = []
    for r in data.get('items', []):
        repos.append({
            'name': r['full_name'],
            'stars': r.get('stargazers_count', 0),
            'lang': r.get('language', ''),
            'desc': (r.get('description') or '')[:120],
            'url': f"https://github.com/{r['full_name']}",
            'updated': r.get('updated_at', '')[:10],
        })
    return repos

# ---------------------------------------------------------------------------
# 4. Web Search (DuckDuckGo Lite)
# ---------------------------------------------------------------------------
def search_web(query, max_results=3):
    q = urllib.parse.quote_plus(query)
    url = f"https://lite.duckduckgo.com/lite/?q={q}"
    req = urllib.request.Request(url, headers={
        'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36',
        'Accept': 'text/html',
    })
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            html = resp.read().decode('utf-8', errors='ignore')
    except Exception:
        return []
    links = re.findall(r'<a[^>]+class="result-link"[^>]+href="([^"]+)"[^>]*>(.*?)</a>', html, re.DOTALL)
    snippets = re.findall(r'<td[^>]+class="result-snippet"[^>]*>(.*?)</td>', html, re.DOTALL)
    results = []
    for i, (href, title_raw) in enumerate(links):
        if i >= max_results:
            break
        title = re.sub(r'<[^>]+>', '', title_raw).strip()
        real_url_match = re.search(r'uddg=([^&]+)', href)
        if real_url_match:
            href = urllib.parse.unquote(real_url_match.group(1))
        snippet = re.sub(r'<[^>]+>', '', snippets[i]).strip() if i < len(snippets) else ""
        results.append({'title': title, 'url': href, 'snippet': snippet[:200]})
    return results

# ---------------------------------------------------------------------------
# 5. Extract Paper Text (jina.ai proxy)
# ---------------------------------------------------------------------------
def extract_paper_text(arxiv_id):
    """Extract full paper text via jina.ai. Falls back to PDF if jina fails."""
    jina_url = f"https://r.jina.ai/http://arxiv.org/pdf/{arxiv_id}"
    try:
        req = urllib.request.Request(jina_url, headers={'User-Agent': 'HermesAgent/1.0'})
        with urllib.request.urlopen(req, timeout=30) as resp:
            text = resp.read().decode('utf-8', errors='ignore')
        if len(text) > 500 and 'References' in text:
            return text
    except Exception:
        pass
    # Fallback: download PDF
    pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    tmpdir = tempfile.mkdtemp()
    pdf_path = os.path.join(tmpdir, "paper.pdf")
    txt_path = os.path.join(tmpdir, "paper.txt")
    try:
        urllib.request.urlretrieve(pdf_url, pdf_path)
        run(f"pdftotext '{pdf_path}' '{txt_path}' 2>/dev/null")
        if os.path.exists(txt_path):
            with open(txt_path) as f:
                return f.read()
    except Exception:
        pass
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
    return ""

# ---------------------------------------------------------------------------
# 6. Extract key sections from paper text
# ---------------------------------------------------------------------------
def extract_key_sections(text):
    """Extract hyperparameters and algorithm details from paper text."""
    findings = {}
    # LR
    lr_matches = re.findall(r'learning rate[\s:=]+([0-9.e\-]+)', text, re.I)
    if lr_matches:
        findings['learning_rate'] = lr_matches
    # Batch size
    bs_matches = re.findall(r'batch size[\s:=]+([0-9,]+)', text, re.I)
    if bs_matches:
        findings['batch_size'] = bs_matches
    # Warmup
    wm_matches = re.findall(r'warm[- ]?up[\s:=]+([0-9,]+)', text, re.I)
    if wm_matches:
        findings['warmup'] = wm_matches
    # Optimizer
    opt_matches = re.findall(r'optimizer[\s:=]+([A-Za-z0-9\s\-]+)', text, re.I)
    if opt_matches:
        findings['optimizer'] = opt_matches[:3]
    # Weight decay
    wd_matches = re.findall(r'weight decay[\s:=]+([0-9.e\-]+)', text, re.I)
    if wd_matches:
        findings['weight_decay'] = wd_matches
    # Gradient clipping
    gc_matches = re.findall(r'gradient clipping|clip_grad_norm|max_grad_norm[\s:=]+([0-9.e\-]+)', text, re.I)
    if gc_matches:
        findings['grad_clip'] = gc_matches
    # Scheduler
    sched_matches = re.findall(r'(cosine|linear|exponential|constant|warmup)[\s\-]*(schedule|decay|lr)', text, re.I)
    if sched_matches:
        findings['scheduler'] = [m[0] for m in sched_matches[:3]]
    return findings

# ---------------------------------------------------------------------------
# 7. Map to codebase compatibility
# ---------------------------------------------------------------------------
def map_to_codebase(findings):
    """Check if findings are compatible with CPUMasterModel + pretrain_megatrain.py."""
    issues = []
    notes = []
    # Check for FSDP/DeepSpeed mentions
    if any(k in str(findings).lower() for k in ['fsdp', 'deepspeed', 'distributed', 'megatron']):
        issues.append("Paper uses distributed training (FSDP/DeepSpeed/Megatron) — may not apply to single-GPU CPUMasterModel")
    # Check for flash-attention
    if 'flash' in str(findings).lower():
        notes.append("Flash Attention mentioned — already using SDPA, flash-attn CE not available")
    # Check scale
    if any('70B' in str(v) or '100B' in str(v) or 'large' in str(v).lower() for v in findings.values()):
        notes.append("Paper validated on large models — hyperparameters may need scaling down for 1B")
    # Check for architecture changes
    if any(k in str(findings).lower() for k in ['mamba', 'moe', 'sparse', 'expert']):
        issues.append("Architecture change (MoE/Mamba/sparse) — incompatible with current Llama stack")
    return issues, notes

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    query = sys.argv[1]
    max_papers = int(sys.argv[sys.argv.index("--max-papers") + 1]) if "--max-papers" in sys.argv else MAX_PAPERS
    max_repos = int(sys.argv[sys.argv.index("--max-repos") + 1]) if "--max-repos" in sys.argv else MAX_REPOS
    extract_only = "--extract-only" in sys.argv
    full_paper = "--full-paper" in sys.argv

    print(f"{'='*60}")
    print(f"RESEARCH TOPIC: {query}")
    print(f"{'='*60}\n")

    # Phase 1: Search all sources
    print("[1/4] Searching arXiv...")
    arxiv_papers = search_arxiv(query, max_papers)
    print(f"      Found {len(arxiv_papers)} paper(s)")

    print("[2/4] Searching Semantic Scholar...")
    scholar_papers = search_semantic(query, max_papers)
    print(f"      Found {len(scholar_papers)} paper(s)")

    print("[3/4] Searching GitHub...")
    repos = search_github(query, max_repos)
    print(f"      Found {len(repos)} repo(s)")

    print("[4/4] Searching web...")
    web_results = search_web(query, MAX_WEB)
    print(f"      Found {len(web_results)} result(s)\n")

    # Phase 2: Display synthesis
    print(f"{'='*60}")
    print("SYNTHESIS")
    print(f"{'='*60}\n")

    # Papers
    if arxiv_papers:
        print("## Top Papers (arXiv)")
        for i, p in enumerate(arxiv_papers):
            print(f"{i+1}. [{p['id']}] {p['title']}")
            print(f"   Authors: {p['authors']}")
            print(f"   Published: {p['published']}")
            print(f"   Abstract: {p['abstract']}...")
            print(f"   PDF: {p['pdf']}")
            print()

    # Scholar impact
    if scholar_papers:
        print("## Impact (Semantic Scholar)")
        for p in scholar_papers:
            print(f"- {p['title']} ({p['year']}) — cited {p['citations']}x, influential {p['influential']}x")
            if p['arxiv']:
                print(f"  arXiv: {p['arxiv']}")
            if p['open_access']:
                print(f"  Open access: {p['open_access']}")
        print()

    # Code
    if repos:
        print("## Code Implementations")
        for i, r in enumerate(repos):
            print(f"{i+1}. {r['name']}  ⭐ {r['stars']:,}  [{r['lang']}]  (updated {r['updated']})")
            print(f"   {r['desc']}")
            print(f"   {r['url']}")
            print()

    # Web
    if web_results:
        print("## Web Results")
        for i, w in enumerate(web_results):
            print(f"{i+1}. {w['title']}")
            print(f"   {w['url']}")
            if w['snippet']:
                print(f"   {w['snippet']}")
            print()

    # Phase 3: Deep dive on first arXiv paper
    if arxiv_papers and (full_paper or not extract_only):
        print(f"{'='*60}")
        print(f"DEEP DIVE: {arxiv_papers[0]['id']}")
        print(f"{'='*60}\n")
        text = extract_paper_text(arxiv_papers[0]['id'])
        if text:
            findings = extract_key_sections(text)
            if findings:
                print("## Extracted Hyperparameters / Settings")
                for k, v in findings.items():
                    print(f"  {k}: {v}")
                print()

            issues, notes = map_to_codebase(findings)
            if issues:
                print("## ⚠️ Compatibility Issues")
                for issue in issues:
                    print(f"  - {issue}")
                print()
            if notes:
                print("## ℹ️ Notes")
                for note in notes:
                    print(f"  - {note}")
                print()

            # Save full text for manual inspection
            out_path = f"/tmp/research_{arxiv_papers[0]['id']}.txt"
            with open(out_path, 'w') as f:
                f.write(text)
            print(f"## Full paper text saved to: {out_path}")
        else:
            print("Could not extract paper text.")

    print(f"\n{'='*60}")
    print("NEXT STEPS")
    print(f"{'='*60}")
    print("1. Review findings above")
    print("2. Read full paper text file")
    print("3. Check compatibility issues")
    print("4. If compatible, apply to pretrain_megatrain.py")
    print("5. Run: python3 test_pretrain.py  # must pass before training")


if __name__ == "__main__":
    main()
