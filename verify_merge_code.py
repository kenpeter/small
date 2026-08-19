import pandas as pd, json
from pathlib import Path
from collections import Counter

def verify_rows(df, src):
    out = []
    for _, r in df.iterrows():
        ns = {}
        try:
            exec(r['completion'], ns)
            candidate = eval(r['entry_point'], ns)
            ns2 = {}
            exec(r['test'], ns2)
            ns2['check'](candidate)
            out.append({'task_id': r['task_id'], 'difficulty': r['difficulty'],
                        'tags': r.get('tags'), 'problem': r['problem_description'],
                        'completion': r['completion'], 'entry_point': r['entry_point'],
                        'source': src})
        except Exception:
            pass
    return out

hq = pd.read_json('/mnt/file_drive/data/high_quality_leetcode/train.jsonl', lines=True)
nf = pd.read_json('/mnt/file_drive/data/newfacade_LeetCodeDataset/leetcode_train.jsonl', lines=True)

hq_v = verify_rows(hq, 'high_quality')
nf_v = verify_rows(nf, 'newfacade')
print(f"HQ verified: {len(hq_v)}   newfacade verified: {len(nf_v)}")

by_id = {}
for p in hq_v + nf_v:
    by_id.setdefault(p['task_id'], []).append(p)

# choose completion from whichever passes; keep one per task_id
merged = []
for tid, lst in by_id.items():
    # prefer HQ completion if present
    merged.append(lst[0])
print(f"unique verified task_ids: {len(merged)}")
print("difficulty:", dict(Counter(p['difficulty'] for p in merged)))

out_dir = Path('/home/kenpeter/work/data/_code_gold_hq')
out_dir.mkdir(parents=True, exist_ok=True)
with open(out_dir / 'code_gold_hq_merged.jsonl', 'w') as f:
    for p in merged:
        f.write(json.dumps(p) + '\n')
print(f"wrote {len(merged)} → code_gold_hq_merged.jsonl")
