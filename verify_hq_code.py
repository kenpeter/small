import pandas as pd, json
from pathlib import Path
from collections import Counter

df = pd.read_json('/mnt/file_drive/data/high_quality_leetcode/train.jsonl', lines=True)
out_dir = Path('/home/kenpeter/work/data/_code_gold_hq')

def verify_row(r):
    ns = {}
    try:
        exec(r['completion'], ns)          # defines class Solution
        candidate = eval(r['entry_point'], ns)   # Solution().twoSum (bound method)
        ns2 = {}
        exec(r['test'], ns2)               # defines check
        check = ns2['check']
        check(candidate)                    # runs ALL asserts
        return {'task_id': r['task_id'], 'difficulty': r['difficulty'],
                'tags': r.get('tags'),
                'problem': r['problem_description'], 'completion': r['completion'],
                'entry_point': r['entry_point']}
    except Exception:
        return None

passed = []
for _, r in df.iterrows():
    res = verify_row(r)
    if res is not None:
        passed.append(res)

print(f"PASSED: {len(passed)} / {len(df)}")
print("difficulty:", dict(Counter(p['difficulty'] for p in passed)))
out_dir.mkdir(parents=True, exist_ok=True)
with open(out_dir / 'code_gold_hq.jsonl', 'w') as f:
    for p in passed:
        f.write(json.dumps(p) + '\n')
print(f"wrote {len(passed)} → {out_dir/'code_gold_hq.jsonl'}")
