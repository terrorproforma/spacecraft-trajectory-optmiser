from pathlib import Path
import json
root=Path('/home/ubuntu/spacepdhcg-bound-types-v569/repo/build/performance/bound-types-v569')
for mode in ['baseline','candidate']:
 rows=json.loads((root/mode/'results.json').read_text())
 r=rows[87]
 print(mode,json.dumps({k:r[k] for k in ['status','diagnostic','workspace_creations','iterations','accepted','seconds']}))
 print('first',json.dumps(r['history'][:2]));print('last',json.dumps(r['history'][-2:]))
