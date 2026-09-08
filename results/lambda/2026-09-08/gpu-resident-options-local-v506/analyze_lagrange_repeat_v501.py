from pathlib import Path
import json,collections
root=Path('/home/ubuntu/spacepdhcg-lagrange-repeat-v501');r=json.loads((root/'report.json').read_text());assert r['complete'] and not r.get('error')
summary={}
for name in ['prior','candidate']:
 rows=json.loads((root/name/'solves.json').read_text());groups=collections.defaultdict(list)
 for row in rows:groups[(row['precision'],row['origin'],row['boundary'])].append(row)
 data=[]
 for (precision,origin,boundary),values in groups.items():
  fresh=[v['mass'] for v in values if v['pool']==0];reused=[v['mass'] for v in values if v['pool']==1]
  data.append(dict(precision=precision,origin=origin,boundary=boundary,fresh_spread=max(fresh)-min(fresh),pool_spread=max(reused)-min(reused),combined_spread=max(fresh+reused)-min(fresh+reused),statuses=dict(collections.Counter(v['status'] for v in values)),max_iterations=max(v['iterations'] for v in values)))
 summary[name]=data
(root/'analysis.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
