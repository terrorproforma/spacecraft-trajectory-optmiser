from pathlib import Path
import json,statistics
root=Path('/home/ubuntu/spacepdhcg-early-graph-v471')
r=json.loads((root/'report.json').read_text());assert r['complete'] and not r.get('error')
old=json.loads((root/'baseline/results.json').read_text());new=json.loads((root/'candidate/results.json').read_text())
assert len(old)==len(new)==225
rows=[]
for a,b in zip(old,new,strict=True):
 assert a['index']==b['index']
 rows.append(dict(index=a['index'],old_status=a['status'],new_status=b['status'],old=a['seconds'],new=b['seconds'],delta=b['seconds']-a['seconds'],old_iters=a['iterations'],new_iters=b['iterations']))
summary={}
for name,group in [('all',rows),('baseline_converged',[x for x in rows if x['old_status']=='converged']),('baseline_unsuccessful',[x for x in rows if x['old_status']!='converged'])]:
 summary[name]=dict(cases=len(group),old_seconds=sum(x['old'] for x in group),new_seconds=sum(x['new'] for x in group),median_ratio=statistics.median(x['new']/x['old'] for x in group))
summary['slowest_deltas']=sorted(rows,key=lambda x:-x['delta'])[:8]
summary['fastest_deltas']=sorted(rows,key=lambda x:x['delta'])[:5]
summary['uncertified']=[x['index'] for x in new if x['status']=='converged' and not x['certified']]
summary['lost_baseline']=[x['index'] for x in rows if x['old_status']=='converged' and x['new_status']!='converged']
(root/'performance-analysis.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
