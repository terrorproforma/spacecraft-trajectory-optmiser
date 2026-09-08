from pathlib import Path
import json

root=Path('/home/angus/spacepdhcg-retry-fleet-legs-v688')
report=json.loads((root/'report.json').read_text());assert report['complete']
a,b=[json.loads((root/name/'results.json').read_text()) for name in ('baseline','candidate')]
assert len(a)==len(b)==225
def good(r):return r['status']=='converged' and r.get('certified',False) and r.get('mass_margin_kg',-1)>=0
lost=[];gained=[];mass=[]
for x,y in zip(a,b,strict=True):
    assert x['index']==y['index']
    if good(x) and not good(y):lost.append(y['index'])
    if good(y) and not good(x):gained.append(y['index'])
    if good(x) and good(y):mass.append(abs(x['certificate']['final_mass_kg']-y['certificate']['final_mass_kg']))
summary=dict(cases=225,baseline_certified=sum(map(good,a)),candidate_certified=sum(map(good,b)),lost=lost,gained=gained,maximum_shared_final_mass_difference_kg=max(mass),baseline_solver_seconds=sum(r['seconds'] for r in a),candidate_solver_seconds=sum(r['seconds'] for r in b),baseline_creations=sum(r['workspace_creations'] for r in a),candidate_creations=sum(r['workspace_creations'] for r in b))
(root/'comparison.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary))
