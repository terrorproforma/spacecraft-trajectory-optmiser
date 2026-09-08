from pathlib import Path
import json,sys,collections
root=Path(sys.argv[1]);r=json.loads((root/'report.json').read_text());assert r['complete'] and not r.get('error')
groups={name:json.loads((root/name/'results.json').read_text()) for name in ['baseline','candidate']}
result={};mass_delta=0.;lost=[]
for name,rows in groups.items():
    assert len(rows)==225 and [r['index'] for r in rows]==list(range(225))
    assert all(r.get('certified',False) for r in rows if r['status']=='converged')
    result[name]=dict(seconds=sum(r['seconds'] for r in rows),converged=sum(r['status']=='converged' for r in rows),priming=sum(r['priming'] for r in rows),control_bytes=sum(r['outer_transfer_bytes']['control_download_bytes'] for r in rows),status_counts=dict(collections.Counter(r['status'] for r in rows)),lost_prior=[r['index'] for r in rows if r['prior_status']=='converged' and r['status']!='converged'])
for old,new in zip(groups['baseline'],groups['candidate'],strict=True):
    if old['status']=='converged':
        if new['status']!='converged':lost.append(old['index'])
        else:mass_delta=max(mass_delta,abs(old['certificate']['final_mass_kg']-new['certificate']['final_mass_kg']))
result.update(lost_baseline=lost,max_certified_mass_delta_kg=mass_delta,less_time_percent=100*(1-result['candidate']['seconds']/result['baseline']['seconds']))
(root/'analysis.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
