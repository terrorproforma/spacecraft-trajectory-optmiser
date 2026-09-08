import sys
sys.argv=['analysis','/home/ubuntu/spacepdhcg-workspace-pool-replay-v486']
exec("from pathlib import Path\nimport json,sys,collections\nroot=Path(sys.argv[1]);r=json.loads((root/'report.json').read_text());assert r['complete'] and not r.get('error')\ngroups={name:json.loads((root/name/'results.json').read_text()) for name in ['baseline','candidate']}\nresult={};mass_delta=0.;lost=[]\nfor name,rows in groups.items():\n    assert len(rows)==225 and [r['index'] for r in rows]==list(range(225))\n    assert all(r.get('certified',False) for r in rows if r['status']=='converged')\n    result[name]=dict(seconds=sum(r['seconds'] for r in rows),converged=sum(r['status']=='converged' for r in rows),priming=sum(r['priming'] for r in rows),control_bytes=sum(r['outer_transfer_bytes']['control_download_bytes'] for r in rows),status_counts=dict(collections.Counter(r['status'] for r in rows)),lost_prior=[r['index'] for r in rows if r['prior_status']=='converged' and r['status']!='converged'])\nfor old,new in zip(groups['baseline'],groups['candidate'],strict=True):\n    if old['status']=='converged':\n        if new['status']!='converged':lost.append(old['index'])\n        else:mass_delta=max(mass_delta,abs(old['certificate']['final_mass_kg']-new['certificate']['final_mass_kg']))\nresult.update(lost_baseline=lost,max_certified_mass_delta_kg=mass_delta,less_time_percent=100*(1-result['candidate']['seconds']/result['baseline']['seconds']))\n(root/'analysis.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))\n")
from pathlib import Path
import json,hashlib,tarfile,shutil
root=Path('/home/ubuntu/spacepdhcg-workspace-pool-replay-v486')
r=json.loads((root/'report.json').read_text());assert r['complete'] and not r.get('error')
for name,sha in r['source_sha256'].items():
 p=root/'repo'/name;assert hashlib.sha256(p.read_bytes()).hexdigest()==sha
 target=root/'source-overlay'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
for c in r['campaigns']:
 f=json.loads((root/c['name']/'output/run_report.json').read_text());assert f['best']['accepted'] and f['best']['official']['ok'] and f['best']['independent']['ok']
paths=[p for p in root.rglob('*') if p.is_file() and p.relative_to(root).parts[0] not in ['repo','core-build','compact-options-probe'] and p.name not in ['files-sha256.json','workspace-pool-probe']]
manifest={p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}
(root/'files-sha256.json').write_text(json.dumps(manifest,indent=2))
archive=root.with_suffix('.tar.gz')
with tarfile.open(archive,'w:gz') as t:
 for name in list(manifest)+['files-sha256.json']:t.add(root/name,arcname=name,recursive=False)
print(json.dumps(dict(path=str(archive),sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),bytes=archive.stat().st_size,files=len(manifest))))
