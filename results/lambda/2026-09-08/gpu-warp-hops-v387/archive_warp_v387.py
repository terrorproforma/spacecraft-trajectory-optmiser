from pathlib import Path
import json,hashlib,subprocess,shutil,tarfile
root=Path('/home/ubuntu/spacepdhcg-warp-hops-v387')
r=json.loads((root/'report.json').read_text());assert r['complete'] and not r.get('error')
native=Path('/home/ubuntu/spacepdhcg-warp-native-v390')
assert json.loads((native/'report.json').read_text())['complete']
program='''
from pathlib import Path
import json,numpy as np
root=Path('/home/ubuntu/spacepdhcg-warp-hops-v387');cases=0
for ref in sorted((root/'micro_baseline0').glob('*.npz')):
 a=np.load(ref)
 for mode in ['candidate0','candidate1','baseline1']:
  b=np.load(root/('micro_'+mode)/ref.name)
  assert set(a.files)==set(b.files)
  for key in a.files:assert np.array_equal(a[key],b[key],equal_nan=True),(ref.name,mode,key)
  cases+=1
assert cases==216,cases
reports={m:json.loads((root/('micro_'+m)/'report.json').read_text())['rows'] for m in ['baseline0','candidate0','candidate1','baseline1']}
timings=[]
for i,row in enumerate(reports['baseline0']):
 baseline=float(np.median(reports['baseline0'][i]['seconds']+reports['baseline1'][i]['seconds']))
 candidate=float(np.median(reports['candidate0'][i]['seconds']+reports['candidate1'][i]['seconds']))
 timings.append(dict(count=row['count'],baseline_ms=1000*baseline,candidate_ms=1000*candidate,speedup=baseline/candidate))
(root/'audit.json').write_text(json.dumps(dict(compared_outputs=cases,all_fields_exact_equal=True,timings=timings),indent=2))
'''
subprocess.run(['/home/ubuntu/spacepdhcg/v1/.venv/bin/python','-c',program],check=True)
shutil.copytree(native,root/'native-v390',ignore=shutil.ignore_patterns('probe'))
for name,sha in r['source_sha256'].items():
 p=root/'repo'/name;assert hashlib.sha256(p.read_bytes()).hexdigest()==sha
 target=root/'source-overlay'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
campaigns=r['campaigns'];assert len(campaigns)==4
assert all(c['screening']==campaigns[0]['screening'] for c in campaigns)
for c in campaigns:
 f=json.loads((root/c['name']/'output/run_report.json').read_text())
 assert f['best']['official']['ok'] and f['best']['independent']['ok']
paths=[p for p in root.rglob('*') if p.is_file() and p.relative_to(root).parts[0] not in ['repo','core-build'] and p.name!='files-sha256.json']
manifest={p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}
(root/'files-sha256.json').write_text(json.dumps(manifest,indent=2))
archive=root.with_suffix('.tar.gz')
with tarfile.open(archive,'w:gz') as t:
 for name in list(manifest)+['files-sha256.json']:t.add(root/name,arcname=name,recursive=False)
print(json.dumps(dict(path=str(archive),sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),bytes=archive.stat().st_size,files=len(manifest))))
