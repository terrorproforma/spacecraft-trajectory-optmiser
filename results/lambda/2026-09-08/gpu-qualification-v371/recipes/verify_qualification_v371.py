from pathlib import Path
import json,gzip,hashlib,tarfile,sys,shutil
root=Path('results/lambda/2026-09-08/gpu-qualification-v371')
sys.path.insert(0,str(Path('build/performance/qp-ir-v309').resolve()))
from audit import problem,audit
qp=Path('build/performance/qp-ir-v309/qp.txt');assert hashlib.sha256(qp.read_bytes()).hexdigest()=='14fdc84e9b654df4ac16db71f0bce4fccfa62577f9620ae622ad2196e2f17080'
with (root/'qp.txt.gz').open('wb') as f:
 with gzip.GzipFile(filename='',mode='wb',fileobj=f,mtime=0) as z:z.write(qp.read_bytes())
shutil.copyfile('build/performance/qp-ir-v309/audit.py',root/'recipes/audit.py')
shutil.copyfile('build/performance/preserve_qualification_v371.py',root/'recipes/preserve_qualification_v371.py')
shutil.copyfile(__file__,root/'recipes/verify_qualification_v371.py')
def read(p):return gzip.decompress(p.read_bytes()).decode() if p.suffix=='.gz' else p.read_text()
data=problem(qp);count=0
for name in ['interior-nt-replay-v369b','kkt-equilibration-replay-v371']:
 report=json.loads((root/name/'report.json').read_text());assert report['complete']
 for row in report['rows']:
  p=root/name/(row['name']+'.log');p=p if p.exists() else p.with_suffix('.log.gz')
  records=[json.loads(s[10:]) for s in read(p).splitlines() if s.startswith('QP_REPLAY ')]
  audits=[audit(data,r) for r in records]
  assert len(audits)==16 and sum(a['qualified'] for a in audits)==row['qualified']
  for a,b in zip(audits,row['audits']):
   for key in ['qualified','primal','dual','gap','cone_violation']:assert a[key]==b[key],(name,key)
  count+=len(audits)
for folder in root.glob('frozen-v*'):
 expected=json.loads((folder/'source-sha256.json').read_text())
 with tarfile.open(folder/'source.tar.gz','r:gz') as t:
  got={m.name:hashlib.sha256(t.extractfile(m).read()).hexdigest() for m in t.getmembers() if m.isfile()}
 assert got==expected,folder
trace=json.loads((root/'division-trace-v370/report.json').read_text())
with tarfile.open(root/'division-trace-v370/snapshots.tar.gz','r:gz') as t:
 got={m.name:hashlib.sha256(t.extractfile(m).read()).hexdigest() for m in t.getmembers() if m.isfile()}
assert got=={r['file']:r['sha256'] for r in trace['rows']}
shutil.copyfile(__file__,root/'recipes/verify_qualification_v371.py')
(root/'evidence-sha256.json').write_text(json.dumps({str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.rglob('*')) if p.is_file() and p.name!='evidence-sha256.json'},indent=2))
print('Re-audited',count,'QP results; verified three source archives and',len(got),'trace snapshots')
