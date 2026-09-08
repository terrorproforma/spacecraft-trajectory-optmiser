from pathlib import Path
import hashlib,json,tarfile
base=Path('/home/ubuntu')
sources={
 'build-v545':base/'spacepdhcg-scaled-pool-v545',
 'failed-build-v541':base/'spacepdhcg-scaled-pool-v541',
 'validation-v545':base/'spacepdhcg-scaled-pool-v545/repo/build/performance/scaled-pool-v542',
 'campaign-v548':base/'spacepdhcg-scaled-pool-campaign-v548',
 'replay-v549':base/'spacepdhcg-scaled-followup-v549',
 'diagnostic-v550':base/'spacepdhcg-scaled-followup-v550',
 'scoped-v552':base/'spacepdhcg-scaled-followup-v552',
 'baseline-race-v554':base/'spacepdhcg-scaled-followup-v554',
}
for label,root in sources.items():
 r=json.loads((root/'report.json').read_text())
 if label in ['build-v545','failed-build-v541','validation-v545']:assert r.get('error'),label
 else:assert r['complete'] and not r.get('error'),(label,r)
assert '117 passed' in (sources['validation-v545']/'pytest.log').read_text()
for label in ['baseline','candidate']:
 r=json.loads((sources['replay-v549']/label/'results.json').read_text())
 assert len(r)==225 and all(v.get('certified') for v in r if v['status']=='converged')
files={}
for label,root in sources.items():
 for path in sorted(root.rglob('*')):
  if not path.is_file():continue
  rel=path.relative_to(root)
  if '__pycache__' in rel.parts or '.git' in rel.parts:continue
  if label=='failed-build-v541' and len(rel.parts)>1:continue
  if label=='build-v545':
   if rel.parts[0] in ['build','repo']:continue
   if rel.parts[0]=='core-build' and rel.as_posix() not in ['core-build/cuda/libspacepdhcg_cuda.so','core-build/CMakeCache.txt']:continue
  files[label+'/'+rel.as_posix()]=path
manifest={name:hashlib.sha256(path.read_bytes()).hexdigest() for name,path in files.items()}
output=base/'spacepdhcg-scaled-results-v553.tar.gz'
assert not output.exists()
meta=base/'spacepdhcg-scaled-results-v553-manifest.json';meta.write_text(json.dumps(manifest,indent=2))
with tarfile.open(output,'w:gz') as t:
 for name,path in files.items():t.add(path,arcname=name,recursive=False)
 t.add(meta,arcname='archive-manifest.json',recursive=False)
print(json.dumps(dict(path=str(output),sha256=hashlib.sha256(output.read_bytes()).hexdigest(),bytes=output.stat().st_size,files=len(files))))
