from pathlib import Path
import hashlib,json,tarfile
base=Path('/home/ubuntu')
sources={
 'build-v557':base/'spacepdhcg-device-init-v557',
 'build-v565':base/'spacepdhcg-device-init-v565',
 'validation-v556':base/'spacepdhcg-device-init-v557/repo/build/performance/device-init-v556',
 'validation-v559':base/'spacepdhcg-device-init-v557/repo/build/performance/device-init-v559',
 'replay-v563':base/'spacepdhcg-device-init-v557/repo/build/performance/device-init-v563',
 'validation-v565':base/'spacepdhcg-device-init-v565/repo/build/performance/device-init-v565',
 'campaign-v561':base/'spacepdhcg-device-init-campaign-v561',
}
for label,root in sources.items():
 r=json.loads((root/'report.json').read_text())
 if label in ['build-v557','validation-v556','validation-v559']:assert r.get('error'),label
 else:assert r['complete'] and not r.get('error'),(label,r)
assert '65 passed' in (sources['validation-v565']/'pytest.log').read_text()
assert 'ERROR SUMMARY: 0 errors' in (sources['validation-v565']/'memcheck.log').read_text()
files={}
for label,root in sources.items():
 for path in sorted(root.rglob('*')):
  if not path.is_file():continue
  rel=path.relative_to(root)
  if set(rel.parts)&{'__pycache__','.git'}:continue
  if label.startswith('build-'):
   if rel.parts[0]=='repo':continue
   if rel.parts[0]=='core-build' and rel.as_posix() not in ['core-build/cuda/libspacepdhcg_cuda.so','core-build/CMakeCache.txt']:continue
  files[label+'/'+rel.as_posix()]=path
manifest={name:hashlib.sha256(path.read_bytes()).hexdigest() for name,path in files.items()}
output=base/'spacepdhcg-device-init-results-v566.tar.gz'
assert not output.exists()
meta=base/'spacepdhcg-device-init-results-v566-manifest.json';meta.write_text(json.dumps(manifest,indent=2))
with tarfile.open(output,'w:gz') as t:
 for name,path in files.items():t.add(path,arcname=name,recursive=False)
 t.add(meta,arcname='archive-manifest.json',recursive=False)
print(json.dumps(dict(path=str(output),sha256=hashlib.sha256(output.read_bytes()).hexdigest(),bytes=output.stat().st_size,files=len(files))))
