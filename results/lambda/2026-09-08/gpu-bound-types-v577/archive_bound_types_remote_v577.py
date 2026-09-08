from pathlib import Path
import hashlib,json,tarfile
base=Path('/home/ubuntu')
sources={
 'build-v569':base/'spacepdhcg-bound-types-v569',
 'build-v576':base/'spacepdhcg-bound-types-v576',
 'validation-v569':base/'spacepdhcg-bound-types-v569/repo/build/performance/bound-types-v569',
 'validation-v576':base/'spacepdhcg-bound-types-v576/repo/build/performance/bound-types-v576',
 'native-v574':base/'spacepdhcg-bound-types-v569/repo/build/performance/bound-types-native-v574',
 'campaign-v572':base/'spacepdhcg-bound-types-campaign-v572',
}
for label,root in sources.items():
 r=json.loads((root/'report.json').read_text())
 assert r['complete'] and not r.get('error'),(label,r)
assert '154 passed' in (sources['validation-v569']/'pytest.log').read_text()
assert '77 passed' in (sources['validation-v576']/'pytest.log').read_text()
for tool in ['memcheck','synccheck','racecheck']:
 log=(sources['validation-v569']/('bounds-'+tool+'.log')).read_text()
 assert 'ERROR SUMMARY: 0 errors' in log or 'RACECHECK SUMMARY: 0 hazards' in log,tool
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
files['native-v574/probe']=sources['validation-v569']/'native-conversion-test'
files['native-v574/run.py']=base/'spacepdhcg-bound-types-v569/repo/build/performance/run_bound_types_native_v574.py'
manifest={name:hashlib.sha256(path.read_bytes()).hexdigest() for name,path in files.items()}
output=base/'spacepdhcg-bound-types-results-v577.tar.gz'
assert not output.exists()
meta=base/'spacepdhcg-bound-types-results-v577-manifest.json';meta.write_text(json.dumps(manifest,indent=2))
with tarfile.open(output,'w:gz') as t:
 for name,path in files.items():t.add(path,arcname=name,recursive=False)
 t.add(meta,arcname='archive-manifest.json',recursive=False)
print(json.dumps(dict(path=str(output),sha256=hashlib.sha256(output.read_bytes()).hexdigest(),bytes=output.stat().st_size,files=len(files))))
