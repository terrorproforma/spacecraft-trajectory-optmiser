from pathlib import Path
import hashlib,json,tarfile
base=Path('/home/ubuntu');files={}
for version,kind in [(605,'split'),(613,'split'),(617,'parallel')]:
 root=base/f'spacepdhcg-{kind}-directions-v{version}'
 report=json.loads((root/'report.json').read_text());assert report['complete'] and not report.get('error'),report
 for name in ['report.json','source-sha256.json','run.py','configure.log','build.log','validation.log','core-build/cuda/libspacepdhcg_cuda.so']:
  files[f'build-v{version}/'+name]=root/name
 for name in ['cpp/cuda/src/orbitweaver_gpu.cu','tests/test_gtoc12_gpu_split_directions.py','tests/test_gtoc12_gpu_fast_lambert_root.py']:
  files[f'build-v{version}/source/'+name]=root/'repo'/name
 work=root/'repo/build/performance'
 for path in work.glob('*directions*.py'):files[f'build-v{version}/workers/'+path.name]=path
 for path in work.glob('*parallel_campaign*.py'):files[f'build-v{version}/workers/'+path.name]=path
 for prefix in [f'{kind}-directions-v{version}']+(['parallel-directions-campaign-v617'] if version==617 else []):
  folder=work/prefix
  r=json.loads((folder/'report.json').read_text());assert r['complete'] and not r.get('error'),r
  for path in sorted(folder.rglob('*')):
   if path.is_file() and '__pycache__' not in path.parts:files[prefix+'/'+path.relative_to(folder).as_posix()]=path
files['qoco/libqoco.so']=base/'spacepdhcg-scaled-pool-v545/final/libqoco.so'
manifest={name:hashlib.sha256(path.read_bytes()).hexdigest() for name,path in files.items()}
meta=base/'parallel-manifest-v622.json';meta.write_text(json.dumps(manifest,indent=2))
output=base/'parallel-results-v622.tar.gz';assert not output.exists()
with tarfile.open(output,'w:gz') as t:
 for name,path in files.items():t.add(path,arcname=name,recursive=False)
 t.add(meta,arcname='archive-manifest.json',recursive=False)
print(json.dumps(dict(path=str(output),bytes=output.stat().st_size,sha256=hashlib.sha256(output.read_bytes()).hexdigest())))
