from pathlib import Path
import hashlib,json,tarfile
base=Path('/home/ubuntu')
experiment=base/'spacepdhcg-fast-root-v584';final=base/'spacepdhcg-fast-root-v588'
for root in (experiment,final):
 r=json.loads((root/'report.json').read_text());assert r['complete'] and not r.get('error'),r
groups={
 'validation-experiment':experiment/'repo/build/performance/fast-root-v584',
 'campaign-experiment':experiment/'repo/build/performance/fast-root-campaign-v584',
 'microbenchmark':experiment/'repo/build/performance/fast-root-v586',
 'validation-final':final/'repo/build/performance/fast-root-v588',
 'campaign-final':final/'repo/build/performance/fast-root-campaign-v588',
}
files={}
for prefix,root in groups.items():
 r=json.loads((root/'report.json').read_text());assert r['complete'] and not r.get('error'),prefix
 for path in sorted(root.rglob('*')):
  if path.is_file() and '__pycache__' not in path.parts:files[prefix+'/'+path.relative_to(root).as_posix()]=path
for prefix,root in [('experiment',experiment),('final',final)]:
 for name in ['run.py','report.json','source-sha256.json','configure.log','build.log','validation.log','campaign.log','core-build/cuda/libspacepdhcg_cuda.so']:
  files[prefix+'/'+name]=root/name
 for name in ['cpp/cuda/src/orbitweaver_gpu.cu','tests/test_gtoc12_gpu_fast_lambert_root.py','cpp/cuda/tests/orbitweaver_hop_test.cu','build/performance/solver_phase_details.py']:
  files[prefix+'/source/'+name]=root/'repo'/name
 for path in (root/'repo/build/performance').glob('*fast_root*.py'):
  files[prefix+'/workers/'+path.name]=path
files['qoco/libqoco.so']=base/'spacepdhcg-scaled-pool-v545/final/libqoco.so'
manifest={name:hashlib.sha256(path.read_bytes()).hexdigest() for name,path in files.items()}
meta=base/'fast-root-manifest-v589.json';meta.write_text(json.dumps(manifest,indent=2))
output=base/'fast-root-results-v589.tar.gz';assert not output.exists()
with tarfile.open(output,'w:gz') as t:
 for name,path in files.items():t.add(path,arcname=name,recursive=False)
 t.add(meta,arcname='archive-manifest.json',recursive=False)
print(json.dumps(dict(path=str(output),sha256=hashlib.sha256(output.read_bytes()).hexdigest(),bytes=output.stat().st_size)))
