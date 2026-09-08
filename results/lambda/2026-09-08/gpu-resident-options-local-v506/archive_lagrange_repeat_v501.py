from pathlib import Path
import json,hashlib,tarfile,shutil
root=Path('/home/ubuntu/spacepdhcg-lagrange-repeat-v501')
r=json.loads((root/'report.json').read_text());assert r['complete'] and not r.get('error')
for name,sha in r.get('source_sha256',{}).items():
 p=root/'repo'/name;assert hashlib.sha256(p.read_bytes()).hexdigest()==sha
 target=root/'source-overlay'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
for c in r['campaigns']:
 f=json.loads((root/c['name']/'output/run_report.json').read_text());assert f['best']['accepted'] and f['best']['official']['ok'] and f['best']['independent']['ok']
basis=Path('/home/ubuntu/spacepdhcg-resident-options-v500/repo')
inputs=['tests/test_gtoc12_gpu_discretisation.py','src/spacepdhcg/gtoc12/verifier.py','src/spacepdhcg/gtoc12/low_thrust.py','src/spacepdhcg/gtoc12/gpu_scvx.py','src/spacepdhcg/gtoc12/constants.py']
for name in inputs:
 target=root/'source-overlay'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(basis/name,target)
(root/'diagnostic-source-sha256.json').write_text(json.dumps({name:hashlib.sha256((basis/name).read_bytes()).hexdigest() for name in inputs},indent=2))
libraries=[Path('/home/ubuntu/spacepdhcg-workspace-pool-v491/core-build/cuda/libspacepdhcg_cuda.so'),Path('/home/ubuntu/spacepdhcg-resident-options-v497/core-build/cuda/libspacepdhcg_cuda.so')]
(root/'diagnostic-runtime-sha256.json').write_text(json.dumps({str(q):hashlib.sha256(q.read_bytes()).hexdigest() for q in libraries},indent=2))
paths=[p for p in root.rglob('*') if p.is_file() and p.relative_to(root).parts[0] not in ['repo','core-build','compact-options-probe'] and p.name not in ['files-sha256.json','resident-options-probe']]
manifest={p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}
(root/'files-sha256.json').write_text(json.dumps(manifest,indent=2))
archive=root.with_suffix('.tar.gz')
with tarfile.open(archive,'w:gz') as t:
 for name in list(manifest)+['files-sha256.json']:t.add(root/name,arcname=name,recursive=False)
print(json.dumps(dict(path=str(archive),sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),bytes=archive.stat().st_size,files=len(manifest))))
