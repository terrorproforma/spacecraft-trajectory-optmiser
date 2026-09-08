from pathlib import Path
import hashlib,io,json,subprocess,tarfile
def blob(path):return subprocess.check_output(['git','show',':'+path.as_posix()])
root=Path('results/lambda/2026-09-08/gpu-scaled-workspace-v553')
manifest=json.loads(blob(root/'files-sha256.json'))
for name,sha in manifest.items():assert hashlib.sha256(blob(root/name)).hexdigest()==sha,name
for prefix in ['local','lambda']:
 expected=json.loads(blob(root/(prefix+'-archive-manifest.json')))
 with tarfile.open(fileobj=io.BytesIO(blob(root/(prefix+'-raw.tar.gz')))) as archive:
  actual={m.name:hashlib.sha256(archive.extractfile(m).read()).hexdigest() for m in archive.getmembers() if m.isfile()}
 if prefix=='lambda':actual.pop('archive-manifest.json')
 assert actual==expected,prefix
 print(prefix,len(actual),'raw files verified')
tested=json.loads(Path('build/performance/scaled-pool-v542/report.json').read_text())['source_sha256']
remote=json.loads(Path('build/performance/retrieved-scaled-v553/validation-v545/report.json').read_text())['source_sha256']
for name in ['cpp/cuda/src/gtoc12_qoco.cu','cpp/cuda/src/native_qoco_adapter.cpp','scripts/gpu/prepare_qoco_preserve_objective.py','tests/test_gtoc12_gpu_scaled_workspace_pool.py']:
 path=Path(name);assert hashlib.sha256(path.read_bytes()).hexdigest()==tested[name]==remote[name],name
 assert blob(path).replace(b'\r\n',b'\n')==path.read_bytes().replace(b'\r\n',b'\n'),name
print(len(manifest),'published files verified; staged code matches both tested source snapshots')
