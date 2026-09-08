from pathlib import Path
import hashlib,io,json,subprocess,tarfile
def blob(path):return subprocess.check_output(['git','show',':'+path.as_posix()])
root=Path('results/lambda/2026-09-08/gpu-bound-types-v577')
manifest=json.loads(blob(root/'files-sha256.json'))
for name,sha in manifest.items():assert hashlib.sha256(blob(root/name)).hexdigest()==sha,name
for prefix in ['local','lambda']:
 expected=json.loads(blob(root/(prefix+'-archive-manifest.json')))
 with tarfile.open(fileobj=io.BytesIO(blob(root/(prefix+'-raw.tar.gz')))) as archive:
  actual={m.name:hashlib.sha256(archive.extractfile(m).read()).hexdigest() for m in archive.getmembers() if m.isfile()}
 if prefix=='lambda':actual.pop('archive-manifest.json')
 assert actual==expected,prefix
 print(prefix,len(actual),'raw files verified')
tested=json.loads(Path('build/performance/bound-types-v575/report.json').read_text())['source_sha256']
remote=json.loads(Path('build/performance/retrieved-bound-types-v577/validation-v576/report.json').read_text())['source_sha256']
for name in ['cpp/cuda/internal/native_qoco_gpu.h','cpp/cuda/src/native_qoco_gpu.cu','cpp/cuda/src/native_qoco_adapter.cpp','cpp/cuda/tests/native_qoco_conversion_test.cu','cpp/cuda/tests/qoco_gpu_bound_types_test.cu','tests/test_gtoc12_gpu_bound_types.py']:
 path=Path(name);assert hashlib.sha256(path.read_bytes()).hexdigest()==tested[name]==remote[name],name
 assert blob(path).replace(b'\r\n',b'\n')==path.read_bytes().replace(b'\r\n',b'\n'),name
print(len(manifest),'published files verified; staged code matches both tested source snapshots')
