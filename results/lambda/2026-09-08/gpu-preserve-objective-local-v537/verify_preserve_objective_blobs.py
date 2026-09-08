from pathlib import Path
import hashlib,io,json,subprocess,tarfile
def blob(path):return subprocess.check_output(['git','show',':'+path.as_posix()])
for tag in ['preserve-objective-local-v537','preserve-objective-v535','preserve-campaign-v538']:
 root=Path('results/lambda/2026-09-08/gpu-'+tag)
 manifest=json.loads(blob(root/'files-sha256.json'))
 for name,sha in manifest.items():assert hashlib.sha256(blob(root/name)).hexdigest()==sha,(tag,name)
 if (root/'archive-manifests.json').exists():archives=json.loads(blob(root/'archive-manifests.json'))
 else:archives={'raw.tar.gz':json.loads(blob(root/'archive-manifest.json'))}
 for name,members in archives.items():
  with tarfile.open(fileobj=io.BytesIO(blob(root/name))) as archive:
   actual={m.name:hashlib.sha256(archive.extractfile(m).read()).hexdigest() for m in archive.getmembers() if m.isfile()}
  if name=='raw.tar.gz':actual.pop('files-sha256.json')
  assert actual==members,(tag,name)
 print(tag,len(manifest),'files and archive members verified')
check=json.loads(Path('build/performance/preserve-preparation-verification.json').read_text())
path=Path('scripts/gpu/prepare_qoco_preserve_objective.py')
assert hashlib.sha256(path.read_bytes()).hexdigest()==check['final_preparer_sha256'] and check['compiled_outputs_match']
tested=json.loads(Path('build/performance/preserve-objective-v536/report.json').read_text())['source_sha256']
for path in [Path('scripts/gpu/prepare_qoco_gpu.py'),Path('cpp/cuda/tests/qoco_gpu_numeric_update_test.cu')]:
 assert hashlib.sha256(path.read_bytes()).hexdigest()==tested[path.as_posix()]
for path in [Path('scripts/gpu/prepare_qoco_preserve_objective.py'),Path('scripts/gpu/prepare_qoco_gpu.py'),Path('cpp/cuda/tests/qoco_gpu_numeric_update_test.cu')]:
 assert blob(path).replace(b'\r\n',b'\n')==path.read_bytes().replace(b'\r\n',b'\n')
print('Final staged sources match tested native code and verified preparation')
