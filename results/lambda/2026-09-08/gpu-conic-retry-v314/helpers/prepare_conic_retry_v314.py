from pathlib import Path
import tarfile,json,hashlib,os,subprocess
files=['cpp/cuda/src/gtoc12_scvx.cu','cpp/cuda/src/native_qoco_adapter.cpp','cpp/cuda/tests/gtoc12_scvx_test.cu','cpp/cuda/include/spacepdhcg/cuda/gtoc12_scvx_c_api.h','src/spacepdhcg/gtoc12/gpu_scvx.py']
manifest={p:hashlib.sha256(Path(p).read_bytes().replace(b'\r\n',b'\n')).hexdigest() for p in files}
Path('build/performance/conic-retry-v313/source-sha256.json').write_text(json.dumps(manifest,indent=2))
with tarfile.open('/tmp/conic-retry-v314.tar.gz','w:gz') as t:
 for p in files:t.add(p,arcname=p)
 t.add('build/performance/conic-retry-v313/source-sha256.json',arcname='retry-source-sha256.json')
key=Path('/tmp/traj-key.pem')
if not key.exists():
 fd=os.open(str(key),os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
 with os.fdopen(fd,'wb') as f:f.write(Path('traj-key.pem').read_bytes())
subprocess.run(['scp','-q','-i',str(key),'-o','BatchMode=yes','/tmp/conic-retry-v314.tar.gz','ubuntu@192.222.55.229:/tmp/conic-retry-v314.tar.gz'],check=True,timeout=45)
print('Uploaded',manifest)
