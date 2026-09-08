from pathlib import Path
import os,subprocess,json,hashlib,tarfile
files=['cpp/cuda/src/orbitweaver_gpu.cu','cpp/cuda/src/gtoc12_retime.cu','cpp/cuda/include/spacepdhcg/cuda/orbitweaver_gpu_c_api.h','cpp/cuda/tests/orbitweaver_grid_cache_test.cu','src/spacepdhcg/gtoc12/cli.py','tests/test_gtoc12_run_final_verification.py']
manifest=Path('build/performance/cache-source-sha256-v400.json')
manifest.write_text(json.dumps({name:hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in files},indent=2))
archive=Path('/tmp/grid-cache-v400.tar.gz')
with tarfile.open(archive,'w:gz') as t:
 for name in files:t.add(name,arcname=name)
 t.add(manifest,arcname='cache-source-sha256.json')
key=Path('/tmp/traj-key.pem')
if not key.exists():
 fd=os.open(str(key),os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
 with os.fdopen(fd,'wb') as out:out.write(Path('traj-key.pem').read_bytes())
subprocess.run(['scp','-q','-i',str(key),'-o','BatchMode=yes',str(archive),'ubuntu@192.222.55.229:/tmp/grid-cache-v400.tar.gz'],check=True,timeout=55)
source=Path('build/performance/run_grid_cache_v400.py').read_text()
program="from pathlib import Path\nimport subprocess,json\nroot=Path('/home/ubuntu/spacepdhcg-grid-cache-v400');root.mkdir(exist_ok=False)\n(root/'run.py').write_text("+repr(source)+")\nwith (root/'runner.log').open('x') as log:p=subprocess.Popen(['python3',str(root/'run.py')],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)\nprint(json.dumps(dict(pid=p.pid,root=str(root))))\n"
Path('build/performance/launch_grid_cache_v400.py').write_text(program)
print('Uploaded archive SHA256',hashlib.sha256(archive.read_bytes()).hexdigest())
