from pathlib import Path
import tarfile,json,hashlib,os,subprocess
files=['cpp/cuda/src/gtoc12_collect_dp.cu','cpp/cuda/include/spacepdhcg/cuda/gtoc12_collect_dp_c_api.h','src/spacepdhcg/gtoc12/collectdp.py','src/spacepdhcg/gtoc12/gpu_collect_tables.py','tests/test_gtoc12_gpu_harvest_window.py','tests/test_gtoc12_gpu_resident_collect_tables.py','tests/test_gtoc12_gpu_collect_tables.py','tests/test_gtoc12_gpu_collect_dp.py','tests/test_gtoc12_gpu_scvx.py','tests/test_gtoc12_gpu_cli.py','tests/test_gtoc12_collectdp.py']
manifest=Path('build/performance/harvest-window-v376/harvest-source-sha256.json');manifest.write_text(json.dumps({p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in files},indent=2))
archive=Path('/tmp/harvest-window-v378.tar.gz')
with tarfile.open(archive,'w:gz') as t:
 for p in files:t.add(p,arcname=p)
 t.add(manifest,arcname='harvest-source-sha256.json')
key=Path('/tmp/traj-key.pem')
if not key.exists():
 fd=os.open(str(key),os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
 with os.fdopen(fd,'wb') as f:f.write(Path('traj-key.pem').read_bytes())
subprocess.run(['scp','-q','-i',str(key),'-o','BatchMode=yes',str(archive),'ubuntu@192.222.55.229:/tmp/harvest-window-v378.tar.gz'],check=True,timeout=60)
source=Path('build/performance/run_harvest_lambda_v378.py').read_text();compile(source,'run378','exec')
launch="from pathlib import Path\nimport subprocess,json\nroot=Path('/home/ubuntu/spacepdhcg-harvest-window-v378');root.mkdir(exist_ok=False)\n(root/'run.py').write_text("+repr(source)+")\nwith (root/'runner.log').open('x') as log:p=subprocess.Popen(['python3',str(root/'run.py')],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)\nprint(json.dumps(dict(pid=p.pid,root=str(root))))\n"
Path('build/performance/launch_harvest_lambda_v378.py').write_text(launch)
print('Uploaded source bundle',hashlib.sha256(archive.read_bytes()).hexdigest())
