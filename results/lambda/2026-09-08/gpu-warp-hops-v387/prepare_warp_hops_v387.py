from pathlib import Path
import tarfile,json,hashlib,os,subprocess
files=['cpp/cuda/src/orbitweaver_gpu.cu','tests/test_gtoc12_gpu_hops.py','tests/test_gtoc12_run_refinement.py','build/performance/hop_micro_v384.py']
manifest=Path('build/performance/warp-hops-v385/warp-source-sha256.json');manifest.write_text(json.dumps({p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in files},indent=2))
archive=Path('/tmp/warp-hops-v387.tar.gz')
with tarfile.open(archive,'w:gz') as t:
 for p in files:t.add(p,arcname=p)
 t.add(manifest,arcname='warp-source-sha256.json')
key=Path('/tmp/traj-key.pem')
if not key.exists():
 fd=os.open(str(key),os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
 with os.fdopen(fd,'wb') as f:f.write(Path('traj-key.pem').read_bytes())
subprocess.run(['scp','-q','-i',str(key),'-o','BatchMode=yes',str(archive),'ubuntu@192.222.55.229:/tmp/warp-hops-v387.tar.gz'],check=True,timeout=60)
source=Path('build/performance/run_warp_hops_v387.py').read_text();compile(source,'run378','exec')
launch="from pathlib import Path\nimport subprocess,json\nroot=Path('/home/ubuntu/spacepdhcg-warp-hops-v387');root.mkdir(exist_ok=False)\n(root/'run.py').write_text("+repr(source)+")\nwith (root/'runner.log').open('x') as log:p=subprocess.Popen(['python3',str(root/'run.py')],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)\nprint(json.dumps(dict(pid=p.pid,root=str(root))))\n"
Path('build/performance/launch_warp_hops_v387.py').write_text(launch)
print('Uploaded source bundle',hashlib.sha256(archive.read_bytes()).hexdigest())
