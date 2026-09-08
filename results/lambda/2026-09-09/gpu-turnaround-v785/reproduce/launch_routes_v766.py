from pathlib import Path
import hashlib,json,os,shutil,subprocess,tarfile
root=Path.home()/'spacepdhcg-fleet-routes-v766';(root/'input').mkdir(parents=True,exist_ok=False)
for source,name in [('results/lambda/2026-09-09/gpu-fleet-exchange-v733/h100-best/campaign-report.json','campaign.json'),('results/lambda/2026-09-09/gpu-fleet-exchange-v733/h100-best/Result.txt','incumbent.txt'),('results/lambda/2026-09-09/gpu-fleet-tree-v765/pool.json','pool.json'),('results/gtoc12/hop_inflation_fit.json','fit.json')]:shutil.copyfile(source,root/'input'/name)
shutil.copyfile('build/performance/search_fleet_routes_v766.py',root/'run.py')
(root/'input-hashes.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (root/'input').iterdir()},indent=2))
launch='''from pathlib import Path
import os,subprocess
home=Path.home();root=home/'spacepdhcg-fleet-routes-v766';remote=home.name=='ubuntu'
qoco=home/('spacepdhcg-retry-conditioning-v686/final/libqoco.so' if remote else 'spacepdhcg-retry-conditioning-v683/final/libqoco.so')
core=home/'spacepdhcg-fleet-v763/final/libspacepdhcg_cuda.so';cuda='/usr/local/cuda' if remote else '/usr/local/cuda-12.8'
cudss=str(home/'spacepdhcg-recovery-v152/cudss/lib') if remote else '/home/angus/cudss-isolated-0.8.0.10/nvidia/cu12/lib'
env={k:v for k,v in os.environ.items() if not k.startswith(('SPACEPDHCG_TEST_','QOCO_REPLAY_'))}
env.update(SPACEPDHCG_GTOC12_CUDA_LIBRARY=str(core),SPACEPDHCG_QOCO_LIBRARY=str(qoco),SPACEPDHCG_GTOC12_DATA=str(home/'spacepdhcg/gtoc12/benchmarks/gtoc12/data') if remote else '/home/angus/worktrees/spacepdhcg-release/benchmarks/gtoc12/data',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',LD_LIBRARY_PATH=str(qoco.parent)+':'+str(core.parent)+':'+cudss+':'+cuda+'/lib64')
for name in ('CONDITIONING_RETRY','JOINT_BATCH','JOINT_DEVICE_SELECTION','JOINT_RESIDENT_GEOMETRY','JOINT_DEVICE_MESH','JOINT_DEVICE_SEARCH'):env['SPACEPDHCG_TEST_GTOC12_'+name]='1'
py=str(home/'spacepdhcg/v1/.venv/bin/python') if remote else '/home/angus/worktrees/spacepdhcg-literature-venv/bin/python'
with (root/'worker.log').open('x') as log:print(subprocess.Popen([py,str(root/'run.py')],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True).pid)
'''
(root/'launch.py').write_text(launch)
with tarfile.open(root/'input.tar.gz','w:gz') as tar:
    for p in [*(root/'input').iterdir(),root/'run.py',root/'launch.py',root/'input-hashes.json']:tar.add(p,arcname=p.relative_to(root).as_posix())
key=Path('/tmp/traj-key.pem')
if not key.exists():
    fd=os.open(key,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
    with os.fdopen(fd,'wb') as out:out.write(Path('traj-key.pem').read_bytes())
subprocess.run(['scp','-q','-i',str(key),'-o','BatchMode=yes',str(root/'input.tar.gz'),'ubuntu@192.222.55.229:/home/ubuntu/fleet-routes-input-v766.tar.gz'],check=True,timeout=40)
code="from pathlib import Path\nimport tarfile\nroot=Path.home()/'spacepdhcg-fleet-routes-v766';root.mkdir()\nwith tarfile.open(Path.home()/'fleet-routes-input-v766.tar.gz') as tar:tar.extractall(root,filter='data')\nexec((root/'launch.py').read_text())\n"
r=subprocess.run(['ssh','-i',str(key),'-o','BatchMode=yes','ubuntu@192.222.55.229','python3 -'],input=code,text=True,capture_output=True,timeout=30);print(r.stdout,r.stderr);r.check_returncode()
exec(launch)
