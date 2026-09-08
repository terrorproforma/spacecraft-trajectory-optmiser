from pathlib import Path
import os,subprocess
root=Path('/home/angus/spacepdhcg-grid-probe-v787');root.mkdir()
script=Path('build/performance/probe_grid_v787.py').read_text()
(root/'run.py').write_text(script)
launch='''from pathlib import Path
import os,subprocess
home=Path.home();root=home/'spacepdhcg-grid-probe-v787';remote=home.name=='ubuntu'
core=home/'spacepdhcg-grid-v786/final/libspacepdhcg_cuda.so';cuda='/usr/local/cuda' if remote else '/usr/local/cuda-12.8'
env={k:v for k,v in os.environ.items() if not k.startswith(('SPACEPDHCG_TEST_','QOCO_REPLAY_'))}
env.update(SPACEPDHCG_GTOC12_CUDA_LIBRARY=str(core),SPACEPDHCG_GTOC12_DATA=str(home/'spacepdhcg/gtoc12/benchmarks/gtoc12/data') if remote else '/home/angus/worktrees/spacepdhcg-release/benchmarks/gtoc12/data',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',LD_LIBRARY_PATH=cuda+'/lib64')
py=str(home/'spacepdhcg/v1/.venv/bin/python') if remote else '/home/angus/worktrees/spacepdhcg-literature-venv/bin/python'
with (root/'worker.log').open('x') as log:print(subprocess.Popen([py,str(root/'run.py')],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True).pid)
'''
(root/'launch.py').write_text(launch)
key=Path('/tmp/traj-key.pem')
if not key.exists():
    fd=os.open(key,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
    with os.fdopen(fd,'wb') as out:out.write(Path('traj-key.pem').read_bytes())
code="from pathlib import Path\nroot=Path.home()/'spacepdhcg-grid-probe-v787';root.mkdir()\n(root/'run.py').write_text("+repr(script)+")\n(root/'launch.py').write_text("+repr(launch)+")\nexec((root/'launch.py').read_text())"
r=subprocess.run(['ssh','-i',str(key),'-o','BatchMode=yes','ubuntu@192.222.55.229','python3 -'],input=code,text=True,capture_output=True,timeout=40);print(r.stdout,r.stderr);r.check_returncode()
exec(launch)
