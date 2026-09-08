from pathlib import Path
import os,subprocess
home=Path.home();root=home/'spacepdhcg-fleet-routes-v784';remote=home.name=='ubuntu'
qoco=home/('spacepdhcg-retry-conditioning-v686/final/libqoco.so' if remote else 'spacepdhcg-retry-conditioning-v683/final/libqoco.so')
core=home/'spacepdhcg-turnaround-v782/final/libspacepdhcg_cuda.so';cuda='/usr/local/cuda' if remote else '/usr/local/cuda-12.8'
cudss=str(home/'spacepdhcg-recovery-v152/cudss/lib') if remote else '/home/angus/cudss-isolated-0.8.0.10/nvidia/cu12/lib'
env={k:v for k,v in os.environ.items() if not k.startswith(('SPACEPDHCG_TEST_','QOCO_REPLAY_'))}
env.update(SPACEPDHCG_GTOC12_CUDA_LIBRARY=str(core),SPACEPDHCG_QOCO_LIBRARY=str(qoco),SPACEPDHCG_GTOC12_DATA=str(home/'spacepdhcg/gtoc12/benchmarks/gtoc12/data') if remote else '/home/angus/worktrees/spacepdhcg-release/benchmarks/gtoc12/data',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',LD_LIBRARY_PATH=str(qoco.parent)+':'+str(core.parent)+':'+cudss+':'+cuda+'/lib64')
for name in ('CONDITIONING_RETRY','JOINT_BATCH','JOINT_DEVICE_SELECTION','JOINT_RESIDENT_GEOMETRY','JOINT_DEVICE_MESH','JOINT_DEVICE_SEARCH','JOINT_DEVICE_INSERTIONS','JOINT_DEVICE_LAYOUTS'):env['SPACEPDHCG_TEST_GTOC12_'+name]='1'
py=str(home/'spacepdhcg/v1/.venv/bin/python') if remote else '/home/angus/worktrees/spacepdhcg-literature-venv/bin/python'
with (root/'worker.log').open('x') as log:print(subprocess.Popen([py,str(root/'run.py')],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True).pid)
