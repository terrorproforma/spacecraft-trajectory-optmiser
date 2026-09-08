from pathlib import Path
import os,subprocess
home=Path.home();root=home/'spacepdhcg-wide-insertions-v781';remote=home.name=='ubuntu'
core=home/'spacepdhcg-layouts-v778/final/libspacepdhcg_cuda.so';cuda='/usr/local/cuda' if remote else '/usr/local/cuda-12.8'
env={k:v for k,v in os.environ.items() if not k.startswith(('SPACEPDHCG_TEST_','QOCO_REPLAY_'))}
env.update(SPACEPDHCG_GTOC12_CUDA_LIBRARY=str(core),SPACEPDHCG_GTOC12_DATA=str(home/'spacepdhcg/gtoc12/benchmarks/gtoc12/data') if remote else '/home/angus/worktrees/spacepdhcg-release/benchmarks/gtoc12/data',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',LD_LIBRARY_PATH=cuda+'/lib64')
py=str(home/'spacepdhcg/v1/.venv/bin/python') if remote else '/home/angus/worktrees/spacepdhcg-literature-venv/bin/python'
with (root/'worker.log').open('x') as log:print(subprocess.Popen([py,str(root/'run.py')],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True).pid)
