from pathlib import Path
import os,sys,subprocess,fcntl
env=os.environ.copy();env.update(PYTHONPATH=str(Path('src').resolve()),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',SPACEPDHCG_GTOC12_GPU_TESTS='1',SPACEPDHCG_GTOC12_CUDA_LIBRARY='/home/angus/build-spacepdhcg-gtoc12-v141/cuda/libspacepdhcg_cuda.so',SPACEPDHCG_GTOC12_DATA='/home/angus/worktrees/spacepdhcg-release/benchmarks/gtoc12/data')
lock=open('/home/angus/.spacepdhcg-gpu.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
boot="import sys,runpy;sys.meta_path=[f for f in sys.meta_path if f.__class__.__module__!='_editable_skbc_spacepdhcg'];runpy.run_module('pytest',run_name='__main__')"
with Path(sys.argv[1]).open('x') as log:r=subprocess.run(['/home/angus/worktrees/spacepdhcg-literature-venv/bin/python','-c',boot,*sys.argv[2:]],env=env,stdout=log,stderr=subprocess.STDOUT)
print(Path(sys.argv[1]).read_text()[-7000:]);sys.exit(r.returncode)
