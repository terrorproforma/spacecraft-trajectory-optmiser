from pathlib import Path
import os,subprocess,fcntl
env=os.environ.copy();env.update(PYTHONPATH=str(Path('src').resolve()),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',SPACEPDHCG_GTOC12_CUDA_LIBRARY='/home/angus/build-spacepdhcg-gtoc12-v141/cuda/libspacepdhcg_cuda.so',SPACEPDHCG_GTOC12_DATA='/home/angus/worktrees/spacepdhcg-release/benchmarks/gtoc12/data')
lock=open('/home/angus/.spacepdhcg-gpu.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
source=Path('results/lambda/2026-09-08/gpu-collect-profile-v272/v272/timing.json')
root=Path('build/performance/gpu-collect-resident-v296');root.mkdir(exist_ok=True)
boot="import sys,runpy;sys.meta_path=[f for f in sys.meta_path if f.__class__.__module__!='_editable_skbc_spacepdhcg'];runpy.run_path('scripts/gpu/benchmark_gtoc12_collection_dp.py',run_name='__main__')"
with (root/'benchmark-gather.log').open('w') as log:r=subprocess.run(['/home/angus/worktrees/spacepdhcg-literature-venv/bin/python','-c',boot,str(source),str(root/'benchmark-gather.json'),'--resident-tables'],env=env,stdout=log,stderr=subprocess.STDOUT)
print((root/'benchmark-gather.log').read_text());r.check_returncode()


