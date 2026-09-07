from pathlib import Path
import os,subprocess,fcntl
env=os.environ.copy();env.update(PYTHONPATH=str(Path('../spacecraft-main-publish-20260906/src').resolve()),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',SPACEPDHCG_GTOC12_CUDA_LIBRARY='/home/angus/build-spacepdhcg-finish-barriers-v265/final/libspacepdhcg_cuda.so',SPACEPDHCG_GTOC12_DATA='/home/angus/worktrees/spacepdhcg-release/benchmarks/gtoc12/data')
lock=open('/home/angus/.spacepdhcg-gpu.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
root=Path('results/lambda/2026-09-08/gpu-collect-profile-v272')
boot="import sys,runpy;sys.meta_path=[f for f in sys.meta_path if f.__class__.__module__!='_editable_skbc_spacepdhcg'];runpy.run_path('scripts/gpu/replay_gtoc12_collection_fixtures.py',run_name='__main__')"
with (root/'local-baseline-replay.log').open('x') as log:
    r=subprocess.run(['/home/angus/worktrees/spacepdhcg-literature-venv/bin/python','-c',boot,str(root/'v272/timing.json'),str(root/'local-baseline-replay.json'),'--reference','archived','--mass-key','rounded'],env=env,stdout=log,stderr=subprocess.STDOUT)
print((root/'local-baseline-replay.log').read_text());r.check_returncode()
