from pathlib import Path
import os, subprocess, json, time, fcntl
out=Path('build/performance/forward-final-v251/local');out.mkdir(parents=True,exist_ok=True)
env=os.environ.copy()
env.update(PYTHONPATH=str(Path('src').resolve()),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',SPACEPDHCG_GTOC12_GPU_TESTS='1',SPACEPDHCG_GTOC12_CUDA_LIBRARY='/home/angus/build-spacepdhcg-gtoc12-v141/cuda/libspacepdhcg_cuda.so',SPACEPDHCG_GTOC12_DATA='/home/angus/worktrees/spacepdhcg-release/benchmarks/gtoc12/data')
python='/home/angus/worktrees/spacepdhcg-literature-venv/bin/python'
boot="import sys,runpy;sys.meta_path=[f for f in sys.meta_path if f.__class__.__module__!='_editable_skbc_spacepdhcg'];runpy.run_module('pytest',run_name='__main__')"
lock=open('/home/angus/.spacepdhcg-gpu.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
tests=json.loads(Path('build/performance/forward-final-v251/tests.json').read_text())
start=time.time()
with (out/'tests.log').open('x') as log:r=subprocess.run([python,'-c',boot,*tests],env=env,stdout=log,stderr=subprocess.STDOUT)
(out/'report.json').write_text(json.dumps(dict(returncode=r.returncode,seconds=time.time()-start),indent=2));r.check_returncode()
fcntl.flock(lock,fcntl.LOCK_UN)
source='results/lambda/2026-09-08/gpu-sweeps-v234/input-refinements.json'
sweep='results/lambda/2026-09-08/gpu-sweeps-v234/input-return.json'
for name,cmd in [
 ('release-comparison',[python,'build/performance/benchmark_forward_release_v251.py',str(out/'release-comparison'),source,sweep,'src/spacepdhcg/gtoc12/gpu_retime.py','/home/angus/build-spacepdhcg-warp-retime-v245/final/libspacepdhcg_cuda.so',env['SPACEPDHCG_GTOC12_CUDA_LIBRARY']]),
 ('forward-comparison',[python,'build/performance/benchmark_forward_v248.py',str(out/'forward-comparison'),source,sweep])]:
    with (out/(name+'.log')).open('x') as log:r=subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT)
    r.check_returncode()
print((out/'tests.log').read_text());print((out/'forward-comparison.log').read_text())
