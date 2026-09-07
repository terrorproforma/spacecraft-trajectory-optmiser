from pathlib import Path
import os,sys,subprocess,json,fcntl
env=os.environ.copy();env.update(PYTHONPATH=str(Path('src').resolve()),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',SPACEPDHCG_GTOC12_GPU_TESTS='1',SPACEPDHCG_GTOC12_CUDA_LIBRARY='/home/angus/build-spacepdhcg-gtoc12-v141/cuda/libspacepdhcg_cuda.so',SPACEPDHCG_GTOC12_DATA='/home/angus/worktrees/spacepdhcg-release/benchmarks/gtoc12/data')
python='/home/angus/worktrees/spacepdhcg-literature-venv/bin/python'
out=Path('build/performance/driver-retime-v255')
with (out/'benchmark.log').open('x') as log:r=subprocess.run([python,'build/performance/benchmark_driver_v253.py',str(out/'driver-comparison'),'results/lambda/2026-09-08/gpu-sweeps-v234/input-refinements.json','results/lambda/2026-09-08/gpu-sweeps-v234/input-return.json'],env=env,stdout=log,stderr=subprocess.STDOUT)
print((out/'benchmark.log').read_text());r.check_returncode()
tests=json.loads(Path('build/performance/forward-final-v251/tests.json').read_text())
r=subprocess.run(['python3','build/performance/test_driver_v253.py',str(out/'tests.log'),'tests/test_gtoc12_retime_driver.py',*tests],env=env)
sys.exit(r.returncode)
