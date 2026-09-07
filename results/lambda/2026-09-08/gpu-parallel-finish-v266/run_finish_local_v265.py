from pathlib import Path
import os,subprocess,json,sys
env=os.environ.copy();env.update(PYTHONPATH=str(Path('src').resolve()),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',SPACEPDHCG_GTOC12_GPU_TESTS='1',SPACEPDHCG_GTOC12_CUDA_LIBRARY='/home/angus/build-spacepdhcg-gtoc12-v141/cuda/libspacepdhcg_cuda.so',SPACEPDHCG_GTOC12_DATA='/home/angus/worktrees/spacepdhcg-release/benchmarks/gtoc12/data')
out=Path('build/performance/finish-barriers-v265');python='/home/angus/worktrees/spacepdhcg-literature-venv/bin/python'
with (out/'benchmark.log').open('x') as log:r=subprocess.run([python,'build/performance/benchmark_finish_v259.py',str(out/'comparison'),'results/lambda/2026-09-08/gpu-sweeps-v234/input-refinements.json','results/lambda/2026-09-08/gpu-sweeps-v234/input-return.json','/home/angus/build-spacepdhcg-driver-retime-v255/final/libspacepdhcg_cuda.so',env['SPACEPDHCG_GTOC12_CUDA_LIBRARY']],env=env,stdout=log,stderr=subprocess.STDOUT)
print((out/'benchmark.log').read_text());r.check_returncode()
tests=json.loads(Path('build/performance/forward-final-v251/tests.json').read_text())
r=subprocess.run(['python3','build/performance/test_driver_v253.py',str(out/'tests.log'),'tests/test_gtoc12_retime_finish.py','tests/test_gtoc12_retime_driver.py',*tests],env=env);sys.exit(r.returncode)
