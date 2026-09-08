from pathlib import Path
import subprocess,os,json,time,fcntl
source=Path('/home/angus/build-qoco-nonfinite-ir-v336/source');lib=source.parent/'final'
root=Path('build/performance/nonfinite-ir-v336');root.mkdir(exist_ok=False)
exe=lib/'qoco_ir_control_probe'
cmd=['/usr/local/cuda-12.8/bin/nvcc','-O2','-std=c++17','-arch=sm_120','--default-stream','per-thread','cpp/cuda/tests/qoco_ir_control_probe.cu','-Icpp/cuda/patches','-I'+str(source/'algebra/cuda'),'-I'+str(source/'include'),'-I'+str(source/'lib/qdldl/include'),'-I/home/angus/cudss-isolated-0.8.0.10/nvidia/cu12/include','-L'+str(lib),'-lqoco','-ldl','-o',str(exe)]
r=subprocess.run(cmd,capture_output=True,text=True);(root/'build.log').write_text(r.stdout+r.stderr);r.check_returncode()
env={k:v for k,v in os.environ.items() if not k.startswith('SPACEPDHCG_TEST_')}
env.update(LD_LIBRARY_PATH=str(lib)+':/home/angus/cudss-isolated-0.8.0.10/nvidia/cu12/lib:/usr/local/cuda-12.8/lib64',PYTHONPATH='src',SPACEPDHCG_QOCO_LIBRARY=str(lib/'libqoco.so'),SPACEPDHCG_GTOC12_CUDA_LIBRARY='/home/angus/build-spacepdhcg-conic-retry-v313/final/libspacepdhcg_cuda.so',SPACEPDHCG_GTOC12_GPU_TESTS='1',SPACEPDHCG_GTOC12_DATA='/home/angus/worktrees/spacepdhcg-release/benchmarks/gtoc12/data')
checks=[]
with open('/home/angus/.spacepdhcg-gpu.lock','a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 for name,cmd in [('control',[str(exe)])]+[(tool,['/usr/local/cuda-12.8/bin/compute-sanitizer','--tool',tool,'--error-exitcode','99',str(exe)]) for tool in ['memcheck','initcheck','racecheck','synccheck']]+[('pytest',['/home/angus/worktrees/spacepdhcg-literature-venv/bin/python','-m','pytest','tests/test_gtoc12_gpu_scvx.py','tests/test_gtoc12_gpu_cli.py','-q'])]:
  start=time.perf_counter();r=subprocess.run(cmd,env=env,capture_output=True,text=True,timeout=120)
  (root/(name+'.log')).write_text(r.stdout+r.stderr);checks.append(dict(name=name,returncode=r.returncode,seconds=time.perf_counter()-start,command=cmd));(root/'report.json').write_text(json.dumps(dict(checks=checks),indent=2));print(name,r.returncode,(r.stdout+r.stderr)[-650:],flush=True);r.check_returncode()
