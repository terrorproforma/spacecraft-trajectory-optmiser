from pathlib import Path
import os,subprocess,json,fcntl,time,sys
base='/home/angus/build-spacepdhcg-regularization-v205/final'
env={k:v for k,v in os.environ.items() if not k.startswith('SPACEPDHCG_TEST_')}
env.update(PYTHONPATH='src',SPACEPDHCG_GTOC12_CUDA_LIBRARY=base+'/libspacepdhcg_cuda.so',SPACEPDHCG_GTOC12_GPU_TESTS='1',SPACEPDHCG_GTOC12_DATA='/home/angus/worktrees/spacepdhcg-release/benchmarks/gtoc12/data',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',SPACEPDHCG_QOCO_LIBRARY='/home/angus/build-qoco-gpu-device-ir-v137/final/libqoco.so',LD_LIBRARY_PATH=base+':/home/angus/cudss-isolated-0.8.0.10/nvidia/cu12/lib:/usr/local/cuda-12.8/lib64')
for flag in ['GTOC12_STATE_ORIGIN','GTOC12_DEFERRED_REPORTS','GTOC12_DEVICE_SCHEDULING','GTOC12_DEVICE_REFRESH','GTOC12_DEVICE_ASSEMBLY_VALIDATION','QOCO_DEVICE_VALIDATION','QOCO_NATIVE_NUMERIC_REPLAY','GTOC12_DEVICE_QUALIFICATION','QOCO_NATIVE_REPLAY','QOCO_IPM_GRAPH']:env['SPACEPDHCG_TEST_'+flag]='1'
venv='/home/angus/worktrees/spacepdhcg-literature-venv/bin/'
commands=[('regression',[venv+'python','-m','pytest',*[str(p) for p in sorted(Path('tests').glob('test_gtoc12*.py'))],'-q','--tb=short'])]
checks=[]
with Path('/home/angus/.spacepdhcg-gpu.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 for name,cmd in commands:
  start=time.monotonic();r=subprocess.run(cmd,env=env,capture_output=True,text=True,timeout=900)
  checks.append(dict(name=name,command=cmd,returncode=r.returncode,stdout=r.stdout,stderr=r.stderr,seconds=time.monotonic()-start))
  Path('build/performance/regularization-v206-regression.json').write_text(json.dumps(checks,indent=2)+'\n')
  print(name,r.returncode,(r.stdout+r.stderr)[-3000:],flush=True)
  if r.returncode:sys.exit(r.returncode)
