from pathlib import Path
import os,subprocess,json,time,hashlib,shutil,fcntl
root=Path('build/performance/recovery-v380');root.mkdir(exist_ok=False)
frozen=Path('/home/angus/build-spacepdhcg-recovery-v380/final');frozen.mkdir(parents=True,exist_ok=False)
core=frozen/'libspacepdhcg_cuda.so';shutil.copy2('/home/angus/build-spacepdhcg-gtoc12-v141/cuda/libspacepdhcg_cuda.so',core)
qoco=Path('/home/angus/build-qoco-soc-step-v358/final/libqoco.so')
names=['cpp/cuda/src/gtoc12_collect_dp.cu','cpp/cuda/include/spacepdhcg/cuda/gtoc12_collect_dp_c_api.h','src/spacepdhcg/gtoc12/collectdp.py','src/spacepdhcg/gtoc12/gpu_collect_tables.py','src/spacepdhcg/gtoc12/refinement_queue.py','src/spacepdhcg/gtoc12/cli.py','src/spacepdhcg/gtoc12/lambert.py','tests/test_gtoc12_gpu_harvest_window.py','tests/test_gtoc12_refinement_queue.py','tests/test_gtoc12_run_refinement.py']
report=dict(pid=os.getpid(),complete=False,stages=[],source_sha256={n:hashlib.sha256(Path(n).read_bytes()).hexdigest() for n in names},runtime_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [core,qoco]})
for n in names:
 target=root/'source'/n;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(n,target)
def save():(root/'report.json').write_text(json.dumps(report,indent=2))
save()
env={k:v for k,v in os.environ.items() if not k.startswith(('SPACEPDHCG_TEST_','QOCO_REPLAY_'))}
env.update(PYTHONPATH=str(Path('src').resolve()),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',SPACEPDHCG_GTOC12_GPU_TESTS='1',SPACEPDHCG_GTOC12_CUDA_LIBRARY=str(core),SPACEPDHCG_QOCO_LIBRARY=str(qoco),SPACEPDHCG_GTOC12_DATA='/home/angus/worktrees/spacepdhcg-release/benchmarks/gtoc12/data',LD_LIBRARY_PATH=str(core.parent)+':'+str(qoco.parent)+':/home/angus/cudss-isolated-0.8.0.10/nvidia/cu12/lib:/usr/local/cuda-12.8/lib64')
py='/home/angus/worktrees/spacepdhcg-literature-venv/bin/python'
boot="import sys;sys.meta_path=[f for f in sys.meta_path if f.__class__.__module__!='_editable_skbc_spacepdhcg'];import pytest;sys.exit(pytest.main(sys.argv[1:]))"
def run(name,cmd,timeout):
 start=time.perf_counter()
 with (root/(name+'.log')).open('x') as log:
  child=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT);report.update(stage=name,child_pid=child.pid);save()
  try:code=child.wait(timeout=timeout)
  except subprocess.TimeoutExpired:child.kill();child.wait();raise
 report['stages'].append(dict(name=name,code=code,seconds=time.perf_counter()-start,command=cmd));save()
 assert code==0,(name,code)
try:
 with open('/home/angus/.spacepdhcg-gpu.lock','a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  tests=['tests/test_gtoc12_refinement_queue.py','tests/test_gtoc12_run_refinement.py','tests/test_gtoc12_gpu_harvest_window.py','tests/test_gtoc12_gpu_resident_collect_tables.py','tests/test_gtoc12_gpu_collect_tables.py','tests/test_gtoc12_gpu_collect_dp.py','tests/test_gtoc12_gpu_scvx.py','tests/test_gtoc12_gpu_cli.py']
  run('pytest',[py,'-c',boot,*tests,'-q'],300)
  run('memcheck',['/usr/local/cuda-12.8/bin/compute-sanitizer','--tool','memcheck','--error-exitcode','99',py,'-c',boot,tests[0],'-q'],120)
  cli="import sys,runpy;sys.meta_path=[f for f in sys.meta_path if f.__class__.__module__!='_editable_skbc_spacepdhcg'];runpy.run_module('spacepdhcg',run_name='__main__')"
  cmd=[py,'-c',cli,'gtoc12','run','--run-id','recovery380','--output',str(root/'output'),'--full-catalogue','--ships','4','--beam-width','32','--max-deploys','10','--neighbours','48','--refine-top','5','--refine-recovery','16','--search-budget-seconds','120','--budget-seconds','900','--retime-attempts','8','--retime-budget-seconds','300','--no-cooperative','--node-days','2','--scvx-iterations','40','--screening-backend','cuda','--seed-backend','cuda','--discretisation-backend','cuda','--assembly-backend','cuda','--convex-solver','qoco','--outer-loop-backend','cuda','--gpu-execution','graph']
  run('fleet',cmd,1200)
 report['complete']=True
except Exception as e:report['error']=repr(e)
save();print(json.dumps({k:v for k,v in report.items() if k not in ['source_sha256','stages']},indent=2))
