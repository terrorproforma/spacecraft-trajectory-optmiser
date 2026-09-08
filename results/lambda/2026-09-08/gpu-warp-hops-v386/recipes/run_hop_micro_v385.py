from pathlib import Path
import subprocess,os,fcntl,shutil,json,time,hashlib
root=Path('build/performance/warp-hops-v385');frozen=Path('/home/angus/build-spacepdhcg-warp-hops-v385/final');frozen.mkdir(parents=True,exist_ok=False)
core=frozen/'libspacepdhcg_cuda.so';shutil.copy2('/home/angus/build-spacepdhcg-gtoc12-v141/cuda/libspacepdhcg_cuda.so',core)
baseline=Path('/home/angus/build-spacepdhcg-recovery-v380/final/libspacepdhcg_cuda.so');qoco=Path('/home/angus/build-qoco-soc-step-v358/final/libqoco.so')
env={k:v for k,v in os.environ.items() if not k.startswith(('SPACEPDHCG_TEST_','QOCO_REPLAY_'))}
env.update(PYTHONPATH=str(Path('src').resolve()),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',SPACEPDHCG_GTOC12_GPU_TESTS='1',SPACEPDHCG_QOCO_LIBRARY=str(qoco),SPACEPDHCG_GTOC12_DATA='/home/angus/worktrees/spacepdhcg-release/benchmarks/gtoc12/data')
py='/home/angus/worktrees/spacepdhcg-literature-venv/bin/python';boot="import sys,runpy;sys.meta_path=[f for f in sys.meta_path if f.__class__.__module__!='_editable_skbc_spacepdhcg'];runpy.run_path('build/performance/hop_micro_v384.py',run_name='__main__')"
report=dict(pid=os.getpid(),complete=False,stages=[],candidate_sha256=hashlib.sha256(core.read_bytes()).hexdigest(),baseline_sha256=hashlib.sha256(baseline.read_bytes()).hexdigest())
def save():(root/'report.json').write_text(json.dumps(report,indent=2))
save()
try:
 with open('/home/angus/.spacepdhcg-gpu.lock','a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  for name,lib in [('baseline0',baseline),('candidate0',core),('candidate1',core),('baseline1',baseline)]:
   selected=dict(env,SPACEPDHCG_GTOC12_CUDA_LIBRARY=str(lib),LD_LIBRARY_PATH=str(lib.parent)+':'+str(qoco.parent)+':/home/angus/cudss-isolated-0.8.0.10/nvidia/cu12/lib:/usr/local/cuda-12.8/lib64')
   start=time.perf_counter()
   with (root/(name+'.log')).open('x') as log:
    child=subprocess.Popen([py,'-c',boot,str(root/name)],env=selected,stdout=log,stderr=subprocess.STDOUT);report.update(stage=name,child_pid=child.pid);save()
    try:code=child.wait(timeout=300)
    except subprocess.TimeoutExpired:child.kill();child.wait();raise
   report['stages'].append(dict(name=name,code=code,seconds=time.perf_counter()-start));save();assert code==0,(name,code)
 report['complete']=True
except Exception as e:report['error']=repr(e)
save();print(json.dumps(report,indent=2))
