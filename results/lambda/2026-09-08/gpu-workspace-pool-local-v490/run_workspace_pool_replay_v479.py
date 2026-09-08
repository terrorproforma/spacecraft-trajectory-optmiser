from pathlib import Path
import subprocess,os,fcntl,json,time,hashlib,sys
root=Path('build/performance/workspace-pool-replay-v479');root.mkdir(exist_ok=False)
core='/home/angus/build-spacepdhcg-workspace-pool-v476/final/libspacepdhcg_cuda.so';qoco='/home/angus/build-qoco-soc-step-v358/final/libqoco.so'
env={k:v for k,v in os.environ.items() if not k.startswith(('SPACEPDHCG_TEST_','QOCO_REPLAY_'))}
env.update(PYTHONPATH=str(Path('src').resolve()),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',SPACEPDHCG_GTOC12_CUDA_LIBRARY=core,SPACEPDHCG_QOCO_LIBRARY=qoco,SPACEPDHCG_GTOC12_DATA='/home/angus/worktrees/spacepdhcg-release/benchmarks/gtoc12/data',LD_LIBRARY_PATH=str(Path(core).parent)+':'+str(Path(qoco).parent)+':/home/angus/cudss-isolated-0.8.0.10/nvidia/cu12/lib:/usr/local/cuda-12.8/lib64')
boot="import sys,runpy;sys.meta_path=[f for f in sys.meta_path if f.__class__.__module__!='_editable_skbc_spacepdhcg'];runpy.run_path('build/performance/replay_workspace_pool.py',run_name='__main__')"
report=dict(pid=os.getpid(),complete=False,stages=[],core_sha256=hashlib.sha256(Path(core).read_bytes()).hexdigest(),input_sha256=hashlib.sha256(Path('build/performance/grid-cache-fleet-v403/scvx-calls.json').read_bytes()).hexdigest())
def save():(root/'report.json').write_text(json.dumps(report,indent=2))
save()
try:
 with open('/home/angus/.spacepdhcg-gpu.lock','a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  for label,mode,indices in [('baseline','0',None),('candidate','1',None)]:
   cmd=['/home/angus/worktrees/spacepdhcg-literature-venv/bin/python','-c',boot,mode,str(root/label)]+([indices] if indices else [])
   with (root/f'{label}.log').open('x') as log:
    start=time.perf_counter();child=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT);report.update(stage=label,child_pid=child.pid);save()
    code=child.wait(timeout=1200)
   report['stages'].append(dict(name=label,returncode=code,seconds=time.perf_counter()-start,command=cmd));save();assert code==0
 report['complete']=True
except Exception as e:report['error']=repr(e)
save();print(json.dumps(report,indent=2))
