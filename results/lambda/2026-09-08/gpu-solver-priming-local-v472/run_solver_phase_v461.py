from pathlib import Path
import subprocess,os,fcntl,json,time,hashlib
root=Path('build/performance/solver-phase-v461');root.mkdir(exist_ok=False)
core='/home/angus/build-spacepdhcg-solver-phase-v458/final/libspacepdhcg_cuda.so';qoco='/home/angus/build-qoco-soc-step-v358/final/libqoco.so'
env={k:v for k,v in os.environ.items() if not k.startswith(('SPACEPDHCG_TEST_','QOCO_REPLAY_'))}
env.update(PYTHONPATH=str(Path('src').resolve()),SPACEPDHCG_TEST_GTOC12_FUSED_TABLES='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',SPACEPDHCG_GTOC12_CUDA_LIBRARY=core,SPACEPDHCG_QOCO_LIBRARY=qoco,SPACEPDHCG_GTOC12_DATA='/home/angus/worktrees/spacepdhcg-release/benchmarks/gtoc12/data',LD_LIBRARY_PATH=str(Path(core).parent)+':'+str(Path(qoco).parent)+':/home/angus/cudss-isolated-0.8.0.10/nvidia/cu12/lib:/usr/local/cuda-12.8/lib64')
boot="import sys,runpy;sys.meta_path=[f for f in sys.meta_path if f.__class__.__module__!='_editable_skbc_spacepdhcg'];runpy.run_path('build/performance/solver_phase_details.py',run_name='__main__')"
cmd=['/home/angus/worktrees/spacepdhcg-literature-venv/bin/python','-c',boot,'gtoc12','run','--run-id','solver_phase461','--output',str(root/'output'),'--full-catalogue','--ships','1','--beam-width','16','--max-deploys','10','--neighbours','48','--refine-top','3','--search-budget-seconds','120','--budget-seconds','900','--retime-attempts','4','--retime-budget-seconds','300','--no-cooperative','--node-days','2','--scvx-iterations','40','--screening-backend','cuda','--seed-backend','cuda','--discretisation-backend','cuda','--assembly-backend','cuda','--convex-solver','qoco','--outer-loop-backend','cuda','--gpu-execution','graph']
env['SPACEPDHCG_TEST_GTOC12_PHASE_TRACE']='1'
env['SPACEPDHCG_PHASE_OUTPUT']=str(root)
r=dict(source_sha256={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in ['src/spacepdhcg/gtoc12/search.py','src/spacepdhcg/gtoc12/gpu_lambert.py','src/spacepdhcg/gtoc12/lambert.py','src/spacepdhcg/gtoc12/gpu_beam.py','cpp/cuda/src/gtoc12_scvx.cu','build/performance/solver_phase_details.py']},pid=os.getpid(),complete=False,command=cmd,core_sha256=hashlib.sha256(Path(core).read_bytes()).hexdigest(),qoco_sha256=hashlib.sha256(Path(qoco).read_bytes()).hexdigest())
def save():(root/'report.json').write_text(json.dumps(r,indent=2))
save()
try:
 with open('/home/angus/.spacepdhcg-gpu.lock','a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  start=time.perf_counter()
  with (root/'campaign.log').open('x') as log:
   child=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT);r['child_pid']=child.pid;save();code=child.wait(timeout=900)
  r.update(returncode=code,seconds=time.perf_counter()-start);save();assert code==0
  result=json.loads((root/'output/run_report.json').read_text());assert result['best']['accepted'] and result['best']['official']['ok'] and result['best']['independent']['ok']
  r['complete']=True
except Exception as e:r['error']=repr(e)
save();print(json.dumps(r,indent=2))
