from pathlib import Path
import subprocess,os,fcntl,json,time,hashlib
root=Path('build/performance/retained-replay-campaign-v513');root.mkdir(exist_ok=False)
core='/home/angus/build-spacepdhcg-retained-replay-v512/final/libspacepdhcg_cuda.so';qoco='/home/angus/build-qoco-soc-step-v358/final/libqoco.so'
env={k:v for k,v in os.environ.items() if not k.startswith(('SPACEPDHCG_TEST_','QOCO_REPLAY_'))}
env.update(PYTHONPATH=str(Path('src').resolve()),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',SPACEPDHCG_GTOC12_CUDA_LIBRARY=core,SPACEPDHCG_QOCO_LIBRARY=qoco,SPACEPDHCG_GTOC12_DATA='/home/angus/worktrees/spacepdhcg-release/benchmarks/gtoc12/data',LD_LIBRARY_PATH=str(Path(core).parent)+':'+str(Path(qoco).parent)+':/home/angus/cudss-isolated-0.8.0.10/nvidia/cu12/lib:/usr/local/cuda-12.8/lib64')
boot="import sys,runpy;sys.meta_path=[f for f in sys.meta_path if f.__class__.__module__!='_editable_skbc_spacepdhcg'];runpy.run_path('build/performance/solver_phase_details.py',run_name='__main__')"
cmd=['/home/angus/worktrees/spacepdhcg-literature-venv/bin/python','-c',boot,'gtoc12','run','--run-id','retained_replay513','--output',str(root/'output'),'--full-catalogue','--ships','1','--beam-width','16','--max-deploys','10','--neighbours','48','--refine-top','3','--search-budget-seconds','120','--budget-seconds','900','--retime-attempts','4','--retime-budget-seconds','300','--no-cooperative','--node-days','2','--scvx-iterations','40','--screening-backend','cuda','--seed-backend','cuda','--discretisation-backend','cuda','--assembly-backend','cuda','--convex-solver','qoco','--outer-loop-backend','cuda','--gpu-execution','graph']
r=dict(source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),source_sha256={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in ['src/spacepdhcg/gtoc12/search.py','src/spacepdhcg/gtoc12/gpu_lambert.py','src/spacepdhcg/gtoc12/lambert.py','src/spacepdhcg/gtoc12/gpu_beam.py','src/spacepdhcg/gtoc12/gpu_options.py','cpp/cuda/src/native_qoco_adapter.cpp','build/performance/solver_phase_details.py']},pid=os.getpid(),complete=False,command=cmd,core_sha256=hashlib.sha256(Path(core).read_bytes()).hexdigest(),qoco_sha256=hashlib.sha256(Path(qoco).read_bytes()).hexdigest())
def save():(root/'report.json').write_text(json.dumps(r,indent=2))
save()
try:
 with open('/home/angus/.spacepdhcg-gpu.lock','a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  r['campaigns']=[]
  for name,enabled in [('baseline0',False),('candidate0',True),('candidate1',True),('baseline1',False)]:
   env['SPACEPDHCG_TEST_QOCO_RETAINED_REPLAY']='1' if enabled else '0'
   env['SPACEPDHCG_TEST_GTOC12_PHASE_TRACE']='1'
   env['SPACEPDHCG_TEST_QOCO_NATIVE_REPLAY_TRACE']='1'
   (root/name).mkdir()
   env['SPACEPDHCG_PHASE_OUTPUT']=str(root/name)
   command=cmd.copy();command[command.index('--output')+1]=str(root/name/'output')
   start=time.perf_counter()
   with (root/(name+'.log')).open('x') as log:
    child=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT);r.update(child_pid=child.pid,stage=name);save();code=child.wait(timeout=900)
   seconds=time.perf_counter()-start
   assert code==0,(name,code)
   result=json.loads((root/name/'output/run_report.json').read_text());assert result['best']['accepted'] and result['best']['official']['ok'] and result['best']['independent']['ok']
   screening=result['screening'];assert screening.get('resident_option_builds',0)>0
   if enabled:assert screening.get('resident_option_read_bytes',0)==0 and screening['collection_option_upload_bytes']==0
   r['campaigns'].append(dict(name=name,candidate=enabled,process_seconds=seconds,cli_seconds=result['wall_seconds_total'],score=result['best']['independent']['weighted_score_fixed_bonus_kg'],screening=screening));save()
  r['complete']=True
except Exception as e:r['error']=repr(e)
save();print(json.dumps(r,indent=2))
