from pathlib import Path
import os,subprocess,json,time,fcntl,ast,hashlib
root=Path('/home/ubuntu/spacepdhcg-stationary-fleet-v410')
base=Path('/home/ubuntu/spacepdhcg-stationary-v408');repo=base/'repo'
core=base/'core-build/cuda/libspacepdhcg_cuda.so';qoco=Path('/home/ubuntu/spacepdhcg-step-final-v359/final/libqoco.so')
report=dict(pid=os.getpid(),complete=False,campaigns=[],stages=[],runtime_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [core,qoco]})
def save():(root/'report.json').write_text(json.dumps(report,indent=2))
save()
try:
 with open('/home/ubuntu/.spacepdhcg-gpu.lock','a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  c=json.loads((base/'comparison.json').read_text());assert not c['lost_baseline'] and not c['lost_prior'] and not c['candidate']['uncertified']
  env={k:v for k,v in os.environ.items() if not k.startswith(('SPACEPDHCG_TEST_','QOCO_REPLAY_'))}
  env.update(PYTHONPATH=str(repo/'src'),SPACEPDHCG_GTOC12_CUDA_LIBRARY=str(core),SPACEPDHCG_QOCO_LIBRARY=str(qoco),SPACEPDHCG_GTOC12_DATA='/home/ubuntu/spacepdhcg/gtoc12/benchmarks/gtoc12/data',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
  for node in ast.parse(Path('/home/ubuntu/spacepdhcg-diagnose-v174/diagnose.py').read_text()).body:
   if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='runtime' for t in node.targets):runtime=ast.literal_eval(node.value)
  env['LD_LIBRARY_PATH']=str(core.parent)+':'+str(qoco.parent)+':'+runtime+'/lib:/usr/local/cuda/lib64'
  boot="import sys,runpy;sys.meta_path=[f for f in sys.meta_path if f.__class__.__module__!='_editable_skbc_spacepdhcg'];runpy.run_module('spacepdhcg',run_name='__main__')"
  for name,mode in [('baseline0','0'),('candidate0','1'),('candidate1','1'),('baseline1','0')]:
   env['SPACEPDHCG_TEST_GTOC12_STATIONARY_FAILURE']=mode
   cmd=['/home/ubuntu/spacepdhcg/v1/.venv/bin/python','-c',boot,'gtoc12','run','--run-id','stationary410_'+name,'--output',str(root/name/'output'),'--full-catalogue','--ships','1','--beam-width','16','--max-deploys','10','--neighbours','48','--refine-top','3','--search-budget-seconds','120','--budget-seconds','600','--retime-attempts','4','--retime-budget-seconds','300','--no-cooperative','--node-days','2','--scvx-iterations','40','--screening-backend','cuda','--seed-backend','cuda','--discretisation-backend','cuda','--assembly-backend','cuda','--convex-solver','qoco','--outer-loop-backend','cuda','--gpu-execution','graph']
   with (root/(name+'.log')).open('x') as log:
    start=time.perf_counter();child=subprocess.Popen(cmd,cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT);report.update(stage=name,child_pid=child.pid);save();code=child.wait(timeout=900)
   report['stages'].append(dict(name=name,returncode=code,seconds=time.perf_counter()-start,command=cmd));save();assert code==0
   r=json.loads((root/name/'output/run_report.json').read_text());assert r['best']['accepted'] and r['best']['official']['ok'] and r['best']['independent']['ok']
   report['campaigns'].append(dict(name=name,candidate=mode=='1',seconds=r['wall_seconds_total'],score=r['best']['independent']['weighted_score_fixed_bonus_kg'],screening=r['screening']));save()
  report['complete']=True
except Exception as e:report['error']=repr(e)
save();print(json.dumps(report,indent=2))
