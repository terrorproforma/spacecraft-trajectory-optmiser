from pathlib import Path
import subprocess,os,json,shutil,tarfile,hashlib,ast,fcntl,time
root=Path('/home/ubuntu/spacepdhcg-early-graph-focus-v473');repo=Path('/home/ubuntu/spacepdhcg-early-graph-v471/repo')
report=dict(pid=os.getpid(),complete=False,stages=[],campaigns=[])
def save():(root/'report.json').write_text(json.dumps(report,indent=2))
save()
lock=open('/home/ubuntu/.spacepdhcg-gpu.lock','a');report['stage']='waiting_for_gpu';save()
fcntl.flock(lock,fcntl.LOCK_EX)
try:
 report['source_report']=json.loads(Path('/home/ubuntu/spacepdhcg-early-graph-v471/report.json').read_text())
 cmake='/home/ubuntu/spacepdhcg/v1/.venv/bin/cmake';py='/home/ubuntu/spacepdhcg/v1/.venv/bin/python'
 core=Path('/home/ubuntu/spacepdhcg-early-graph-v466/core-build/cuda/libspacepdhcg_cuda.so');qoco=Path('/home/ubuntu/spacepdhcg-step-final-v359/final/libqoco.so')
 env={k:v for k,v in os.environ.items() if not k.startswith('SPACEPDHCG_TEST_')}
 env['SPACEPDHCG_TEST_QOCO_EARLY_OUTER_GRAPH']='1'
 env.update(PYTHONPATH=str(repo/'src'),SPACEPDHCG_GTOC12_CUDA_LIBRARY=str(core),SPACEPDHCG_QOCO_LIBRARY=str(qoco),SPACEPDHCG_GTOC12_GPU_TESTS='1',SPACEPDHCG_GTOC12_DATA='/home/ubuntu/spacepdhcg/gtoc12/benchmarks/gtoc12/data',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
 for node in ast.parse(Path('/home/ubuntu/spacepdhcg-diagnose-v174/diagnose.py').read_text()).body:
  if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='runtime' for t in node.targets):runtime=ast.literal_eval(node.value)
 env['LD_LIBRARY_PATH']=str(core.parent)+':'+str(qoco.parent)+':'+runtime+'/lib:/usr/local/cuda/lib64'
 def run(name,cmd,timeout=900,environment=None,cwd=None):
  start=time.perf_counter()
  with (root/(name+'.log')).open('x') as log:
   child=subprocess.Popen(cmd,cwd=cwd or repo,env=environment or env,stdout=log,stderr=subprocess.STDOUT)
   report['child_pid']=child.pid;report['stage']=name;save()
   try:code=child.wait(timeout=timeout)
   except subprocess.TimeoutExpired:child.kill();child.wait();raise
  report['stages'].append(dict(name=name,returncode=code,seconds=time.perf_counter()-start,command=cmd));save()
  assert code==0,(name,code)
 report['runtime_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [core,qoco]};save()
 boot="import sys;sys.meta_path=[f for f in sys.meta_path if f.__class__.__module__!='_editable_skbc_spacepdhcg'];import pytest;sys.exit(pytest.main(sys.argv[1:]))"
 replay="import sys,runpy;sys.meta_path=[f for f in sys.meta_path if f.__class__.__module__!='_editable_skbc_spacepdhcg'];runpy.run_path('build/performance/replay_early_graph.py',run_name='__main__')"
 for name,mode in [('baseline0','0'),('candidate0','1'),('candidate1','1'),('baseline1','0')]:
  run(name,[py,'-c',replay,mode,str(root/name),'44,201,98'],300)
 report['complete']=True
except Exception as e:report['error']=repr(e)
save();print('complete',report['complete'],report.get('error'),flush=True)
