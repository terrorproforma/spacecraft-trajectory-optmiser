from pathlib import Path
import os,subprocess,json,time,hashlib,shutil,fcntl
root=Path('build/performance/earth-beam-v440');root.mkdir(exist_ok=False)
core=Path('/home/angus/build-spacepdhcg-earth-beam-v431/final/libspacepdhcg_cuda.so')
qoco=Path('/home/angus/build-qoco-soc-step-v358/final/libqoco.so')
names=['cpp/cuda/src/orbitweaver_gpu.cu', 'cpp/cuda/src/orbitweaver_beam.cuh', 'cpp/cuda/include/spacepdhcg/cuda/orbitweaver_gpu_c_api.h', 'cpp/cuda/tests/orbitweaver_options_test.cu', 'src/spacepdhcg/gtoc12/gpu_lambert.py', 'src/spacepdhcg/gtoc12/gpu_beam.py', 'src/spacepdhcg/gtoc12/lambert.py', 'src/spacepdhcg/gtoc12/search.py', 'tests/test_gtoc12_gpu_elements.py', 'tests/test_gtoc12_gpu_beam.py']
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
  cli="import sys,runpy;sys.meta_path=[f for f in sys.meta_path if f.__class__.__module__!='_editable_skbc_spacepdhcg'];runpy.run_module('spacepdhcg',run_name='__main__')"
  for name,candidate in [('baseline0',False),('candidate0',True),('candidate1',True),('baseline1',False)]:
   lib=core
   env['SPACEPDHCG_TEST_GTOC12_EARTH_BEAM']='1' if candidate else '0'
   env['SPACEPDHCG_GTOC12_CUDA_LIBRARY']=str(lib);env['LD_LIBRARY_PATH']=str(lib.parent)+':'+str(qoco.parent)+':/home/angus/cudss-isolated-0.8.0.10/nvidia/cu12/lib:/usr/local/cuda-12.8/lib64'
   cmd=[py,'-c',cli,'gtoc12','run','--run-id','earth440_'+name,'--output',str(root/name/'output'),'--full-catalogue','--ships','1','--beam-width','16','--max-deploys','10','--neighbours','48','--refine-top','3','--search-budget-seconds','120','--budget-seconds','600','--retime-attempts','4','--retime-budget-seconds','300','--no-cooperative','--node-days','2','--scvx-iterations','40','--screening-backend','cuda','--seed-backend','cuda','--discretisation-backend','cuda','--assembly-backend','cuda','--convex-solver','qoco','--outer-loop-backend','cuda','--gpu-execution','graph']
   run(name,cmd,900)
   r=json.loads((root/name/'output/run_report.json').read_text());assert r['best']['accepted'] and r['best']['official']['ok'] and r['best']['independent']['ok']
   report.setdefault('campaigns',[]).append(dict(name=name,candidate=candidate,seconds=r['wall_seconds_total'],score=r['best']['independent']['weighted_score_fixed_bonus_kg'],screening=r['screening']));save()
 report['complete']=True
except Exception as e:report['error']=repr(e)
save();print(json.dumps({k:v for k,v in report.items() if k not in ['source_sha256','stages']},indent=2))
