from pathlib import Path
import os,subprocess,json,time,hashlib,shutil,fcntl
root=Path('build/performance/warp-pipeline-v386');root.mkdir(exist_ok=False)
core=Path('/home/angus/build-spacepdhcg-warp-hops-v385/final/libspacepdhcg_cuda.so')
qoco=Path('/home/angus/build-qoco-soc-step-v358/final/libqoco.so')
names=['cpp/cuda/src/orbitweaver_gpu.cu','tests/test_gtoc12_gpu_hops.py','cpp/cuda/src/gtoc12_collect_dp.cu','cpp/cuda/include/spacepdhcg/cuda/gtoc12_collect_dp_c_api.h','src/spacepdhcg/gtoc12/collectdp.py','src/spacepdhcg/gtoc12/gpu_collect_tables.py','src/spacepdhcg/gtoc12/refinement_queue.py','src/spacepdhcg/gtoc12/cli.py','src/spacepdhcg/gtoc12/lambert.py','tests/test_gtoc12_gpu_harvest_window.py','tests/test_gtoc12_refinement_queue.py','tests/test_gtoc12_run_refinement.py']
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
  tests=['tests/test_gtoc12_gpu_hops.py','tests/test_gtoc12_gpu_lambert.py','tests/test_gtoc12_refinement_queue.py','tests/test_gtoc12_run_refinement.py','tests/test_gtoc12_gpu_harvest_window.py','tests/test_gtoc12_gpu_resident_collect_tables.py','tests/test_gtoc12_gpu_collect_tables.py','tests/test_gtoc12_gpu_collect_dp.py','tests/test_gtoc12_gpu_scvx.py','tests/test_gtoc12_gpu_cli.py']
  run('pytest',[py,'-c',boot,*tests,'-q'],300)
  run('memcheck',['/usr/local/cuda-12.8/bin/compute-sanitizer','--tool','memcheck','--error-exitcode','99',py,'-c',boot,tests[0],'-q'],120)
  run('synccheck',['/usr/local/cuda-12.8/bin/compute-sanitizer','--tool','synccheck','--error-exitcode','99',py,'-c',boot,tests[0],'-q'],120)
  cli="import sys,runpy;sys.meta_path=[f for f in sys.meta_path if f.__class__.__module__!='_editable_skbc_spacepdhcg'];runpy.run_module('spacepdhcg',run_name='__main__')"
  for name,candidate in [('baseline0',False),('candidate0',True),('candidate1',True),('baseline1',False)]:
   lib=core if candidate else Path('/home/angus/build-spacepdhcg-recovery-v380/final/libspacepdhcg_cuda.so')
   env['SPACEPDHCG_GTOC12_CUDA_LIBRARY']=str(lib);env['LD_LIBRARY_PATH']=str(lib.parent)+':'+str(qoco.parent)+':/home/angus/cudss-isolated-0.8.0.10/nvidia/cu12/lib:/usr/local/cuda-12.8/lib64'
   cmd=[py,'-c',cli,'gtoc12','run','--run-id','warp386_'+name,'--output',str(root/name/'output'),'--full-catalogue','--ships','1','--beam-width','16','--max-deploys','10','--neighbours','48','--refine-top','3','--search-budget-seconds','120','--budget-seconds','600','--retime-attempts','4','--retime-budget-seconds','300','--no-cooperative','--node-days','2','--scvx-iterations','40','--screening-backend','cuda','--seed-backend','cuda','--discretisation-backend','cuda','--assembly-backend','cuda','--convex-solver','qoco','--outer-loop-backend','cuda','--gpu-execution','graph']
   run(name,cmd,900)
   r=json.loads((root/name/'output/run_report.json').read_text());assert r['best']['official']['ok'] and r['best']['independent']['ok']
   report.setdefault('campaigns',[]).append(dict(name=name,candidate=candidate,seconds=r['wall_seconds_total'],score=r['best']['independent']['weighted_score_fixed_bonus_kg'],screening=r['screening']));save()
 report['complete']=True
except Exception as e:report['error']=repr(e)
save();print(json.dumps({k:v for k,v in report.items() if k not in ['source_sha256','stages']},indent=2))
