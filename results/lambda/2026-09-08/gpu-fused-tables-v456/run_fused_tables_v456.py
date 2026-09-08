from pathlib import Path
import subprocess,os,json,shutil,tarfile,hashlib,ast,fcntl,time
root=Path('/home/ubuntu/spacepdhcg-fused-tables-v456');repo=root/'repo';baseline=Path('/home/ubuntu/spacepdhcg-gpu-execution-v328/repo')
report=dict(pid=os.getpid(),complete=False,stages=[],campaigns=[])
def save():(root/'report.json').write_text(json.dumps(report,indent=2))
save()
lock=open('/home/ubuntu/.spacepdhcg-gpu.lock','a');report['stage']='waiting_for_gpu';save()
fcntl.flock(lock,fcntl.LOCK_EX)
try:
 shutil.copytree('/home/ubuntu/spacepdhcg-fused-tables-v449/repo',repo,ignore=shutil.ignore_patterns('__pycache__','.pytest_cache','.git'))
 with tarfile.open('/tmp/fused-tables-v456.tar.gz') as t:
  for m in t.getmembers():
   target=(repo/m.name).resolve();assert target.is_relative_to(repo.resolve()) and m.isfile()
   target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(t.extractfile(m).read())
 source=json.loads((repo/'fused-tables-source-sha256.json').read_text())
 assert all(hashlib.sha256((repo/p).read_bytes()).hexdigest()==sha for p,sha in source.items())
 report['source_sha256']=source
 subprocess.run(['git','init',str(repo)],check=True,capture_output=True)
 subprocess.run(['git','-C',str(repo),'add','.'],check=True,capture_output=True)
 subprocess.run(['git','-C',str(repo),'-c','user.name=GPU validation','-c','user.email=gpu-validation@localhost','commit','-m','Freeze fleet recovery validation source'],check=True,capture_output=True)
 report['frozen_source_commit']=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
 cmake='/home/ubuntu/spacepdhcg/v1/.venv/bin/cmake';py='/home/ubuntu/spacepdhcg/v1/.venv/bin/python'
 core=root/'core-build/cuda/libspacepdhcg_cuda.so';qoco=Path('/home/ubuntu/spacepdhcg-step-final-v359/final/libqoco.so')
 env={k:v for k,v in os.environ.items() if not k.startswith('SPACEPDHCG_TEST_')}
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
 run('configure',[cmake,'-S',str(repo/'cpp'),'-B',str(root/'core-build'),'-G','Ninja','-DCMAKE_BUILD_TYPE=Release','-DSPACEPDHCG_BUILD_CUDA=ON','-DSPACEPDHCG_BUILD_NATIVE_TESTS=OFF','-DBUILD_TESTING=ON','-DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc','-DCMAKE_CUDA_ARCHITECTURES=90','-DSPACEPDHCG_PDHCG_SOURCE_ROOT=/home/ubuntu/spacepdhcg/v1/_upstream/pdhcg'])
 run('build',[cmake,'--build',str(root/'core-build'),'--target','spacepdhcg_cuda','-j','3'])
 report['runtime_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [core,qoco]};save()
 boot="import sys;sys.meta_path=[f for f in sys.meta_path if f.__class__.__module__!='_editable_skbc_spacepdhcg'];import pytest;sys.exit(pytest.main(sys.argv[1:]))"
 tests=['tests/test_gtoc12_gpu_fused_tables.py', 'tests/test_gtoc12_gpu_collect_tables.py', 'tests/test_gtoc12_gpu_resident_collect_tables.py', 'tests/test_gtoc12_gpu_harvest_window.py']
 run('pytest',[py,'-c',boot,*tests,'-q'],300)
 for tool in ['memcheck','synccheck','racecheck']:
  run(tool,['/usr/local/cuda/bin/compute-sanitizer','--tool',tool,'--error-exitcode','99',py,'-c',boot,'tests/test_gtoc12_gpu_fused_tables.py','-q'],900)
 cli="import sys,runpy;sys.meta_path=[f for f in sys.meta_path if f.__class__.__module__!='_editable_skbc_spacepdhcg'];runpy.run_module('spacepdhcg',run_name='__main__')"
 for name,candidate in [('default',True)]:
  cmd=[py,'-c',cli,'gtoc12','run','--run-id','fused456_'+name,'--output',str(root/name/'output'),'--full-catalogue','--ships','1','--beam-width','16','--max-deploys','10','--neighbours','48','--refine-top','3','--search-budget-seconds','120','--budget-seconds','600','--retime-attempts','4','--retime-budget-seconds','300','--no-cooperative','--node-days','2','--scvx-iterations','40','--screening-backend','cuda','--seed-backend','cuda','--discretisation-backend','cuda','--assembly-backend','cuda','--convex-solver','qoco','--outer-loop-backend','cuda','--gpu-execution','graph']
  run(name,cmd,900)
  r=json.loads((root/name/'output/run_report.json').read_text());assert r['best']['accepted'] and r['best']['official']['ok'] and r['best']['independent']['ok']
  report['campaigns'].append(dict(name=name,candidate=candidate,seconds=r['wall_seconds_total'],score=r['best']['independent']['weighted_score_fixed_bonus_kg'],screening=r['screening']));save()
 report['complete']=True
except Exception as e:report['error']=repr(e)
save();print('complete',report['complete'],report.get('error'),flush=True)
