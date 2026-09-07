from pathlib import Path
import subprocess
root=Path('/home/ubuntu/spacepdhcg-conic-retry-v314');root.mkdir(exist_ok=False)
runner=r'''from pathlib import Path
import os,subprocess,json,time,fcntl,ast,hashlib,shutil,tarfile
root=Path('/home/ubuntu/spacepdhcg-conic-retry-v314');repo=root/'repo'
report=dict(pid=os.getpid(),complete=False,stages=[])
shutil.copytree('/home/ubuntu/spacepdhcg-collect-resident-v298/repo',repo,ignore=shutil.ignore_patterns('__pycache__','.pytest_cache'))
with tarfile.open('/tmp/conic-retry-v314.tar.gz') as t:
 for m in t.getmembers():
  target=(repo/m.name).resolve();assert target.is_relative_to(repo.resolve()) and m.isfile()
  target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(t.extractfile(m).read())
manifest=json.loads((repo/'retry-source-sha256.json').read_text())
assert all(hashlib.sha256((repo/p).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==h for p,h in manifest.items())
report['source_lf_sha256']=manifest
env={k:v for k,v in os.environ.items() if not k.startswith('SPACEPDHCG_TEST_')}
cmake='/home/ubuntu/spacepdhcg/v1/.venv/bin/cmake';py='/home/ubuntu/spacepdhcg/v1/.venv/bin/python'
core=root/'core-build/cuda/libspacepdhcg_cuda.so';qoco=Path('/home/ubuntu/spacepdhcg-diagnose-v174/final/libqoco.so')
env.update(PYTHONPATH=str(repo/'src'),SPACEPDHCG_GTOC12_CUDA_LIBRARY=str(core),SPACEPDHCG_QOCO_LIBRARY=str(qoco),SPACEPDHCG_GTOC12_GPU_TESTS='1',SPACEPDHCG_GTOC12_DATA='/home/ubuntu/spacepdhcg/gtoc12/benchmarks/gtoc12/data',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
for node in ast.parse(Path('/home/ubuntu/spacepdhcg-diagnose-v174/diagnose.py').read_text()).body:
 if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='runtime' for t in node.targets):runtime=ast.literal_eval(node.value)
env['LD_LIBRARY_PATH']=str(core.parent)+':'+str(qoco.parent)+':'+runtime+'/lib:/usr/local/cuda/lib64'
def run(name,cmd,timeout=600):
 start=time.perf_counter()
 with (root/(name+'.log')).open('x') as log:r=subprocess.run(cmd,cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=timeout)
 report['stages'].append(dict(name=name,command=cmd,returncode=r.returncode,seconds=time.perf_counter()-start));(root/'report.json').write_text(json.dumps(report,indent=2));print(name,r.returncode,flush=True);r.check_returncode()
try:
 run('configure',[cmake,'-S',str(repo/'cpp'),'-B',str(root/'core-build'),'-G','Ninja','-DCMAKE_BUILD_TYPE=Release','-DSPACEPDHCG_BUILD_CUDA=ON','-DSPACEPDHCG_BUILD_NATIVE_TESTS=OFF','-DBUILD_TESTING=ON','-DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc','-DCMAKE_CUDA_ARCHITECTURES=90','-DSPACEPDHCG_PDHCG_SOURCE_ROOT=/home/ubuntu/spacepdhcg/v1/_upstream/pdhcg'])
 native=['gtoc12_scvx_test','gtoc12_outer_graph_test','gtoc12_graph_deadline_test','gtoc12_qoco_guard_test']
 run('build',[cmake,'--build',str(root/'core-build'),'--target','spacepdhcg_cuda',*native,'-j','3'])
 report['runtime_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [core,qoco]}
 with open('/home/ubuntu/.spacepdhcg-gpu.lock','a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  for name in native:run(name,[str(root/'core-build/cuda-tests'/name)],120)
  boot="import sys;sys.meta_path=[f for f in sys.meta_path if f.__class__.__module__!='_editable_skbc_spacepdhcg'];import pytest;sys.exit(pytest.main(sys.argv[1:]))"
  run('pytest',[py,'-c',boot,'tests/test_gtoc12_gpu_scvx.py','-q'],300)
  for tool in ['memcheck','initcheck','synccheck','racecheck']:
   run(tool,['/usr/local/cuda/bin/compute-sanitizer','--tool',tool,'--error-exitcode','99',str(root/'core-build/cuda-tests/gtoc12_scvx_test')],120)
 # The replay acquires the same lock itself; do not nest ownership.
 arc=Path('/home/ubuntu/spacepdhcg-arc-repeat-v304/run.py').read_text().replace('/home/ubuntu/spacepdhcg-arc-repeat-v304',str(root/'arc')).replace('/home/ubuntu/spacepdhcg-collect-resident-v298',str(root))
 (root/'arc').mkdir();(root/'arc/run.py').write_text(arc)
 run('arc',[py,str(root/'arc/run.py')],120)
 arc_report=json.loads((root/'arc/report.json').read_text());assert arc_report.get('complete') and len(arc_report['rows'])==24
 report['arc_qualified']=sum(row['status']=='converged' and row['qualified'] for row in arc_report['rows'])
 assert report['arc_qualified']==24
 report['complete']=True
except Exception as e:report['error']=str(e)
(root/'report.json').write_text(json.dumps(report,indent=2));print('complete',report['complete'],report.get('error'),flush=True)
'''
(root/'run.py').write_text(runner)
with (root/'runner.log').open('x') as log:p=subprocess.Popen(['python3',str(root/'run.py')],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
print(p.pid)
