from pathlib import Path
import os,subprocess,json,time,shutil,tarfile,hashlib,fcntl
root=Path('/home/ubuntu/spacepdhcg-collect-tables-v290');repo=root/'repo'
report=dict(pid=os.getpid(),complete=False,steps=[])
try:
 shutil.copytree('/home/ubuntu/spacepdhcg-collect-dp-v279/repo',repo,ignore=shutil.ignore_patterns('.git','__pycache__','.pytest_cache'),symlinks=True)
 with tarfile.open('/tmp/collect-tables-v290.tar.gz') as tar:tar.extractall(repo,filter='data')
 report['source_sha256']=json.loads((repo/'source-sha256.json').read_text())
 for name,digest in report['source_sha256'].items():assert hashlib.sha256((repo/name).read_text().encode()).hexdigest()==digest,name
 core=Path('/home/ubuntu/spacepdhcg-collect-dp-v279/core-build/cuda/libspacepdhcg_cuda.so')
 report['runtime_sha256']=hashlib.sha256(core.read_bytes()).hexdigest()
 env=os.environ.copy();env.update(PYTHONPATH=str(repo/'src'),SPACEPDHCG_GTOC12_CUDA_LIBRARY=str(core),SPACEPDHCG_GTOC12_DATA='/home/ubuntu/spacepdhcg/gtoc12/benchmarks/gtoc12/data',SPACEPDHCG_GTOC12_GPU_TESTS='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
 python='/home/ubuntu/spacepdhcg/v1/.venv/bin/python'
 prefix="import sys,runpy;sys.meta_path=[f for f in sys.meta_path if f.__class__.__module__!='_editable_skbc_spacepdhcg'];"
 pytest=[python,'-c',prefix+"runpy.run_module('pytest',run_name='__main__')"]
 def run(name,cmd):
  start=time.perf_counter()
  with (root/(name+'.log')).open('x') as log:r=subprocess.run(cmd,cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=180)
  report['steps'].append(dict(name=name,command=cmd,returncode=r.returncode,seconds=time.perf_counter()-start));(root/'report.json').write_text(json.dumps(report,indent=2));r.check_returncode()
 lock=open('/home/ubuntu/.spacepdhcg-gpu.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 run('tests',pytest+['tests/test_gtoc12_gpu_collect_tables.py','tests/test_gtoc12_gpu_elements.py','tests/test_gtoc12_gpu_collect_dp.py','tests/test_gtoc12_collectdp.py','-q'])
 run('memcheck',['/usr/local/cuda/bin/compute-sanitizer','--tool','memcheck','--error-exitcode','99']+pytest+['tests/test_gtoc12_gpu_collect_tables.py','-q'])
 for name,script in [('replay','replay_gtoc12_collection_fixtures.py'),('benchmark','benchmark_gtoc12_collection_tables.py')]:
  run(name,[python,'-c',prefix+"runpy.run_path('scripts/gpu/"+script+"',run_name='__main__')",'/home/ubuntu/spacepdhcg-collect-profile-v272/timing.json',str(root/(name+'.json'))])
 report['complete']=True
except Exception as e:report['error']=str(e)
(root/'report.json').write_text(json.dumps(report,indent=2))
