from pathlib import Path
import os,subprocess,json,time,fcntl,hashlib
root=Path('/home/ubuntu/spacepdhcg-hops-followup-v190');repo=root/'repo'
lock=open('/home/ubuntu/.spacepdhcg-gpu.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
report=dict(start=time.time(),pid=os.getpid(),steps=[],runtime_sha256=hashlib.sha256((root/'liblambert.so').read_bytes()).hexdigest())
env=os.environ.copy();env.update(PYTHONPATH=str(repo/'src'),SPACEPDHCG_GTOC12_CUDA_LIBRARY=str(root/'liblambert.so'),SPACEPDHCG_GTOC12_GPU_TESTS='1',SPACEPDHCG_GTOC12_DATA=str(root/'data'),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
python='/home/ubuntu/spacepdhcg/v1/.venv/bin/python'
boot="import sys,runpy;sys.meta_path=[f for f in sys.meta_path if f.__class__.__module__!='_editable_skbc_spacepdhcg'];runpy.run_module('pytest',run_name='__main__')"
benchmark="import sys,runpy;sys.meta_path=[f for f in sys.meta_path if f.__class__.__module__!='_editable_skbc_spacepdhcg'];sys.argv=sys.argv[1:];runpy.run_path(sys.argv[0],run_name='__main__')"
def run(name,cmd):
 start=time.time()
 with (root/(name+'.log')).open('x') as log:r=subprocess.run(cmd,cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=240)
 report['steps'].append(dict(name=name,command=cmd,returncode=r.returncode,seconds=time.time()-start));(root/'report.json').write_text(json.dumps(report,indent=2))
 if r.returncode:raise RuntimeError(name+' failed')
try:
 run('tests',[python,'-c',boot,'tests/test_gtoc12_gpu_hops.py','tests/test_gtoc12_gpu_lambert.py','tests/test_gtoc12_lambert_performance.py','-q','--tb=short'])
 run('paired',[python,'-c',benchmark,'scripts/gpu/benchmark_gtoc12_combined_hops.py',str(root/'paired.json')])
 run('routes',[python,'-c',benchmark,'scripts/gpu/benchmark_gtoc12_route_search.py',str(root/'routes.json'),'--profile','reduced','--repeats','2'])
 report['complete']=True
except Exception as e:report['complete']=False;report['error']=str(e)
report['end']=time.time();(root/'report.json').write_text(json.dumps(report,indent=2))
