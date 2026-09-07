from pathlib import Path
import os,subprocess,json,time,fcntl
root=Path('/home/ubuntu/spacepdhcg-elements-sanitize-v217');base=Path('/home/ubuntu/spacepdhcg-elements-v215');repo=base/'repo'
report=dict(pid=os.getpid(),start=time.time(),complete=False,steps=[])
env=dict(os.environ,PYTHONPATH=str(repo/'src'),SPACEPDHCG_GTOC12_GPU_TESTS='1',SPACEPDHCG_GTOC12_CUDA_LIBRARY=str(base/'core-build/cuda/libspacepdhcg_cuda.so'),SPACEPDHCG_GTOC12_DATA='/home/ubuntu/spacepdhcg/gtoc12/benchmarks/gtoc12/data',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
lock=open('/home/ubuntu/.spacepdhcg-gpu.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
boot="import sys,runpy;sys.meta_path=[f for f in sys.meta_path if f.__class__.__module__!='_editable_skbc_spacepdhcg'];runpy.run_module('pytest',run_name='__main__')"
try:
 for tool in ['memcheck','initcheck']:
  cmd=['/usr/local/cuda/bin/compute-sanitizer','--tool',tool,'--error-exitcode','99','/home/ubuntu/spacepdhcg/v1/.venv/bin/python','-c',boot,'tests/test_gtoc12_gpu_elements.py','-q']
  with (root/(tool+'.log')).open('x') as log:r=subprocess.run(cmd,cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=600)
  report['steps'].append(dict(tool=tool,command=cmd,returncode=r.returncode));r.check_returncode()
 report['complete']=True
except Exception as e:report['error']=str(e)
report['end']=time.time();(root/'report.json').write_text(json.dumps(report,indent=2))
