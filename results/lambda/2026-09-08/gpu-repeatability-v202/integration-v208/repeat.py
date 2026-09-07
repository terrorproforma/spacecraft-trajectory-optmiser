from pathlib import Path
import os,sys,json,time,fcntl,hashlib,ast
root=Path('/home/ubuntu/spacepdhcg-integration-v208');integrated=Path('/home/ubuntu/spacepdhcg-regularization-v205');repo=integrated/'repo'
sys.meta_path=[f for f in sys.meta_path if f.__class__.__module__!='_editable_skbc_spacepdhcg'];sys.path.insert(0,str(repo/'src'))
for key in list(os.environ):
 if key.startswith('SPACEPDHCG_TEST_'):del os.environ[key]
core=integrated/'core-build/cuda/libspacepdhcg_cuda.so';qoco=Path('/home/ubuntu/spacepdhcg-diagnose-v174/final/libqoco.so')
os.environ.update(SPACEPDHCG_GTOC12_CUDA_LIBRARY=str(core),SPACEPDHCG_QOCO_LIBRARY=str(qoco),SPACEPDHCG_GTOC12_DATA='/home/ubuntu/spacepdhcg/gtoc12/benchmarks/gtoc12/data',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
# cuDSS is dynamically opened by the adapter using its configured absolute paths.
import subprocess
report=dict(pid=os.getpid(),start=time.time(),steps=[])
os.environ['SPACEPDHCG_GTOC12_GPU_TESTS']='1';os.environ['PYTHONPATH']=str(repo/'src');os.environ['PYTHONPATH']=str(repo/'src')
for flag in ['GTOC12_STATE_ORIGIN','GTOC12_DEFERRED_REPORTS','GTOC12_DEVICE_SCHEDULING','GTOC12_DEVICE_REFRESH','GTOC12_DEVICE_ASSEMBLY_VALIDATION','QOCO_DEVICE_VALIDATION','QOCO_NATIVE_NUMERIC_REPLAY','GTOC12_DEVICE_QUALIFICATION','QOCO_NATIVE_REPLAY','QOCO_IPM_GRAPH']:os.environ['SPACEPDHCG_TEST_'+flag]='1'
lock=open('/home/ubuntu/.spacepdhcg-gpu.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
boot="import sys,runpy;sys.meta_path=[f for f in sys.meta_path if f.__class__.__module__!='_editable_skbc_spacepdhcg'];runpy.run_module('pytest',run_name='__main__')"
command=[sys.executable,'-c',boot,*[str(p.relative_to(repo)) for p in sorted((repo/'tests').glob('test_gtoc12_gpu*.py'))],'tests/test_gtoc12_run_refinement.py','-q','--tb=short']
with (root/'integration.log').open('x') as log:r=subprocess.run(command,cwd=repo,stdout=log,stderr=subprocess.STDOUT,timeout=900)
report.update(complete=True,returncode=r.returncode,command=command,end=time.time());(root/'report.json').write_text(json.dumps(report,indent=2))
