from pathlib import Path
import os,subprocess,json,time,fcntl,hashlib
root=Path('/home/ubuntu/spacepdhcg-collection-v197');repo=root/'repo'
lock=open('/home/ubuntu/.spacepdhcg-gpu.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
report=dict(start=time.time(),pid=os.getpid(),steps=[])
env=os.environ.copy();env.update(PYTHONPATH=str(repo/'src'),SPACEPDHCG_GTOC12_CUDA_LIBRARY=str(root/'libscreening.so'),SPACEPDHCG_GTOC12_GPU_TESTS='1',SPACEPDHCG_GTOC12_DATA=str(root/'data'),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',LD_LIBRARY_PATH=str(root))
python='/home/ubuntu/spacepdhcg/v1/.venv/bin/python'
nvcc=['/usr/local/cuda/bin/nvcc','-std=c++20','-arch=sm_90','-lineinfo','-Icpp/cuda/include','-Icpp/include']
def run(name,cmd,timeout=240):
 start=time.time()
 with (root/(name+'.log')).open('x') as log:r=subprocess.run(cmd,cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=timeout)
 report['steps'].append(dict(name=name,command=cmd,returncode=r.returncode,seconds=time.time()-start));(root/'report.json').write_text(json.dumps(report,indent=2))
 if r.returncode:raise RuntimeError(name+' failed')
try:
 run('hardware',['nvidia-smi','--query-gpu=name,utilization.gpu,memory.used','--format=csv'])
 run('collection-build',nvcc+['--fmad=false','-c','-Xcompiler=-fPIC,-Wall,-Wextra,-Werror','cpp/cuda/src/gtoc12_collection.cu','-o',str(root/'collection.o')])
 run('link',nvcc+['-shared',str(root/'collection.o'),str(root/'neighbours.o'),str(root/'lambert.o'),'-o',str(root/'libscreening.so')])
 report['runtime_sha256']=hashlib.sha256((root/'libscreening.so').read_bytes()).hexdigest()
 run('native-build',nvcc+['cpp/cuda/tests/gtoc12_collection_test.cu','-L'+str(root),'-lscreening','-o',str(root/'native')])
 run('native',[str(root/'native')])
 boot="import sys,runpy;sys.meta_path=[f for f in sys.meta_path if f.__class__.__module__!='_editable_skbc_spacepdhcg'];runpy.run_module('pytest',run_name='__main__')"
 run('tests',[python,'-c',boot,'tests/test_gtoc12_gpu_collection.py','tests/test_gtoc12_gpu_neighbours.py','tests/test_gtoc12_gpu_hops.py','tests/test_gtoc12_gpu_lambert.py','tests/test_gtoc12_lambert_performance.py','-q','--tb=short'])
 benchmark="import sys,runpy;sys.meta_path=[f for f in sys.meta_path if f.__class__.__module__!='_editable_skbc_spacepdhcg'];sys.argv=sys.argv[1:];runpy.run_path(sys.argv[0],run_name='__main__')"
 run('paired',[python,'-c',benchmark,'scripts/gpu/benchmark_gtoc12_collection.py',str(root/'paired.json')])
 run('routes',[python,'-c',benchmark,'scripts/gpu/benchmark_gtoc12_route_search.py',str(root/'routes.json'),'--profile','reduced','--repeats','2'])
 for name in ['memcheck','racecheck']:
  run(name,['/usr/local/cuda/bin/compute-sanitizer','--tool',name,'--error-exitcode','98',str(root/'native')])
 report['complete']=True
except Exception as e:report['complete']=False;report['error']=str(e)
report['end']=time.time();(root/'report.json').write_text(json.dumps(report,indent=2))
