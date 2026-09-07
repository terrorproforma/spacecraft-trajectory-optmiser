from pathlib import Path
import os,subprocess,json,time,fcntl,hashlib
root=Path('/home/ubuntu/spacepdhcg-hops-v188');repo=root/'repo'
lock=open('/home/ubuntu/.spacepdhcg-gpu.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
report=dict(start=time.time(),pid=os.getpid(),steps=[])
env=os.environ.copy();env.update(PYTHONPATH=str(repo/'src'),SPACEPDHCG_GTOC12_CUDA_LIBRARY=str(root/'liblambert.so'),SPACEPDHCG_GTOC12_GPU_TESTS='1',SPACEPDHCG_GTOC12_DATA=str(root/'data'),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',LD_LIBRARY_PATH=str(root))
python='/home/ubuntu/spacepdhcg/v1/.venv/bin/python'
nvcc=['/usr/local/cuda/bin/nvcc','-std=c++20','-arch=sm_90','-lineinfo','-Icpp/cuda/include','-Icpp/include']
def run(name,cmd,timeout=300):
 start=time.time()
 with (root/(name+'.log')).open('x') as log:r=subprocess.run(cmd,cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=timeout)
 report['steps'].append(dict(name=name,command=cmd,returncode=r.returncode,seconds=time.time()-start));(root/'report.json').write_text(json.dumps(report,indent=2))
 if r.returncode:raise RuntimeError(name+' failed')
try:
 run('hardware',['nvidia-smi','--query-gpu=name,utilization.gpu,memory.used','--format=csv'])
 run('build',nvcc+['-shared','-Xcompiler=-fPIC,-Wall,-Wextra,-Werror','cpp/cuda/src/orbitweaver_gpu.cu','-o',str(root/'liblambert.so')])
 report['runtime_sha256']=hashlib.sha256((root/'liblambert.so').read_bytes()).hexdigest()
 for name,source in [('native','orbitweaver_hop_test.cu'),('screening-native','orbitweaver_screening_test.cu'),('legacy','orbitweaver_gpu_test.cu')]:
  run(name+'-build',nvcc+['-Xcompiler=-Wall,-Wextra,-Werror','cpp/cuda/tests/'+source,'-L'+str(root),'-llambert','-o',str(root/name)])
  run(name,[str(root/name)])
 boot="import sys,runpy;sys.meta_path=[f for f in sys.meta_path if f.__class__.__module__!='_editable_skbc_spacepdhcg'];runpy.run_module('pytest',run_name='__main__')"
 run('tests',[python,'-c',boot,'tests/test_gtoc12_gpu_hops.py','tests/test_gtoc12_gpu_lambert.py','tests/test_gtoc12_lambert_performance.py','-q','--tb=short'])
 benchmark="import sys,runpy;sys.meta_path=[f for f in sys.meta_path if f.__class__.__module__!='_editable_skbc_spacepdhcg'];sys.argv=sys.argv[1:];runpy.run_path(sys.argv[0],run_name='__main__')"
 for name,source in [('screening','benchmark_gtoc12_screening.py'),('routes','benchmark_gtoc12_route_search.py')]:
  run(name,[python,'-c',benchmark,'scripts/gpu/'+source,str(root/(name+'.json'))])
 for name,tool in [('memcheck','memcheck'),('racecheck','racecheck')]:
  run(name,['/usr/local/cuda/bin/compute-sanitizer','--tool',tool,'--error-exitcode','98',str(root/'native')])
 report['complete']=True
except Exception as e: report['error']=str(e);report['complete']=False
report['end']=time.time();(root/'report.json').write_text(json.dumps(report,indent=2))
