from pathlib import Path
import os,subprocess,json,time,fcntl,hashlib,shutil
root=Path('/home/ubuntu/spacepdhcg-collection-full-v198');repo=root/'repo';base=Path('/home/ubuntu/spacepdhcg-collection-v197')
report=dict(start=time.time(),pid=os.getpid(),steps=[])
env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
python='/home/ubuntu/spacepdhcg/v1/.venv/bin/python';cmake='/home/ubuntu/spacepdhcg/v1/.venv/bin/cmake'
def run(name,cmd,timeout=900):
 start=time.time()
 with (root/(name+'.log')).open('x') as log:r=subprocess.run(cmd,cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=timeout)
 report['steps'].append(dict(name=name,command=cmd,returncode=r.returncode,seconds=time.time()-start));(root/'report.json').write_text(json.dumps(report,indent=2))
 if r.returncode:raise RuntimeError(name+' failed')
try:
 shutil.copytree(base/'repo',repo,ignore=shutil.ignore_patterns('.git','__pycache__','.pytest_cache'),symlinks=True)
 run('git-init',['git','init','-q'])
 run('git-add',['git','add','-f','cpp','src','scripts','pyproject.toml'])
 run('git-snapshot',['git','-c','user.name=GPU snapshot','-c','user.email=snapshot@localhost','commit','-qm','Frozen CUDA collection integration source'])
 run('configure',[cmake,'-S',str(repo/'cpp'),'-B',str(root/'core-build'),'-G','Ninja','-DCMAKE_BUILD_TYPE=Release','-DSPACEPDHCG_BUILD_CUDA=ON','-DSPACEPDHCG_BUILD_NATIVE_TESTS=OFF','-DBUILD_TESTING=ON','-DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc','-DCMAKE_CUDA_ARCHITECTURES=90','-DSPACEPDHCG_PDHCG_SOURCE_ROOT=/home/ubuntu/spacepdhcg/v1/_upstream/pdhcg'])
 run('build',[cmake,'--build',str(root/'core-build'),'--target','spacepdhcg_cuda','gtoc12_collection_test','gtoc12_neighbours_test','gtoc12_outer_graph_test','gtoc12_scvx_test','-j','3'])
 core=root/'core-build/cuda/libspacepdhcg_cuda.so';qoco=Path('/home/ubuntu/spacepdhcg-diagnose-v174/final/libqoco.so')
 report['runtime_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [core,qoco]}
 env.update(PYTHONPATH=str(repo/'src'),SPACEPDHCG_GTOC12_CUDA_LIBRARY=str(core),SPACEPDHCG_QOCO_LIBRARY=str(qoco),SPACEPDHCG_GTOC12_GPU_TESTS='1',SPACEPDHCG_GTOC12_DATA=str(base/'data'),LD_LIBRARY_PATH=str(core.parent)+':'+str(qoco.parent)+':/usr/local/cuda/lib64')
 # Match the proven QOCO v174 runtime search path, without executing that runner.
 import ast
 module=ast.parse(Path('/home/ubuntu/spacepdhcg-diagnose-v174/diagnose.py').read_text())
 for node in module.body:
  if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='runtime' for t in node.targets):
   env['LD_LIBRARY_PATH']+=':'+ast.literal_eval(node.value)+'/lib'
 for flag in ['GTOC12_STATE_ORIGIN','GTOC12_DEVICE_SCHEDULING','GTOC12_DEFERRED_REPORTS','GTOC12_DEVICE_REFRESH','GTOC12_DEVICE_ASSEMBLY_VALIDATION','GTOC12_DEVICE_QUALIFICATION','QOCO_DEVICE_VALIDATION','QOCO_NATIVE_NUMERIC_REPLAY','QOCO_NATIVE_REPLAY','QOCO_IPM_GRAPH']:env['SPACEPDHCG_TEST_'+flag]='1'
 lock=open('/home/ubuntu/.spacepdhcg-gpu.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 for name in ['gtoc12_collection_test','gtoc12_neighbours_test']:
  run(name,[str(root/'core-build/cuda-tests'/name)])
 boot="import sys,runpy;sys.meta_path=[f for f in sys.meta_path if f.__class__.__module__!='_editable_skbc_spacepdhcg'];runpy.run_module('pytest',run_name='__main__')"
 run('integration',[python,'-c',boot,*[str(p.relative_to(repo)) for p in sorted((repo/'tests').glob('test_gtoc12_gpu*.py'))],'-q','--tb=short'])
 report['complete']=True
except Exception as e:report['complete']=False;report['error']=str(e)
report['end']=time.time();(root/'report.json').write_text(json.dumps(report,indent=2))
