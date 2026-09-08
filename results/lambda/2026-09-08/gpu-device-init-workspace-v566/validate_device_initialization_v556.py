from pathlib import Path
import fcntl,hashlib,json,os,shutil,subprocess,time
root=Path('build/performance/device-init-v556');root.mkdir(exist_ok=False)
core=Path('/home/angus/build-spacepdhcg-device-init-v555/final/libspacepdhcg_cuda.so')
qoco=Path('/home/angus/build-qoco-scaled-pool-v540/final/libqoco.so')
py='/home/angus/worktrees/spacepdhcg-literature-venv/bin/python'
env={k:v for k,v in os.environ.items() if not k.startswith(('SPACEPDHCG_TEST_','QOCO_REPLAY_'))}
env.update(PYTHONPATH=str(Path('src').resolve()),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',SPACEPDHCG_GTOC12_CUDA_LIBRARY=str(core),SPACEPDHCG_QOCO_LIBRARY=str(qoco),SPACEPDHCG_GTOC12_DATA='/home/angus/worktrees/spacepdhcg-release/benchmarks/gtoc12/data',LD_LIBRARY_PATH=str(core.parent)+':'+str(qoco.parent)+':/home/angus/cudss-isolated-0.8.0.10/nvidia/cu12/lib:/usr/local/cuda-12.8/lib64',SPACEPDHCG_GTOC12_GPU_TESTS='1',SPACEPDHCG_TEST_QOCO_RUIZ_PRESERVE_OBJECTIVE='1',SPACEPDHCG_TEST_QOCO_RETAINED_REPLAY='1',SPACEPDHCG_GTOC12_OLD_QOCO_LIBRARY='/home/angus/build-qoco-preserve-objective-v534/final/libqoco.so')
env['SPACEPDHCG_TEST_QOCO_DEVICE_INITIALIZATION']='1'
report=dict(pid=os.getpid(),complete=False,stages=[],source_sha256={},runtime_sha256={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in [core,qoco]})
def save():
 t=root/'report.tmp';t.write_text(json.dumps(report,indent=2));t.replace(root/'report.json')
def run(name,cmd,timeout=1800):
 start=time.perf_counter()
 with (root/(name+'.log')).open('x') as log:
  child=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT);report.update(stage=name,child_pid=child.pid);save();rc=child.wait(timeout=timeout)
 report['stages'].append(dict(name=name,returncode=rc,seconds=time.perf_counter()-start,command=cmd));save();assert rc==0,(name,rc)
save()
try:
 files=['tests/test_gtoc12_gpu_device_initialization.py','cpp/cuda/src/gtoc12_qoco.cu','cpp/cuda/src/gtoc12_scvx.cu','cpp/cuda/src/native_qoco_adapter.cpp','scripts/gpu/prepare_qoco_preserve_objective.py','tests/test_gtoc12_gpu_scaled_workspace_pool.py','build/performance/replay_device_initialization.py','build/performance/validate_device_initialization_v556.py','build/performance/grid-cache-fleet-v403/scvx-calls.json']
 files += [str(q) for q in Path('src/spacepdhcg/gtoc12').glob('*.py')]
 for name in files:
  dest=root/'source'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(name,dest);report['source_sha256'][name]=hashlib.sha256(Path(name).read_bytes()).hexdigest()
 boot="import sys;sys.meta_path=[f for f in sys.meta_path if f.__class__.__module__!='_editable_skbc_spacepdhcg'];import pytest;sys.exit(pytest.main(sys.argv[1:]))"
 tests=['tests/'+name for name in ['test_gtoc12_gpu_device_initialization.py','test_gtoc12_gpu_qoco.py','test_gtoc12_gpu_scaled_workspace_pool.py','test_gtoc12_gpu_retained_replay.py','test_gtoc12_gpu_workspace_pool.py','test_gtoc12_verifier_knots.py','test_gtoc12_verifier.py','test_gtoc12_gpu_verifier.py','test_gtoc12_gpu_resident_options.py','test_gtoc12_gpu_collection.py','test_gtoc12_gpu_elements.py','test_gtoc12_gpu_scvx.py','test_gtoc12_gpu_cli.py','test_gtoc12_run_final_verification.py']]
 for name in tests:
  dest=root/'source'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(name,dest)
 with open('/home/angus/.spacepdhcg-gpu.lock','a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  run('pytest',[py,'-c',boot,*tests,'-x','-s','-q'])
  replay="import sys,runpy;sys.meta_path=[f for f in sys.meta_path if f.__class__.__module__!='_editable_skbc_spacepdhcg'];runpy.run_path('build/performance/replay_device_initialization.py',run_name='__main__')"
  for label,mode in [('baseline','0'),('candidate','1')]:run(label,[py,'-c',replay,mode,str(root/label)])
 report['complete']=True
except Exception as error:report['error']=repr(error)
save();print(json.dumps(report,indent=2))
