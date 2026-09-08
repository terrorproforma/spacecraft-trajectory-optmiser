from pathlib import Path
import fcntl,json,os,subprocess,time
root=Path('build/performance/scaled-pool-scoped-sanitizer-v551');root.mkdir(exist_ok=False)
py='/home/angus/worktrees/spacepdhcg-literature-venv/bin/python'
env={k:v for k,v in os.environ.items() if not k.startswith(('SPACEPDHCG_TEST_','QOCO_REPLAY_'))}
env.update(PYTHONPATH=str(Path('src').resolve()),SPACEPDHCG_GTOC12_GPU_TESTS='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',SPACEPDHCG_GTOC12_DATA='/home/angus/worktrees/spacepdhcg-release/benchmarks/gtoc12/data')
boot="import sys;sys.meta_path=[f for f in sys.meta_path if f.__class__.__module__!='_editable_skbc_spacepdhcg'];import pytest;sys.exit(pytest.main(sys.argv[1:]))"
test='tests/test_gtoc12_gpu_scaled_workspace_pool.py::test_scaled_pool_refreshes_and_discards_failed_leg[1-lagrange-1]'
report=dict(pid=os.getpid(),complete=False,scope='cuDSS kernels excluded from instrumentation; this is not a full solver sanitizer pass',cases=[])
def save():
 t=root/'report.tmp';t.write_text(json.dumps(report,indent=2));t.replace(root/'report.json')
save()
try:
 with open('/home/angus/.spacepdhcg-gpu.lock','a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  for label in ['memcheck','synccheck','racecheck']:
   core='/home/angus/build-spacepdhcg-scaled-pool-v539/final';qoco='/home/angus/build-qoco-scaled-pool-v540/final'
   env.update(SPACEPDHCG_GTOC12_CUDA_LIBRARY=core+'/libspacepdhcg_cuda.so',SPACEPDHCG_QOCO_LIBRARY=qoco+'/libqoco.so',LD_LIBRARY_PATH=core+':'+qoco+':/home/angus/cudss-isolated-0.8.0.10/nvidia/cu12/lib:/usr/local/cuda-12.8/lib64')
   cmd=['/usr/local/cuda-12.8/bin/compute-sanitizer','--tool',label,'--kernel-name-exclude','kns=cudss','--error-exitcode','99',py,'-c',boot,test,'-x','-s','-q']
   with (root/(label+'.log')).open('x') as log:
    start=time.perf_counter();child=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT);report.update(stage=label,child_pid=child.pid);save();
    try:rc=child.wait(timeout=300)
    except subprocess.TimeoutExpired:child.kill();child.wait();rc=-999
   text=(root/(label+'.log')).read_text()
   report['cases'].append(dict(name=label,returncode=rc,seconds=time.perf_counter()-start,command=cmd,unknown_error_999='cudaErrorUnknown (error 999)' in text));save()
 report['complete']=True
except Exception as error:report['error']=repr(error)
save()
