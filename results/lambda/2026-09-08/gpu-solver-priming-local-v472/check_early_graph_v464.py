from pathlib import Path
import os,subprocess,fcntl,json,time,shutil,hashlib
root=Path('build/performance/early-graph-v464');root.mkdir(exist_ok=False)
frozen=Path('/home/angus/build-spacepdhcg-early-graph-v465/final');core=frozen/'libspacepdhcg_cuda.so'
sources=['cpp/cuda/src/native_qoco_adapter.cpp','tests/test_gtoc12_gpu_scvx.py','cpp/cuda/src/gtoc12_scvx.cu','cpp/cuda/include/spacepdhcg/cuda/gtoc12_scvx_c_api.h','cpp/cuda/tests/gtoc12_scvx_test.cu','src/spacepdhcg/gtoc12/gpu_scvx.py']
for name in sources:
 dest=root/'source'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(name,dest)
qoco=Path('/home/angus/build-qoco-soc-step-v358/final/libqoco.so')
env={k:v for k,v in os.environ.items() if not k.startswith(('SPACEPDHCG_TEST_','QOCO_REPLAY_'))}
env['SPACEPDHCG_TEST_QOCO_EARLY_OUTER_GRAPH']='1'
env.update(PYTHONPATH=str(Path('src').resolve()),SPACEPDHCG_GTOC12_CUDA_LIBRARY=str(core),SPACEPDHCG_QOCO_LIBRARY=str(qoco),SPACEPDHCG_GTOC12_GPU_TESTS='1',SPACEPDHCG_GTOC12_DATA='/home/angus/worktrees/spacepdhcg-release/benchmarks/gtoc12/data',LD_LIBRARY_PATH=str(core.parent)+':'+str(qoco.parent)+':/home/angus/cudss-isolated-0.8.0.10/nvidia/cu12/lib:/usr/local/cuda-12.8/lib64',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
binary='/home/angus/early-graph-probe-v464'
compile_cmd=['/usr/local/cuda-12.8/bin/nvcc','-std=c++17','--fmad=false','-arch=sm_120','-Icpp/include','-Icpp/cuda/include','cpp/cuda/tests/gtoc12_scvx_test.cu','-L'+str(core.parent),'-lspacepdhcg_cuda','-o',binary]
py='/home/angus/worktrees/spacepdhcg-literature-venv/bin/python'
boot="import sys;sys.meta_path=[f for f in sys.meta_path if f.__class__.__module__!='_editable_skbc_spacepdhcg'];import pytest;sys.exit(pytest.main(sys.argv[1:]))"
jobs=[('compile',compile_cmd),('probe',[binary])]+[(name,['/usr/local/cuda-12.8/bin/compute-sanitizer','--tool',name,'--error-exitcode','99',binary]) for name in ['memcheck','synccheck','racecheck']]+[('pytest',[py,'-c',boot,'tests/test_gtoc12_gpu_scvx.py','tests/test_gtoc12_gpu_cli.py','tests/test_gtoc12_run_final_verification.py','-q'])]
r=dict(pid=os.getpid(),complete=False,stages=[],core_sha256=hashlib.sha256(core.read_bytes()).hexdigest(),qoco_sha256=hashlib.sha256(qoco.read_bytes()).hexdigest(),source_sha256={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sources})
def save():(root/'report.json').write_text(json.dumps(r,indent=2))
save()
try:
 with open('/home/angus/.spacepdhcg-gpu.lock','a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  for name,cmd in jobs:
   start=time.perf_counter()
   with (root/(name+'.log')).open('x') as log:
    child=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT);r.update(stage=name,child_pid=child.pid);save()
    try:code=child.wait(timeout=300)
    except subprocess.TimeoutExpired:child.kill();child.wait();raise
   r['stages'].append(dict(name=name,code=code,seconds=time.perf_counter()-start,command=cmd));save();assert code==0,(name,code)
 r['complete']=True
except Exception as e:r['error']=repr(e)
save();print(json.dumps(r,indent=2))
