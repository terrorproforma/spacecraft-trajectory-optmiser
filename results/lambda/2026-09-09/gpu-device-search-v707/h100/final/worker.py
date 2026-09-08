from pathlib import Path
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import traceback

root=Path(__file__).resolve().parent;repo=root/'repo'
remote=Path('/home/ubuntu').exists()
home=Path('/home/ubuntu' if remote else '/home/angus')
py=str(home/'spacepdhcg/v1/.venv/bin/python') if remote else '/home/angus/worktrees/spacepdhcg-literature-venv/bin/python'
cuda='/usr/local/cuda' if remote else '/usr/local/cuda-12.8'
env={k:v for k,v in os.environ.items() if not k.startswith(('SPACEPDHCG_TEST_','QOCO_REPLAY_'))}
env.update(PYTHONPATH=str(repo/'src'),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',SPACEPDHCG_GTOC12_CUDA_LIBRARY=str(root/'final/libspacepdhcg_cuda.so'),SPACEPDHCG_GTOC12_DATA=str(home/'spacepdhcg/gtoc12/benchmarks/gtoc12/data') if remote else '/home/angus/worktrees/spacepdhcg-release/benchmarks/gtoc12/data',SPACEPDHCG_GTOC12_GPU_TESTS='1',LD_LIBRARY_PATH=cuda+'/lib64')
report=dict(complete=False,success=False,pid=os.getpid(),stages=[])
def save():(root/'report.json').write_text(json.dumps(report,indent=2))
def run(name,cmd,timeout=600):
    started=time.perf_counter()
    with (root/(name+'.log')).open('x') as log:
        child=subprocess.Popen(cmd,cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT)
        report.update(stage=name,child_pid=child.pid);save()
        try:code=child.wait(timeout=timeout)
        except subprocess.TimeoutExpired:child.kill();child.wait();raise
    report['stages'].append(dict(name=name,returncode=code,seconds=time.perf_counter()-started,command=cmd));save();print(name,code,flush=True)
    if code:raise RuntimeError((name,code))
save()
try:
    run('git-init',['git','init']);run('git-add',['git','add','-f','.']);run('git-freeze',['git','-c','user.name=GPU validation','-c','user.email=gpu-validation@localhost','commit','-m','Frozen warp-reduced device epoch search'])
    if True:
        cmake=str(home/'spacepdhcg/v1/.venv/bin/cmake') if remote else '/home/angus/spacecraft-trajectory-optmiser/.venv/bin/cmake'
        run('configure',[cmake,'-S',str(repo/'cpp'),'-B',str(root/'build'),'-G','Ninja','-DCMAKE_BUILD_TYPE=Release','-DSPACEPDHCG_BUILD_CUDA=ON','-DSPACEPDHCG_BUILD_NATIVE_TESTS=OFF','-DBUILD_TESTING=ON','-DCMAKE_CUDA_COMPILER='+cuda+'/bin/nvcc','-DCMAKE_CUDA_ARCHITECTURES='+('90' if remote else '120'),'-DSPACEPDHCG_PDHCG_SOURCE_ROOT='+(str(home/'spacepdhcg/v1/_upstream/pdhcg') if remote else '/home/angus/spacecraft-trajectory-optmiser/_upstream/pdhcg')])
        run('build',[cmake,'--build',str(root/'build'),'--target','spacepdhcg_cuda','-j3'])
        (root/'final').mkdir();shutil.copyfile(root/'build/cuda/libspacepdhcg_cuda.so',root/'final/libspacepdhcg_cuda.so')
    report['core_sha256']=hashlib.sha256((root/'final/libspacepdhcg_cuda.so').read_bytes()).hexdigest();save()
    boot="import sys;sys.meta_path=[f for f in sys.meta_path if f.__class__.__module__!='_editable_skbc_spacepdhcg'];import pytest;sys.exit(pytest.main(sys.argv[1:]))"
    tests=['tests/test_gtoc12_gpu_joint'+suffix+'.py' for suffix in ('_search','_mesh','_geometry','_selection','_compatibility','')]+['tests/test_gtoc12_jointopt.py']
    with (home/'.spacepdhcg-gpu.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        run('pytest',[py,'-c',boot,'-q',*tests])
        for mode in ('memcheck','racecheck','synccheck'):
            run(mode,[cuda+'/bin/compute-sanitizer','--tool',mode,'--error-exitcode','86',py,'-c',boot,'-q',tests[0],'-k','whole_search and moves'])
        run('benchmark',[py,str(root/'benchmark.py'),str(root/'benchmark.json')])
    report['success']=True
except BaseException:report['error']=traceback.format_exc()
report['complete']=True;save()
