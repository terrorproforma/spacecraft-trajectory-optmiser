from pathlib import Path
import fcntl,hashlib,json,os,shutil,subprocess,time,traceback
root=Path(__file__).resolve().parent;repo=root/'repo'
home=Path('/home/ubuntu' if Path('/home/ubuntu').exists() else '/home/angus');remote=home.name=='ubuntu'
py=str(home/'spacepdhcg/v1/.venv/bin/python') if remote else '/home/angus/worktrees/spacepdhcg-literature-venv/bin/python'
cmake=str(home/'spacepdhcg/v1/.venv/bin/cmake') if remote else '/home/angus/spacecraft-trajectory-optmiser/.venv/bin/cmake'
cuda='/usr/local/cuda' if remote else '/usr/local/cuda-12.8'
upstream=str(home/'spacepdhcg/v1/_upstream/pdhcg') if remote else '/home/angus/spacecraft-trajectory-optmiser/_upstream/pdhcg'
env={k:v for k,v in os.environ.items() if not k.startswith(('SPACEPDHCG_TEST_','QOCO_REPLAY_'))}
env.update(PYTHONPATH=str(repo/'src'),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',SPACEPDHCG_GTOC12_GPU_TESTS='1',SPACEPDHCG_GTOC12_CUDA_LIBRARY=str(root/'final/libspacepdhcg_cuda.so'),LD_LIBRARY_PATH=cuda+'/lib64')
env['SPACEPDHCG_GTOC12_DATA']=str(home/'spacepdhcg/gtoc12/benchmarks/gtoc12/data') if remote else '/home/angus/worktrees/spacepdhcg-release/benchmarks/gtoc12/data'
env['SPACEPDHCG_GTOC12_JOINT_ARCHIVE']=str(root/'incumbents')
report=dict(pid=os.getpid(),complete=False,success=False,stages=[])
def save():
    tmp=root/'report.tmp';tmp.write_text(json.dumps(report,indent=2));tmp.replace(root/'report.json')
def run(name,command):
    started=time.perf_counter()
    with (root/(name+'.log')).open('x') as log:
        child=subprocess.Popen(command,cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT)
        report.update(stage=name,child_pid=child.pid);save();code=child.wait()
    report['stages'].append(dict(name=name,code=code,seconds=time.perf_counter()-started,command=command));save()
    if code:raise RuntimeError((name,code))
save()
try:
    for name,command in [('git-init',['git','init']),('git-add',['git','add','-f','.']),('git-freeze',['git','-c','user.name=GPU validation','-c','user.email=gpu-validation@localhost','commit','-m','Freeze CUDA insertion batches'])]:run(name,command)
    run('configure',[cmake,'-S',str(repo/'cpp'),'-B',str(root/'build'),'-G','Ninja','-DCMAKE_BUILD_TYPE=Release','-DSPACEPDHCG_BUILD_CUDA=ON','-DSPACEPDHCG_BUILD_NATIVE_TESTS=OFF','-DBUILD_TESTING=ON','-DCMAKE_CUDA_COMPILER='+cuda+'/bin/nvcc','-DCMAKE_CUDA_ARCHITECTURES='+('90' if remote else '120'),'-DSPACEPDHCG_PDHCG_SOURCE_ROOT='+upstream])
    run('build',[cmake,'--build',str(root/'build'),'--target','spacepdhcg_cuda','-j3'])
    (root/'final').mkdir();shutil.copyfile(root/'build/cuda/libspacepdhcg_cuda.so',root/'final/libspacepdhcg_cuda.so')
    report['core_sha256']=hashlib.sha256((root/'final/libspacepdhcg_cuda.so').read_bytes()).hexdigest();save()
    boot="import sys;sys.meta_path=[f for f in sys.meta_path if f.__class__.__module__!='_editable_skbc_spacepdhcg'];import pytest;sys.exit(pytest.main(sys.argv[1:]))"
    with (home/'.spacepdhcg-gpu.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        run('pytest',[py,'-c',boot,'-q','tests/test_gtoc12_gpu_joint_layouts.py'])
        for mode in ('memcheck','racecheck','synccheck'):
            run(mode,[cuda+'/bin/compute-sanitizer','--tool',mode,'--error-exitcode','86',py,'-c',boot,'-q','tests/test_gtoc12_gpu_joint_layouts.py'])
    report['success']=True
except BaseException:report['error']=traceback.format_exc()
report['complete']=True;save()
