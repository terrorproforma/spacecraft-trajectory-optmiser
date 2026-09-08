from pathlib import Path
import fcntl, hashlib, json, os, subprocess, time

build = Path('/home/ubuntu/spacepdhcg-joint-selection-v632')
source = build / 'repo'
root = build / 'validation-v631'
root.mkdir(exist_ok=False)
library = build / 'build/cuda/libspacepdhcg_cuda.so'
python = '/home/ubuntu/spacepdhcg/v1/.venv/bin/python'
cuda = '/usr/local/cuda'
env = dict(os.environ, SPACEPDHCG_GTOC12_GPU_TESTS='1',
    SPACEPDHCG_GTOC12_CUDA_LIBRARY=str(library),
    SPACEPDHCG_GTOC12_DATA='/home/ubuntu/spacepdhcg/gtoc12/benchmarks/gtoc12/data',
    PYTHONPATH=str(source/'src'), OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
    LD_LIBRARY_PATH=str(library.parent)+':/home/ubuntu/spacepdhcg-scaled-pool-v545/final:/home/ubuntu/spacepdhcg-recovery-v152/cudss/lib:'+cuda+'/lib64')
report = dict(complete=False, core_sha256=hashlib.sha256(library.read_bytes()).hexdigest(), checks=[])
def run(name, command):
    start=time.perf_counter()
    with (root/(name+'.log')).open('x') as log:
        result=subprocess.run(command, cwd=source, env=env, stdout=log, stderr=subprocess.STDOUT)
    report['checks'].append(dict(name=name, command=command, returncode=result.returncode, seconds=time.perf_counter()-start))
    (root/'report.json').write_text(json.dumps(report, indent=2))
    print(name, result.returncode, flush=True)
    if result.returncode:
        print((root/(name+'.log')).read_text(), flush=True)
        raise SystemExit(result.returncode)

for name in ('smoke','selection_test'):
    run('compile-'+name,[cuda+'/bin/nvcc','-std=c++17','-arch=sm_90','--fmad=false',
        '-I'+str(source/'cpp/cuda/include'),str(source/f'cpp/cuda/tests/gtoc12_joint_{name}.cu'),
        str(library),'-Xlinker','-rpath='+str(library.parent),'-o',str(root/name)])
with open('/home/ubuntu/.spacepdhcg-gpu.lock','a') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX)
    boot="import sys;sys.meta_path=[f for f in sys.meta_path if f.__class__.__module__!='_editable_skbc_spacepdhcg'];import pytest;raise SystemExit(pytest.main(['-q','tests/test_gtoc12_gpu_joint.py','tests/test_gtoc12_jointopt.py','tests/test_gtoc12_gpu_joint_selection.py']))"
    run('pytest',[python,'-c',boot])
    for name in ('smoke','selection_test'):
        run(name,[str(root/name)])
        for tool in ('memcheck','synccheck','racecheck'):
            run(name+'-'+tool,[cuda+'/bin/compute-sanitizer','--tool',tool,'--error-exitcode','99',str(root/name)])
report['complete']=True
(root/'report.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
