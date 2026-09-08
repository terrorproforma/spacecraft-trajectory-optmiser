from pathlib import Path
import fcntl,hashlib,json,os,subprocess,time
root=Path('/home/angus/spacepdhcg-joint-mesh-v662');source=root/'repo';output=root/'validation-v665';output.mkdir(exist_ok=False)
library=root/'build/cuda/libspacepdhcg_cuda.so';python='/home/angus/worktrees/spacepdhcg-literature-venv/bin/python'
env={k:v for k,v in os.environ.items() if not k.startswith('SPACEPDHCG_TEST_')}
env.update(PYTHONPATH=str(source/'src'),SPACEPDHCG_GTOC12_GPU_TESTS='1',SPACEPDHCG_GTOC12_CUDA_LIBRARY=str(library),SPACEPDHCG_GTOC12_DATA='/home/angus/worktrees/spacepdhcg-release/benchmarks/gtoc12/data',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
report=dict(complete=False,success=False,core_sha256=hashlib.sha256(library.read_bytes()).hexdigest(),checks=[])
tests=['tests/test_gtoc12_gpu_joint.py','tests/test_gtoc12_jointopt.py','tests/test_gtoc12_gpu_joint_selection.py','tests/test_gtoc12_gpu_joint_compatibility.py','tests/test_gtoc12_gpu_joint_geometry.py','tests/test_gtoc12_gpu_joint_mesh.py']
def command(paths):
    boot="import sys;sys.meta_path=[f for f in sys.meta_path if f.__class__.__module__!='_editable_skbc_spacepdhcg'];import pytest;raise SystemExit(pytest.main("+repr(['-q',*paths])+"))"
    return [python,'-c',boot]
def run(name,cmd):
    start=time.perf_counter()
    with (output/(name+'.log')).open('x') as log:r=subprocess.run(cmd,cwd=source,env=env,stdout=log,stderr=subprocess.STDOUT)
    report['checks'].append(dict(name=name,command=cmd,returncode=r.returncode,seconds=time.perf_counter()-start));(output/'report.json').write_text(json.dumps(report,indent=2));print(name,r.returncode,flush=True)
    if r.returncode:print((output/(name+'.log')).read_text(),flush=True);raise SystemExit(r.returncode)
with open('/home/angus/.spacepdhcg-gpu.lock','a') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX)
    run('pytest',command(tests))
    for sanitizer in ('memcheck','synccheck','racecheck'):
        run(sanitizer,['/usr/local/cuda-12.8/bin/compute-sanitizer','--tool',sanitizer,'--target-processes','all','--error-exitcode','99',*command(tests[-2:])])
report.update(complete=True,success=True);(output/'report.json').write_text(json.dumps(report,indent=2))
