from pathlib import Path
import fcntl,hashlib,json,os,subprocess,time
root=Path('/home/angus/spacepdhcg-joint-selection-v630');source=root/'final-python-v640'
output=root/'compatibility-validation-v640';output.mkdir(exist_ok=False)
library=root/'build/cuda/libspacepdhcg_cuda.so'
env=dict(os.environ,PYTHONPATH=str(source/'src'),SPACEPDHCG_GTOC12_GPU_TESTS='1',SPACEPDHCG_GTOC12_CUDA_LIBRARY=str(library),SPACEPDHCG_GTOC12_DATA='/home/angus/worktrees/spacepdhcg-release/benchmarks/gtoc12/data',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
boot="import sys;sys.meta_path=[f for f in sys.meta_path if f.__class__.__module__!='_editable_skbc_spacepdhcg'];import pytest;raise SystemExit(pytest.main(['-q','tests/test_gtoc12_gpu_joint.py','tests/test_gtoc12_jointopt.py','tests/test_gtoc12_gpu_joint_selection.py','tests/test_gtoc12_gpu_joint_compatibility.py']))"
with open('/home/angus/.spacepdhcg-gpu.lock','a') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX)
    start=time.perf_counter()
    with (output/'pytest.log').open('x') as log:r=subprocess.run(['/home/angus/worktrees/spacepdhcg-literature-venv/bin/python','-c',boot],cwd=source,env=env,stdout=log,stderr=subprocess.STDOUT)
    report=dict(complete=True,returncode=r.returncode,seconds=time.perf_counter()-start,core_sha256=hashlib.sha256(library.read_bytes()).hexdigest(),python_sha256=hashlib.sha256((source/'src/spacepdhcg/gtoc12/gpu_joint.py').read_bytes()).hexdigest())
    (output/'report.json').write_text(json.dumps(report,indent=2));print((output/'pytest.log').read_text())
    raise SystemExit(r.returncode)
