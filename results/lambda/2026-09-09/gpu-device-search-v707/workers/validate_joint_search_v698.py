from pathlib import Path
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import time
import traceback

build=Path('/home/angus/spacepdhcg-joint-search-v697')
assert json.loads((build/'report.json').read_text())['success']
root=Path('/home/angus/spacepdhcg-joint-search-v698');root.mkdir(exist_ok=False)
repo=root/'repo';repo.mkdir()
for name in ('src','tests','benchmarks','results'):
    shutil.copytree(build/'repo'/name,repo/name)
shutil.copyfile(build/'repo/pyproject.toml',repo/'pyproject.toml')
names=['src/spacepdhcg/gtoc12/gpu_joint.py','src/spacepdhcg/gtoc12/jointopt.py','tests/test_gtoc12_gpu_joint_search.py']
for name in names:shutil.copyfile(name,repo/name)
env={k:v for k,v in os.environ.items() if not k.startswith(('SPACEPDHCG_TEST_','QOCO_REPLAY_'))}
env.update(PYTHONPATH=str(repo/'src'),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',SPACEPDHCG_GTOC12_CUDA_LIBRARY=str(build/'final/libspacepdhcg_cuda.so'),SPACEPDHCG_GTOC12_DATA='/home/angus/worktrees/spacepdhcg-release/benchmarks/gtoc12/data',SPACEPDHCG_GTOC12_GPU_TESTS='1',LD_LIBRARY_PATH='/usr/local/cuda-12.8/lib64')
py='/home/angus/worktrees/spacepdhcg-literature-venv/bin/python'
boot="import sys;sys.meta_path=[f for f in sys.meta_path if f.__class__.__module__!='_editable_skbc_spacepdhcg'];import pytest;sys.exit(pytest.main(sys.argv[1:]))"
tests=['tests/'+name for name in ['test_gtoc12_gpu_joint_search.py','test_gtoc12_gpu_joint_mesh.py','test_gtoc12_gpu_joint_geometry.py','test_gtoc12_gpu_joint_selection.py','test_gtoc12_gpu_joint_compatibility.py','test_gtoc12_gpu_joint.py','test_gtoc12_jointopt.py']]
report=dict(complete=False,success=False,source_sha256={name:hashlib.sha256((repo/name).read_bytes()).hexdigest() for name in names},checks=[])
def save():(root/'report.json').write_text(json.dumps(report,indent=2))
save()
try:
    with Path('/home/angus/.spacepdhcg-gpu.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        for name,paths in [('new',tests[:1]),('regression',tests[1:])]:
            cmd=[py,'-c',boot,'-q',*paths];started=time.perf_counter()
            with (root/(name+'.log')).open('x') as log:r=subprocess.run(cmd,cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=600)
            report['checks'].append(dict(name=name,returncode=r.returncode,seconds=time.perf_counter()-started,command=cmd));save();print(name,r.returncode,flush=True);r.check_returncode()
    report['success']=True
except BaseException:report['error']=traceback.format_exc()
report['complete']=True;save()
