from pathlib import Path
import fcntl
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile

root=Path('/home/angus/spacepdhcg-retry-report-v690');root.mkdir(exist_ok=False)
build=Path('/home/angus/spacepdhcg-retry-conditioning-v683')
repo=root/'repo';repo.mkdir()
for folder in ('src','tests'):shutil.copytree(build/'repo'/folder,repo/folder,ignore=shutil.ignore_patterns('__pycache__'))
for name in ['pyproject.toml','src/spacepdhcg/gtoc12/gpu_scvx.py','tests/test_gtoc12_gpu_conditioning_retry.py','scripts/gpu/prepare_qoco_retry_conditioning.py','results/local/2026-09-09/return-replay-v672/fixture.json']:
    (repo/name).parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(name,repo/name)
path=repo/'scripts/gpu/prepare_qoco_retry_conditioning.py'
spec=importlib.util.spec_from_file_location('prepare_retry',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
name='algebra/cuda/qoco_device_update.cuh'
with tempfile.TemporaryDirectory() as temporary:
    work=Path(temporary);p=work/name;p.parent.mkdir(parents=True);shutil.copyfile(Path('/home/angus/build-qoco-scaled-pool-v540/source')/name,p)
    module.prepare(work)
    assert p.read_bytes()==(build/'qoco'/name).read_bytes()
    expected=p.read_bytes()
    try:module.prepare(work)
    except RuntimeError:pass
    else:raise AssertionError('reapplication should fail')
    assert p.read_bytes()==expected
report=dict(complete=False,prepared_source_reproduced_exactly=True,reapplication_rejected_without_mutation=True,source_sha256={n:hashlib.sha256((repo/n).read_bytes()).hexdigest() for n in ['src/spacepdhcg/gtoc12/gpu_scvx.py','tests/test_gtoc12_gpu_conditioning_retry.py','scripts/gpu/prepare_qoco_retry_conditioning.py']})
(root/'report.json').write_text(json.dumps(report,indent=2))
env={k:v for k,v in os.environ.items() if not k.startswith(('SPACEPDHCG_TEST_','QOCO_REPLAY_'))}
env.update(PYTHONPATH=str(repo/'src'),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',SPACEPDHCG_GTOC12_GPU_TESTS='1',SPACEPDHCG_GTOC12_CUDA_LIBRARY=str(build/'final/libspacepdhcg_cuda.so'),SPACEPDHCG_QOCO_LIBRARY=str(build/'final/libqoco.so'),SPACEPDHCG_GTOC12_DATA='/home/angus/worktrees/spacepdhcg-release/benchmarks/gtoc12/data',LD_LIBRARY_PATH=str(build/'final')+':/home/angus/cudss-isolated-0.8.0.10/nvidia/cu12/lib:/usr/local/cuda-12.8/lib64')
boot="import sys;sys.meta_path=[f for f in sys.meta_path if f.__class__.__module__!='_editable_skbc_spacepdhcg'];import pytest;sys.exit(pytest.main(sys.argv[1:]))"
with Path('/home/angus/.spacepdhcg-gpu.lock').open('a') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX)
    with (root/'pytest.log').open('x') as log:r=subprocess.run([sys.executable,'-c',boot,'-q','tests/test_gtoc12_gpu_conditioning_retry.py'],cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=180)
report.update(complete=True,returncode=r.returncode);(root/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
