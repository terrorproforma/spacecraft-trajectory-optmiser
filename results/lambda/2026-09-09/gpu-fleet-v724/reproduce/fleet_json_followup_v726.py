from pathlib import Path
import fcntl,hashlib,json,os,shutil,subprocess,time
home=Path.home();root=home/'spacepdhcg-fleet-v726';root.mkdir()
base=home/'spacepdhcg-fleet-v720';repo=root/'repo'
shutil.copytree(base/'repo',repo,ignore=shutil.ignore_patterns('.git','__pycache__','.pytest_cache'))
payload=json.loads((home/'fleet-json-overlay-v726.json').read_text())
for name,source in payload.items():(repo/name).write_text(source)
(root/'overlay.json').write_text(json.dumps(payload,indent=2))
manifest={p.relative_to(repo).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in repo.rglob('*') if p.is_file()}
(root/'source-manifest.json').write_text(json.dumps(dict(base=str(base),files=manifest),indent=2))
py=str(home/'spacepdhcg/v1/.venv/bin/python') if home.name=='ubuntu' else '/home/angus/worktrees/spacepdhcg-literature-venv/bin/python'
env=dict(os.environ,PYTHONPATH=str(repo/'src'),SPACEPDHCG_GTOC12_GPU_TESTS='1',SPACEPDHCG_GTOC12_CUDA_LIBRARY=str(base/'final/libspacepdhcg_cuda.so'),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
boot="import sys;sys.meta_path=[f for f in sys.meta_path if f.__class__.__module__!='_editable_skbc_spacepdhcg'];import pytest;sys.exit(pytest.main(sys.argv[1:]))"
cmd=[py,'-c',boot,'-q','tests/test_gtoc12_gpu_fleet.py','tests/test_gtoc12_gpu_cli.py','tests/test_gtoc12_cooperative.py','tests/test_gtoc12_collectdp.py']
report=dict(pid=os.getpid(),complete=False,success=False,command=cmd,library_sha256=hashlib.sha256((base/'final/libspacepdhcg_cuda.so').read_bytes()).hexdigest())
(root/'report.json').write_text(json.dumps(report))
with (home/'.spacepdhcg-gpu.lock').open('a') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX);start=time.perf_counter()
    with (root/'pytest.log').open('x') as log:code=subprocess.run(cmd,cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT).returncode
report.update(complete=True,success=code==0,code=code,seconds=time.perf_counter()-start)
(root/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
