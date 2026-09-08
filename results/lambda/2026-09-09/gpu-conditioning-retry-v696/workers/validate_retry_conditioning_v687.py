from pathlib import Path
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import time

root=Path('/home/angus/spacepdhcg-retry-conditioning-v683');repo=root/'repo'
out=root/'validation-v687';out.mkdir(exist_ok=False)
for name in ['tests/test_gtoc12_gpu_conditioning_retry.py','results/local/2026-09-09/return-replay-v672/fixture.json']:
    (repo/name).parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(name,repo/name)
env={k:v for k,v in os.environ.items() if not k.startswith(('SPACEPDHCG_TEST_','QOCO_REPLAY_'))}
env.update(PYTHONPATH=str(repo/'src'),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',SPACEPDHCG_GTOC12_CUDA_LIBRARY=str(root/'final/libspacepdhcg_cuda.so'),SPACEPDHCG_QOCO_LIBRARY=str(root/'final/libqoco.so'),SPACEPDHCG_GTOC12_DATA='/home/angus/worktrees/spacepdhcg-release/benchmarks/gtoc12/data',LD_LIBRARY_PATH=str(root/'final')+':/home/angus/cudss-isolated-0.8.0.10/nvidia/cu12/lib:/usr/local/cuda-12.8/lib64',SPACEPDHCG_GTOC12_GPU_TESTS='1',SPACEPDHCG_GTOC12_OLD_QOCO_LIBRARY='/home/angus/build-qoco-preserve-objective-v534/final/libqoco.so')
py='/home/angus/worktrees/spacepdhcg-literature-venv/bin/python'
boot="import sys;sys.meta_path=[f for f in sys.meta_path if f.__class__.__module__!='_editable_skbc_spacepdhcg'];import pytest;sys.exit(pytest.main(sys.argv[1:]))"
tests=['tests/'+name for name in ['test_gtoc12_gpu_scaled_workspace_pool.py','test_gtoc12_gpu_retained_replay.py','test_gtoc12_gpu_workspace_pool.py','test_gtoc12_verifier_knots.py','test_gtoc12_verifier.py','test_gtoc12_gpu_verifier.py','test_gtoc12_gpu_resident_options.py','test_gtoc12_gpu_collection.py','test_gtoc12_gpu_elements.py','test_gtoc12_gpu_scvx.py','test_gtoc12_gpu_cli.py','test_gtoc12_run_final_verification.py']]
report=dict(complete=False,checks=[],test_sha256=hashlib.sha256((repo/'tests/test_gtoc12_gpu_conditioning_retry.py').read_bytes()).hexdigest())
def save(): (out/'report.json').write_text(json.dumps(report,indent=2))
save()
try:
    with Path('/home/angus/.spacepdhcg-gpu.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        for name,paths,flags in [('new',['tests/test_gtoc12_gpu_conditioning_retry.py'],{}),('regression',tests,{'SPACEPDHCG_TEST_QOCO_RUIZ_PRESERVE_OBJECTIVE':'1','SPACEPDHCG_TEST_QOCO_RETAINED_REPLAY':'1'})]:
            started=time.perf_counter();cmd=[py,'-c',boot,'-q',*paths]
            with (out/(name+'.log')).open('x') as log:r=subprocess.run(cmd,cwd=repo,env=dict(env,**flags),stdout=log,stderr=subprocess.STDOUT,timeout=600)
            report['checks'].append(dict(name=name,returncode=r.returncode,seconds=time.perf_counter()-started,command=cmd,flags=flags));save();print(name,r.returncode,flush=True)
    report['complete']=True
finally:save()
