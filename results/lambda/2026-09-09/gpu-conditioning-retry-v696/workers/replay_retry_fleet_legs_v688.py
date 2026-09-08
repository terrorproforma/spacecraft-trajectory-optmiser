from pathlib import Path
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time

build=Path('/home/angus/spacepdhcg-retry-conditioning-v683')
root=Path('/home/angus/spacepdhcg-retry-fleet-legs-v688');root.mkdir(exist_ok=False)
fixture='build/performance/grid-cache-fleet-v403/scvx-calls.json'
target=build/'repo'/fixture;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(fixture,target)
source=Path('build/performance/replay_scaled_pool.py').read_text()
source=source.replace("os.environ['SPACEPDHCG_TEST_GTOC12_SCALED_QOCO_POOL']=mode","os.environ['SPACEPDHCG_TEST_GTOC12_CONDITIONING_RETRY']=mode")
source=source.replace("os.environ['SPACEPDHCG_TEST_QOCO_RUIZ_PRESERVE_OBJECTIVE']='1'","os.environ['SPACEPDHCG_TEST_QOCO_RUIZ_PRESERVE_OBJECTIVE']='0'")
source=source.replace('qoco_ruiz_iterations=2','qoco_ruiz_iterations=0')
(root/'run.py').write_text(source)
env={k:v for k,v in os.environ.items() if not k.startswith(('SPACEPDHCG_TEST_','QOCO_REPLAY_'))}
env.update(PYTHONPATH=str(build/'repo/src'),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',SPACEPDHCG_GTOC12_CUDA_LIBRARY=str(build/'final/libspacepdhcg_cuda.so'),SPACEPDHCG_QOCO_LIBRARY=str(build/'final/libqoco.so'),SPACEPDHCG_GTOC12_DATA='/home/angus/worktrees/spacepdhcg-release/benchmarks/gtoc12/data',LD_LIBRARY_PATH=str(build/'final')+':/home/angus/cudss-isolated-0.8.0.10/nvidia/cu12/lib:/usr/local/cuda-12.8/lib64')
boot="import sys,runpy;sys.meta_path=[f for f in sys.meta_path if f.__class__.__module__!='_editable_skbc_spacepdhcg'];sys.argv=sys.argv[1:];runpy.run_path(sys.argv[0],run_name='__main__')"
report=dict(complete=False,fixture_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),driver_sha256=hashlib.sha256((root/'run.py').read_bytes()).hexdigest(),stages=[])
def save(): (root/'report.json').write_text(json.dumps(report,indent=2))
save()
try:
    with Path('/home/angus/.spacepdhcg-gpu.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        for name,mode in [('baseline','0'),('candidate','1')]:
            start=time.perf_counter();command=[sys.executable,'-c',boot,str(root/'run.py'),mode,str(root/name)]
            with (root/(name+'.log')).open('x') as log:
                child=subprocess.Popen(command,cwd=build/'repo',env=env,stdout=log,stderr=subprocess.STDOUT)
                report.update(stage=name,child_pid=child.pid);save();code=child.wait(timeout=1200)
            report['stages'].append(dict(name=name,returncode=code,seconds=time.perf_counter()-start,command=command));save()
            if code:break
    report['complete']=True
finally:save()
print(json.dumps(report))
