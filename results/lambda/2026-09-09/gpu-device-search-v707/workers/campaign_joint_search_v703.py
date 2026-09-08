from pathlib import Path
import fcntl
import hashlib
import json
import os
import subprocess
import time
import traceback

root=Path(__file__).resolve().parent;source=root/'repo'
remote=Path('/home/ubuntu').exists();home=Path('/home/ubuntu' if remote else '/home/angus')
build=home/'spacepdhcg-joint-search-v702'
assert json.loads((build/'report.json').read_text())['success']
library=build/'final/libspacepdhcg_cuda.so'
qoco=home/('spacepdhcg-retry-conditioning-v686/final/libqoco.so' if remote else 'spacepdhcg-retry-conditioning-v683/final/libqoco.so')
python=str(home/'spacepdhcg/v1/.venv/bin/python') if remote else '/home/angus/worktrees/spacepdhcg-literature-venv/bin/python'
environment={k:v for k,v in os.environ.items() if not k.startswith(('SPACEPDHCG_TEST_','QOCO_REPLAY_'))}
cudss=str(home/'spacepdhcg-recovery-v152/cudss/lib') if remote else '/home/angus/cudss-isolated-0.8.0.10/nvidia/cu12/lib'
cuda='/usr/local/cuda' if remote else '/usr/local/cuda-12.8'
environment.update(SPACEPDHCG_GTOC12_CUDA_LIBRARY=str(library),SPACEPDHCG_QOCO_LIBRARY=str(qoco),SPACEPDHCG_GTOC12_DATA=str(home/'spacepdhcg/gtoc12/benchmarks/gtoc12/data') if remote else '/home/angus/worktrees/spacepdhcg-release/benchmarks/gtoc12/data',PYTHONPATH=str(source/'src'),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',SPACEPDHCG_TEST_GTOC12_JOINT_BATCH='1',SPACEPDHCG_TEST_GTOC12_JOINT_DEVICE_SELECTION='1',SPACEPDHCG_TEST_GTOC12_JOINT_RESIDENT_GEOMETRY='1',SPACEPDHCG_TEST_GTOC12_JOINT_DEVICE_MESH='1',SPACEPDHCG_TEST_GTOC12_CONDITIONING_RETRY='1',LD_LIBRARY_PATH=str(library.parent)+':'+str(qoco.parent)+':'+cudss+':'+cuda+'/lib64')
report=dict(complete=False,success=False,pid=os.getpid(),core_sha256=hashlib.sha256(library.read_bytes()).hexdigest(),qoco_sha256=hashlib.sha256(qoco.read_bytes()).hexdigest(),runs=[])
def save():
    temp=root/'report.tmp';temp.write_text(json.dumps(report,indent=2));temp.replace(root/'report.json')
save()
try:
    for name,mode in (('baseline0',0),('candidate0',1)):
        env=dict(environment,SPACEPDHCG_TEST_GTOC12_JOINT_DEVICE_SEARCH=str(mode))
        command=[python,str(source/'build/performance/orphan-recovery-v595/run.py'),'--repo',str(source),'--output',str(root/name),'--wall-seconds','1800','--joint-seconds','15','--max-certifications','4','--lock',str(root/'child.lock')]
        with (home/'.spacepdhcg-gpu.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX);started=time.perf_counter()
            with (root/(name+'.log')).open('x') as log:
                child=subprocess.Popen(command,cwd=source,env=env,stdout=log,stderr=subprocess.STDOUT)
                report.update(stage=name,child_pid=child.pid);save();code=child.wait()
            assert code==0,(name,code)
            run=json.loads((root/name/'report.json').read_text())
            assert run['complete'] and run['best']['official']['ok'] and run['best']['independent']['ok']
            solves=[json.loads(line) for line in (root/name/'native-solves.jsonl').read_text().splitlines()]
            report['runs'].append(dict(name=name,device_search=mode,command=command,process_seconds=time.perf_counter()-started,campaign_seconds=run['seconds'],status=run['status'],orders=run['orders'],best=run['best'],screening=run['screening_telemetry'],native_solves=len(solves),native_seconds=sum(r['seconds'] for r in solves),nonconverged=[dict(solve=r['solve'],status=r['status']) for r in solves if r['status']!='converged']));save()
    assert report['runs'][1]['best']['score_kg']>=report['runs'][0]['best']['score_kg']-1e-6
    report['success']=True
except BaseException:report['error']=traceback.format_exc()
report['complete']=True;save();print(json.dumps(report),flush=True)
