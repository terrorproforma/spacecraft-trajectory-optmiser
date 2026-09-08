from pathlib import Path
import fcntl,hashlib,json,os,subprocess,time,traceback
build=Path('/home/angus/spacepdhcg-joint-selection-v630');source=build/'repo'
root=build/'campaign-v639';root.mkdir(exist_ok=False)
library=build/'build/cuda/libspacepdhcg_cuda.so'
qoco=Path('/home/angus/build-qoco-scaled-pool-v540/final/libqoco.so')
python='/home/angus/worktrees/spacepdhcg-literature-venv/bin/python'
environment={k:v for k,v in os.environ.items() if not k.startswith(('SPACEPDHCG_TEST_','QOCO_REPLAY_'))}
environment.update(SPACEPDHCG_GTOC12_CUDA_LIBRARY=str(library),SPACEPDHCG_QOCO_LIBRARY=str(qoco),SPACEPDHCG_GTOC12_DATA='/home/angus/worktrees/spacepdhcg-release/benchmarks/gtoc12/data',PYTHONPATH=str(source/'src'),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',SPACEPDHCG_TEST_GTOC12_JOINT_BATCH='1',LD_LIBRARY_PATH=str(library.parent)+':'+str(qoco.parent)+':/home/angus/cudss-isolated-0.8.0.10/nvidia/cu12/lib:/usr/local/cuda-12.8/lib64')
report=dict(complete=False,success=False,pid=os.getpid(),core_sha256=hashlib.sha256(library.read_bytes()).hexdigest(),qoco_sha256=hashlib.sha256(qoco.read_bytes()).hexdigest(),runs=[])
def save():
    path=root/'report.tmp';path.write_text(json.dumps(report,indent=2));path.replace(root/'report.json')
try:
    for name,mode in (('baseline-retry',0),):
        env=dict(environment,SPACEPDHCG_TEST_GTOC12_JOINT_DEVICE_SELECTION=str(mode))
        command=[python,str(source/'build/performance/orphan-recovery-v595/run.py'),'--repo',str(source),'--output',str(root/name),'--wall-seconds','1800','--joint-seconds','15','--max-certifications','4','--lock',str(root/'child.lock')]
        with open('/home/angus/.spacepdhcg-gpu.lock','a') as shared_lock:
            fcntl.flock(shared_lock,fcntl.LOCK_EX)
            start=time.perf_counter()
            with (root/(name+'.log')).open('x') as log:
                child=subprocess.Popen(command,cwd=source,env=env,stdout=log,stderr=subprocess.STDOUT)
                report.update(stage=name,child_pid=child.pid);save()
                code=child.wait()
            assert code==0,(name,code)
            run=json.loads((root/name/'report.json').read_text())
            assert run['complete'] and run['best']['ok'],name
            report['runs'].append(dict(name=name,selection=mode,command=command,process_seconds=time.perf_counter()-start,campaign_seconds=run['seconds'],status=run['status'],best=run['best'],screening=run['screening_telemetry'],native_solves=run['native_solves_completed']));save()
    report['success']=True
except BaseException:report['exception']=traceback.format_exc()
report['complete']=True;save()
print(json.dumps(report,indent=2))
