from pathlib import Path
import base64
import hashlib
import json
import subprocess

root=Path('/home/ubuntu/spacepdhcg-retry-conditioning-v686')
worker='''from pathlib import Path
import fcntl,hashlib,json,os,subprocess,time,traceback,sys
root=Path('/home/ubuntu/spacepdhcg-retry-conditioning-v686');repo=root/'repo';out=root/'broader-v689';out.mkdir(exist_ok=False)
env={k:v for k,v in os.environ.items() if not k.startswith(('SPACEPDHCG_TEST_','QOCO_REPLAY_'))}
env.update(PYTHONPATH=str(repo/'src'),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',SPACEPDHCG_GTOC12_GPU_TESTS='1',SPACEPDHCG_GTOC12_CUDA_LIBRARY=str(root/'final/libspacepdhcg_cuda.so'),SPACEPDHCG_QOCO_LIBRARY=str(root/'final/libqoco.so'),SPACEPDHCG_GTOC12_DATA='/home/ubuntu/spacepdhcg/gtoc12/benchmarks/gtoc12/data',LD_LIBRARY_PATH=str(root/'final')+':/home/ubuntu/spacepdhcg-recovery-v152/cudss/lib:/usr/local/cuda/lib64')
old=Path('/home/ubuntu/spacepdhcg-preserve-objective-v535/final/libqoco.so')
if old.is_file():env['SPACEPDHCG_GTOC12_OLD_QOCO_LIBRARY']=str(old)
report=dict(complete=False,stages=[],old_library_available=old.is_file())
def save(): (out/'report.json').write_text(json.dumps(report,indent=2))
def run(name,cmd,flags=None):
    started=time.perf_counter()
    with (out/(name+'.log')).open('x') as log:
        child=subprocess.Popen(cmd,cwd=repo,env=dict(env,**(flags or {})),stdout=log,stderr=subprocess.STDOUT)
        report.update(stage=name,child_pid=child.pid);save();code=child.wait(timeout=1200)
    report['stages'].append(dict(name=name,returncode=code,seconds=time.perf_counter()-started,command=cmd));save()
    if code:raise RuntimeError(name)
save()
try:
    sys.path.insert(0,str(root));from audit_helper import problem,audit
    data=problem(root/'input-qp.txt');records=[json.loads(line[10:]) for line in (root/'qp-replay.log').read_text().splitlines() if line.startswith('QP_REPLAY ')]
    report['qp_audits']=[dict(mode=r['repeat']%3,status=r['status'],iterations=r['iterations'],**audit(data,r)) for r in records];save()
    tests=['tests/'+name for name in ['test_gtoc12_gpu_scaled_workspace_pool.py','test_gtoc12_gpu_retained_replay.py','test_gtoc12_gpu_workspace_pool.py','test_gtoc12_verifier_knots.py','test_gtoc12_verifier.py','test_gtoc12_gpu_verifier.py','test_gtoc12_gpu_resident_options.py','test_gtoc12_gpu_collection.py','test_gtoc12_gpu_elements.py','test_gtoc12_gpu_scvx.py','test_gtoc12_gpu_cli.py','test_gtoc12_run_final_verification.py']]
    boot="import sys;sys.meta_path=[f for f in sys.meta_path if f.__class__.__module__!='_editable_skbc_spacepdhcg'];import pytest;sys.exit(pytest.main(sys.argv[1:]))"
    replay="import sys,runpy;sys.meta_path=[f for f in sys.meta_path if f.__class__.__module__!='_editable_skbc_spacepdhcg'];sys.argv=sys.argv[1:];runpy.run_path(sys.argv[0],run_name='__main__')"
    with Path('/home/ubuntu/.spacepdhcg-gpu.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        run('regression',[sys.executable,'-c',boot,'-q',*tests],{'SPACEPDHCG_TEST_QOCO_RUIZ_PRESERVE_OBJECTIVE':'1','SPACEPDHCG_TEST_QOCO_RETAINED_REPLAY':'1'})
        for name,mode in [('baseline','0'),('candidate','1')]:run(name,[sys.executable,'-c',replay,str(root/'legs-run.py'),mode,str(out/name)])
except BaseException:report['error']=traceback.format_exc()
report['complete']=True;save()
'''
payloads={'legs-run.py':Path('/home/angus/spacepdhcg-retry-fleet-legs-v688/run.py').read_bytes(),'repo/build/performance/grid-cache-fleet-v403/scvx-calls.json':Path('build/performance/grid-cache-fleet-v403/scvx-calls.json').read_bytes()}
launch="from pathlib import Path\nimport base64,hashlib,json,subprocess\nroot=Path('/home/ubuntu/spacepdhcg-retry-conditioning-v686')\n"
for name,raw in payloads.items():
    launch+='p=root/'+repr(name)+';p.parent.mkdir(parents=True,exist_ok=True)\n'
    launch+='raw=base64.b64decode('+repr(base64.b64encode(raw).decode())+')\n'
    launch+='assert not p.exists()\np.write_bytes(raw)\nassert hashlib.sha256(p.read_bytes()).hexdigest()=='+repr(hashlib.sha256(raw).hexdigest())+'\n'
launch+="(root/'broader-worker.py').write_text("+repr(worker)+")\nwith (root/'broader-worker.log').open('x') as log:child=subprocess.Popen(['/home/ubuntu/spacepdhcg/v1/.venv/bin/python',str(root/'broader-worker.py')],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)\nprint(json.dumps(dict(pid=child.pid,root=str(root/'broader-v689'))))\n"
Path('build/performance/launch_retry_h100_v689.py').write_text(launch)
subprocess.run(['python3','build/performance/remote_exec.py','build/performance/launch_retry_h100_v689.py'],check=True)
