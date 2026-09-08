from pathlib import Path
import dataclasses
import fcntl
import hashlib
import json
import os
import subprocess
import time
import traceback

root=Path(__file__).resolve().parent
home=Path('/home/ubuntu' if Path('/home/ubuntu').exists() else '/home/angus')
old=home/'spacepdhcg-search-campaign-v703/worker-v706.py'
namespace={'__file__':str(old)}
exec(old.read_text().split('report=dict(')[0],namespace)
env=dict(namespace['environment'],PYTHONPATH=str(root/'repo/src'),SPACEPDHCG_GTOC12_GPU_TESTS='1',SPACEPDHCG_TEST_GTOC12_JOINT_DEVICE_SEARCH='1')
repo=root/'repo';py=namespace['python'];cuda=namespace['cuda']
report=dict(complete=False,success=False,pid=os.getpid(),stages=[],campaigns=[],core_sha256=hashlib.sha256(namespace['library'].read_bytes()).hexdigest())
def save():
    tmp=root/'report.tmp';tmp.write_text(json.dumps(report,indent=2));tmp.replace(root/'report.json')
def run(name,command,environment=env):
    started=time.perf_counter()
    with (root/(name+'.log')).open('x') as log:
        child=subprocess.Popen(command,cwd=repo,env=environment,stdout=log,stderr=subprocess.STDOUT)
        report.update(stage=name,child_pid=child.pid);save()
        code=child.wait()
    report['stages'].append(dict(name=name,returncode=code,seconds=time.perf_counter()-started,command=command));save()
    if code:raise RuntimeError((name,code))
save()
boot="import sys;sys.meta_path=[f for f in sys.meta_path if f.__class__.__module__!='_editable_skbc_spacepdhcg'];import pytest;sys.exit(pytest.main(sys.argv[1:]))"
try:
    with (home/'.spacepdhcg-gpu.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        run('pytest',[py,'-c',boot,'-q','tests/test_gtoc12_certificate_backend.py','tests/test_gtoc12_gpu_verifier.py','tests/test_gtoc12_gpu_cli.py','tests/test_gtoc12_run_final_verification.py','tests/test_gtoc12_run_refinement.py'])
        run('pipeline-gpu',[py,'-c',boot,'-q','tests/test_gtoc12_gpu_scvx.py','-k','pipeline_native_certificate or native_solution_passes_batched'])
        for mode in ('memcheck','racecheck','synccheck'):
            run(mode,[cuda+'/bin/compute-sanitizer','--tool',mode,'--error-exitcode','86',py,'-c',boot,'-q','tests/test_gtoc12_gpu_verifier.py','-k','retained_session'])
        for mode in ('cpu','auto'):
            name='campaign-'+mode
            command=[py,str(repo/'build/performance/orphan-recovery-v595/run.py'),'--repo',str(repo),'--output',str(root/name),'--wall-seconds','1800','--joint-seconds','15','--max-certifications','4','--lock',str(root/'child.lock')]
            run(name,command,dict(env,SPACEPDHCG_BENCH_CERTIFICATE_BACKEND=mode))
            data=json.loads((root/name/'report.json').read_text())
            assert data['complete'] and data['best']['independent']['ok'] and data['best']['official']['ok']
            report['campaigns'].append(dict(mode=mode,seconds=data['seconds'],best=data['best'],screening=data['screening_telemetry']));save()
    report['success']=True
except BaseException:report['error']=traceback.format_exc()
report['complete']=True;save()
