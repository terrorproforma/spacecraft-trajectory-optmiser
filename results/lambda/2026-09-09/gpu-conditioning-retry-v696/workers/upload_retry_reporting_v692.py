from pathlib import Path
import hashlib
import json
import os
import subprocess
import tarfile

local=Path('/home/angus/spacepdhcg-retry-report-v690/repo')
build=Path('/home/angus/spacepdhcg-retry-conditioning-v683')
files=['pyproject.toml','src/spacepdhcg/gtoc12/gpu_scvx.py','tests/test_gtoc12_gpu_conditioning_retry.py','scripts/gpu/prepare_qoco_retry_conditioning.py','results/local/2026-09-09/return-replay-v672/fixture.json',*json.loads((build/'campaign-fixtures.json').read_text())]
archive=Path('/tmp/retry-reporting-v692.tar.gz');manifest={}
with tarfile.open(archive,'w:gz') as tar:
    for name in sorted(set(files)):
        p=local/name;manifest[name]=hashlib.sha256(p.read_bytes()).hexdigest();tar.add(p,arcname=name)
runner=(build/'campaign-runner-v691.py').read_text()
mapping={'/home/angus/spacepdhcg-retry-conditioning-v683':'/home/ubuntu/spacepdhcg-retry-conditioning-v686','/home/angus/spacepdhcg-retry-report-v690/repo':'/home/ubuntu/spacepdhcg-retry-conditioning-v686/reporting-v692/repo','/home/angus/worktrees/spacepdhcg-literature-venv/bin/python':'/home/ubuntu/spacepdhcg/v1/.venv/bin/python','/home/angus/worktrees/spacepdhcg-release/benchmarks/gtoc12/data':'/home/ubuntu/spacepdhcg/gtoc12/benchmarks/gtoc12/data','/home/angus/cudss-isolated-0.8.0.10/nvidia/cu12':'/home/ubuntu/spacepdhcg-recovery-v152/cudss','/usr/local/cuda-12.8':'/usr/local/cuda','/home/angus/.spacepdhcg-gpu.lock':'/home/ubuntu/.spacepdhcg-gpu.lock'}
for old,new in mapping.items():runner=runner.replace(old,new)
runner=runner.replace('campaign-v691','campaign-v693').replace("(('baseline0',0),('candidate0',1),('candidate1',1),('baseline1',0))","(('candidate0',1),)")
worker='''from pathlib import Path
import fcntl,json,os,subprocess,time,traceback,sys
build=Path('/home/ubuntu/spacepdhcg-retry-conditioning-v686');root=build/'reporting-v692';repo=root/'repo'
env={k:v for k,v in os.environ.items() if not k.startswith(('SPACEPDHCG_TEST_','QOCO_REPLAY_'))}
env.update(PYTHONPATH=str(repo/'src'),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',SPACEPDHCG_GTOC12_GPU_TESTS='1',SPACEPDHCG_GTOC12_CUDA_LIBRARY=str(build/'final/libspacepdhcg_cuda.so'),SPACEPDHCG_QOCO_LIBRARY=str(build/'final/libqoco.so'),SPACEPDHCG_GTOC12_DATA='/home/ubuntu/spacepdhcg/gtoc12/benchmarks/gtoc12/data',LD_LIBRARY_PATH=str(build/'final')+':/home/ubuntu/spacepdhcg-recovery-v152/cudss/lib:/usr/local/cuda/lib64')
report=dict(complete=False,stages=[])
def save(): (root/'report.json').write_text(json.dumps(report,indent=2))
def run(name,cmd):
    start=time.perf_counter()
    with (root/(name+'.log')).open('x') as log:
        child=subprocess.Popen(cmd,cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT);report.update(stage=name,child_pid=child.pid);save();code=child.wait(timeout=1800)
    report['stages'].append(dict(name=name,returncode=code,seconds=time.perf_counter()-start,command=cmd));save();return code
save()
try:
    boot="import sys;sys.meta_path=[f for f in sys.meta_path if f.__class__.__module__!='_editable_skbc_spacepdhcg'];import pytest;sys.exit(pytest.main(sys.argv[1:]))"
    with Path('/home/ubuntu/.spacepdhcg-gpu.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if run('pytest',[sys.executable,'-c',boot,'-q','tests/test_gtoc12_gpu_conditioning_retry.py']):raise RuntimeError('pytest')
        run('memcheck',['/usr/local/cuda/bin/compute-sanitizer','--tool','memcheck','--target-processes','all','--error-exitcode','99',sys.executable,'-c',boot,'-q','tests/test_gtoc12_gpu_conditioning_retry.py::test_captured_return_converges_and_certifies_after_rebinding[True-True]'])
    if run('campaign',[sys.executable,str(build/'campaign-runner-v693.py')]):raise RuntimeError('campaign')
except BaseException:report['error']=traceback.format_exc()
report['complete']=True;save()
'''
key=Path('/tmp/traj-key.pem')
if not key.exists():
    fd=os.open(str(key),os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    with os.fdopen(fd,'wb') as file:file.write(Path('traj-key.pem').read_bytes())
subprocess.run(['scp','-q','-i',str(key),'-o','BatchMode=yes',str(archive),'ubuntu@192.222.55.229:'+str(archive)],check=True,timeout=55)
launch="from pathlib import Path\nimport hashlib,json,shutil,subprocess,tarfile\nbuild=Path('/home/ubuntu/spacepdhcg-retry-conditioning-v686');root=build/'reporting-v692';root.mkdir(exist_ok=False);repo=root/'repo';repo.mkdir()\n"
launch+="for folder in ('src','tests'):shutil.copytree(build/'repo'/folder,repo/folder,ignore=shutil.ignore_patterns('__pycache__'))\nmanifest="+repr(manifest)+"\nwith tarfile.open('/tmp/retry-reporting-v692.tar.gz') as tar:\n for member in tar:assert member.isfile() and member.name in manifest\n tar.extractall(repo,filter='data')\nfor name,digest in manifest.items():assert hashlib.sha256((repo/name).read_bytes()).hexdigest()==digest,name\n"
launch+="(root/'source-manifest.json').write_text(json.dumps(manifest,indent=2))\n(build/'campaign-runner-v693.py').write_text("+repr(runner)+")\n(root/'worker.py').write_text("+repr(worker)+")\nwith (root/'worker.log').open('x') as log:child=subprocess.Popen(['/home/ubuntu/spacepdhcg/v1/.venv/bin/python',str(root/'worker.py')],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)\nprint(json.dumps(dict(pid=child.pid,root=str(root))))\n"
Path('build/performance/launch_retry_reporting_v692.py').write_text(launch)
r=subprocess.run(['ssh','-i',str(key),'-o','BatchMode=yes','-o','ConnectTimeout=10','ubuntu@192.222.55.229','python3 -'],input=launch,text=True,capture_output=True,timeout=55);print(r.stdout);print(r.stderr);r.check_returncode()
