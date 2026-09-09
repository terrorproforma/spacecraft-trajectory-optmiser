from pathlib import Path
import subprocess
worker="""from pathlib import Path
import fcntl,hashlib,json,os,subprocess,time,traceback
home=Path.home();root=Path(__file__).resolve().parent;build=home/'spacepdhcg-expansion-default-v834'
report=dict(complete=False,success=False,stage='waiting_for_benchmark')
def save():
    p=root/'report.tmp';p.write_text(json.dumps(report,indent=2));p.replace(root/'report.json')
save()
try:
    while not json.loads((home/'spacepdhcg-expansion-bench-v831/report.json').read_text())['complete']:time.sleep(2)
    remote=home.name=='ubuntu';cuda='/usr/local/cuda' if remote else '/usr/local/cuda-12.8'
    py=str(home/'spacepdhcg/v1/.venv/bin/python') if remote else '/home/angus/worktrees/spacepdhcg-literature-venv/bin/python'
    env=dict(os.environ,SPACEPDHCG_GTOC12_GPU_TESTS='1',SPACEPDHCG_GTOC12_CUDA_LIBRARY=str(build/'final/libspacepdhcg_cuda.so'),PYTHONPATH=str(build/'repo/src'),LD_LIBRARY_PATH=cuda+'/lib64',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    boot="import sys;sys.meta_path=[f for f in sys.meta_path if f.__class__.__module__!='_editable_skbc_spacepdhcg'];import pytest;sys.exit(pytest.main(sys.argv[1:]))"
    command=[cuda+'/bin/compute-sanitizer','--tool','memcheck','--leak-check','full','--error-exitcode','86',py,'-c',boot,'-q','tests/test_gtoc12_gpu_expansion.py','-k','not layout and not empty']
    report.update(command=command,core_sha256=hashlib.sha256((build/'final/libspacepdhcg_cuda.so').read_bytes()).hexdigest(),source_manifest_sha256=hashlib.sha256((build/'source-manifest.json').read_bytes()).hexdigest());save()
    with (home/'.spacepdhcg-gpu.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);report['stage']='leakcheck';save()
        with (root/'leakcheck.log').open('x') as log:r=subprocess.run(command,cwd=build/'repo',env=env,stdout=log,stderr=subprocess.STDOUT)
        report['returncode']=r.returncode;save();assert r.returncode==0
    report['success']=True
except BaseException:report['error']=traceback.format_exc()
report['complete']=True;save()
"""
code='WORKER='+repr(worker)+"\nfrom pathlib import Path\nimport subprocess\nroot=Path.home()/'spacepdhcg-expansion-leaks-v836';root.mkdir()\n(root/'worker.py').write_text(WORKER)\nwith (root/'worker.log').open('x') as log:print(subprocess.Popen(['python3',str(root/'worker.py')],stdout=log,stderr=subprocess.STDOUT,start_new_session=True).pid)\n"
Path('build/performance/launch_expansion_leaks_both_v836.py').write_text(code)
r=subprocess.run(['ssh','-i','/home/angus/.ssh/spacepdhcg-expansion-key.pem','-o','BatchMode=yes','ubuntu@192.222.55.229','python3 -'],input=code,text=True,capture_output=True,check=True,timeout=40);print(r.stdout,r.stderr);exec(code)
