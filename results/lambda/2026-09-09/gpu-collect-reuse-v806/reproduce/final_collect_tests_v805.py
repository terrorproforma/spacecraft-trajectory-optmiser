from pathlib import Path
import fcntl,hashlib,json,os,shutil,subprocess,time,traceback
home=Path.home();root=home/'spacepdhcg-collect-final-v805';repo=root/'repo';base=home/'spacepdhcg-collect-v800'
report=dict(pid=os.getpid(),complete=False,success=False,stages=[])
def save():
    p=root/'report.tmp';p.write_text(json.dumps(report,indent=2));p.replace(root/'report.json')
save()
try:
    env=dict(os.environ,PYTHONPATH=str(repo/'src'),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',SPACEPDHCG_GTOC12_GPU_TESTS='1',SPACEPDHCG_GTOC12_CUDA_LIBRARY=str(base/'final/libspacepdhcg_cuda.so'),SPACEPDHCG_GTOC12_DATA=str(home/('spacepdhcg/gtoc12/benchmarks/gtoc12/data' if home.name=='ubuntu' else 'worktrees/spacepdhcg-release/benchmarks/gtoc12/data')),LD_LIBRARY_PATH=('/usr/local/cuda' if home.name=='ubuntu' else '/usr/local/cuda-12.8')+'/lib64')
    py=str(home/'spacepdhcg/v1/.venv/bin/python') if home.name=='ubuntu' else '/home/angus/worktrees/spacepdhcg-literature-venv/bin/python'
    boot="import sys;sys.meta_path=[f for f in sys.meta_path if f.__class__.__module__!='_editable_skbc_spacepdhcg'];import pytest;sys.exit(pytest.main(sys.argv[1:]))"
    tests=['tests/test_gtoc12_gpu_collect_dp.py','tests/test_gtoc12_gpu_collect_workspace.py','tests/test_gtoc12_gpu_collect_tables.py','tests/test_gtoc12_gpu_resident_collect_tables.py']
    commands=[('pytest',[py,'-c',boot,'-q',*tests])]
    for mode in ('memcheck','racecheck','synccheck'):
        commands.append((mode,[('/usr/local/cuda' if home.name=='ubuntu' else '/usr/local/cuda-12.8')+'/bin/compute-sanitizer','--tool',mode,'--error-exitcode','86',py,'-c',boot,'-q','tests/test_gtoc12_gpu_collect_workspace.py','tests/test_gtoc12_gpu_resident_collect_tables.py']))
    with (home/'.spacepdhcg-gpu.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        for name,cmd in commands:
            report['stage']=name;save();begin=time.perf_counter()
            with (root/(name+'.log')).open('x') as out:
                child=subprocess.Popen(cmd,cwd=repo,env=env,stdout=out,stderr=subprocess.STDOUT);report['child_pid']=child.pid;save();code=child.wait()
            report['stages'].append(dict(name=name,code=code,seconds=time.perf_counter()-begin,command=cmd));save();assert code==0
    report['core_sha256']=hashlib.sha256((base/'final/libspacepdhcg_cuda.so').read_bytes()).hexdigest();report['success']=True
except BaseException:report['error']=traceback.format_exc()
report['complete']=True;save()
