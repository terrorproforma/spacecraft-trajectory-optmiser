from pathlib import Path
import fcntl,hashlib,json,os,subprocess,sys,time,traceback
home=Path.home();root=Path(__file__).resolve().parent;build=home/'spacepdhcg-catalogue-v808';repo=build/'repo';remote=home.name=='ubuntu'
cuda='/usr/local/cuda' if remote else '/usr/local/cuda-12.8'
py=str(home/'spacepdhcg/v1/.venv/bin/python') if remote else '/home/angus/worktrees/spacepdhcg-literature-venv/bin/python'
core=build/'final/libspacepdhcg_cuda.so';controller=build/'build/cuda-tests/gtoc12_scvx_test'
env={k:v for k,v in os.environ.items() if not k.startswith(('SPACEPDHCG_TEST_','QOCO_REPLAY_'))}
env.update(PYTHONPATH=str(repo/'src'),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',SPACEPDHCG_GTOC12_GPU_TESTS='1',SPACEPDHCG_GTOC12_CUDA_LIBRARY=str(core),SPACEPDHCG_COMPLETION_TEST_LIBRARY=str(core),LD_LIBRARY_PATH=cuda+'/lib64',SPACEPDHCG_GTOC12_DATA=str(home/'spacepdhcg/gtoc12/benchmarks/gtoc12/data') if remote else '/home/angus/worktrees/spacepdhcg-release/benchmarks/gtoc12/data')
report=dict(pid=os.getpid(),complete=False,success=False,stages=[],core_sha256=hashlib.sha256(core.read_bytes()).hexdigest(),controller_sha256=hashlib.sha256(controller.read_bytes()).hexdigest(),source_manifest_sha256=hashlib.sha256((build/'source-manifest.json').read_bytes()).hexdigest(),runner_correction='v808 looked in build/cuda; actual executable is build/cuda-tests. Builds reused unchanged; v809/v810 started zero benchmark runs and zero native solves.')
def save():
    p=root/'report.tmp';p.write_text(json.dumps(report,indent=2));p.replace(root/'report.json')
def run(name,command):
    started=time.perf_counter()
    with (root/(name+'.log')).open('x') as log:
        child=subprocess.Popen(command,cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT)
        report.update(stage=name,child_pid=child.pid);save();code=child.wait()
    report['stages'].append(dict(name=name,code=code,seconds=time.perf_counter()-started,command=command));save()
    assert code==0,(name,code)
save()
try:
    original=json.loads((build/'report.json').read_text());assert original['complete'] and any(s['name']=='build' and s['code']==0 for s in original['stages'])
    assert report['core_sha256']==original['core_sha256']
    for p,h in json.loads((build/'source-manifest.json').read_text())['files'].items():assert hashlib.sha256((repo/p).read_bytes()).hexdigest()==h,p
    boot="import sys;sys.meta_path=[f for f in sys.meta_path if f.__class__.__module__!='_editable_skbc_spacepdhcg'];import pytest;sys.exit(pytest.main(sys.argv[1:]))"
    tests=['tests/test_gtoc12_gpu_collect_dp.py','tests/test_gtoc12_gpu_collect_workspace.py','tests/test_gtoc12_gpu_collect_tables.py','tests/test_gtoc12_gpu_resident_collect_tables.py','tests/test_gtoc12_gpu_completion_model.py','tests/test_gtoc12_gpu_completion.py','tests/test_gtoc12_catalogue_immutability.py','tests/test_gtoc12_rules_and_data.py','tests/test_gtoc12_completion_costs.py','tests/test_gtoc12_completion_capture.py','tests/test_gtoc12_zoh_trajectory_seed.py']
    with (home/'.spacepdhcg-gpu.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        run('controller',[str(controller)])
        run('pytest',[py,'-c',boot,'-q',*tests])
        for mode in ('memcheck','racecheck','synccheck'):
            run(mode,[cuda+'/bin/compute-sanitizer','--tool',mode,'--error-exitcode','86',py,'-c',boot,'-q','tests/test_gtoc12_gpu_collect_workspace.py','tests/test_gtoc12_gpu_resident_collect_tables.py','tests/test_gtoc12_gpu_completion_model.py'])
            run(mode+'-controller',[cuda+'/bin/compute-sanitizer','--tool',mode,'--error-exitcode','86',str(controller)])
    report['success']=True
except BaseException:report['error']=traceback.format_exc()
report['complete']=True;save()
