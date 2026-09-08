from pathlib import Path
import fcntl,json,os,shutil,subprocess,sys,time,traceback
home=Path.home();root=home/'spacepdhcg-fleet-v730';root.mkdir()
shutil.copyfile(home/'fleet-exchange-final-test.py',root/'test_gtoc12_gpu_fleet.py')
remote=home.name=='ubuntu';py=str(home/'spacepdhcg/v1/.venv/bin/python') if remote else '/home/angus/worktrees/spacepdhcg-literature-venv/bin/python'
cuda='/usr/local/cuda' if remote else '/usr/local/cuda-12.8'
env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',SPACEPDHCG_GTOC12_CUDA_LIBRARY=str(home/'spacepdhcg-fleet-v727/final/libspacepdhcg_cuda.so'))
code="""from pathlib import Path
import json,sys
home=Path.home();sys.meta_path=[f for f in sys.meta_path if f.__class__.__module__!='_editable_skbc_spacepdhcg'];sys.path.insert(0,str(home/'spacepdhcg-fleet-v727/repo/src'))
from spacepdhcg.gtoc12.cooperative import FleetColumn,fleet_feasible
from spacepdhcg.gtoc12.gpu_fleet import solve_fleet_cuda
p=json.loads((home/'spacepdhcg-fleet-pool-v716/pool.json').read_text());cols=[]
for r in p['rows']:cols.append(FleetColumn(r['identifier'],r['ship_id'],r['label'],*({int(k):v for k,v in r[key].items()} for key in ('deploys','collects','foreign','mass')),True))
warm=tuple(c for c in cols if c.identifier in p['warm']);weights={int(k):v for k,v in p['weights'].items()}
r=solve_fleet_cuda(cols,weights=weights,incumbent=warm,node_cap=0)
assert not fleet_feasible(r.selected) and [c.identifier for c in r.selected]==json.loads((home/'spacepdhcg-fleet-refine-v729/input/selected.json').read_text())
assert r.nodes==0 and abs(r.objective-12842.970672270907)<1e-8
print(json.dumps(dict(nodes=r.nodes,objective=r.objective,native_seconds=r.native_seconds)))
"""
(root/'case.py').write_text(code);report=dict(pid=os.getpid(),complete=False,success=False,stages=[])
def save():
    (root/'report.json').write_text(json.dumps(report,indent=2))
save()
try:
    with (home/'.spacepdhcg-gpu.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        for mode in ('pytest','memcheck','racecheck','synccheck'):
            t=time.perf_counter();cmd=[cuda+'/bin/compute-sanitizer','--tool',mode,'--error-exitcode','86',py,str(root/'case.py')]
            if mode=='pytest':
                env.update(SPACEPDHCG_GTOC12_GPU_TESTS='1',PYTHONPATH=str(home/'spacepdhcg-fleet-v727/repo/src'))
                boot="import sys;sys.meta_path=[f for f in sys.meta_path if f.__class__.__module__!='_editable_skbc_spacepdhcg'];import pytest;sys.exit(pytest.main(sys.argv[1:]))"
                cmd=[py,'-c',boot,'-q',str(root/'test_gtoc12_gpu_fleet.py'),*[str(home/'spacepdhcg-fleet-v727/repo/tests'/name) for name in ('test_gtoc12_gpu_cli.py','test_gtoc12_cooperative.py','test_gtoc12_collectdp.py')]]
            with (root/(mode+'.log')).open('x') as log:
                child=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT);report.update(stage=mode,child_pid=child.pid);save();code=child.wait()
            report['stages'].append(dict(name=mode,code=code,seconds=time.perf_counter()-t,command=cmd));save()
            if code:raise RuntimeError(mode)
    report['success']=True
except BaseException:report['error']=traceback.format_exc()
report['complete']=True;save()
