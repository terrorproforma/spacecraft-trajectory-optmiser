from pathlib import Path
import fcntl,json,os,subprocess,time,traceback
home=Path.home();root=home/'spacepdhcg-fleet-workspace-v744';root.mkdir(exist_ok=False)
build=home/'spacepdhcg-fleet-v742';repo=build/'repo';remote=home.name=='ubuntu'
py=str(home/'spacepdhcg/v1/.venv/bin/python') if remote else str(home/'worktrees/spacepdhcg-literature-venv/bin/python')
cuda='/usr/local/cuda' if remote else '/usr/local/cuda-12.8'
env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',SPACEPDHCG_GTOC12_GPU_TESTS='1',PYTHONPATH=str(repo/'src'),SPACEPDHCG_GTOC12_CUDA_LIBRARY=str(build/'final/libspacepdhcg_cuda.so'))
code='''from pathlib import Path
import json,sys
home=Path.home();sys.meta_path=[f for f in sys.meta_path if f.__class__.__module__!='_editable_skbc_spacepdhcg'];sys.path.insert(0,str(home/'spacepdhcg-fleet-v742/repo/src'))
from spacepdhcg.gtoc12.cooperative import FleetColumn,fleet_feasible
from spacepdhcg.gtoc12.gpu_fleet import CudaFleetWorkspace
p=json.loads((home/'spacepdhcg-fleet-pool-v716/pool.json').read_text())
cols=[FleetColumn(r['identifier'],r['ship_id'],r['label'],*({int(k):v for k,v in r[key].items()} for key in ('deploys','collects','foreign','mass')),True) for r in p['rows']]
warm=tuple(c for c in cols if c.identifier in p['warm']);weights={int(k):v for k,v in p['weights'].items()}
expected=json.loads((home/'spacepdhcg-fleet-refine-v729/input/selected.json').read_text())
for repeat in range(2):
    with CudaFleetWorkspace(cols,weights=weights) as workspace:
        for cap,rounds,ships,incumbent in [(0,16,100,warm),(0,0,0,None),(200000,16,100,warm)]:
            r=workspace.solve(incumbent=incumbent,node_cap=cap,exchange_rounds=rounds,max_ships=ships)
            assert not fleet_feasible(r.selected)
            if ships:
                assert [c.identifier for c in r.selected]==expected
                assert r.nodes==(35145 if cap else 0) and abs(r.objective-12842.970672270907)<1e-8
                assert r.exchange_proposals==179205 and r.exchange_moves==2
            else:assert not r.selected and r.objective==0
            print(json.dumps(dict(repeat=repeat,cap=cap,rounds=rounds,nodes=r.nodes,objective=r.objective,native_seconds=r.native_seconds)))
'''
(root/'case.py').write_text(code);report=dict(pid=os.getpid(),complete=False,success=False,stages=[])
def save():
    tmp=root/'report.tmp';tmp.write_text(json.dumps(report,indent=2));tmp.replace(root/'report.json')
save()
try:
    with (home/'.spacepdhcg-gpu.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        for mode in ('pytest','memcheck','racecheck','synccheck'):
            if mode=='nsys' and remote:continue
            started=time.perf_counter()
            if mode=='pytest':
                boot="import sys;sys.meta_path=[f for f in sys.meta_path if f.__class__.__module__!='_editable_skbc_spacepdhcg'];import pytest;sys.exit(pytest.main(sys.argv[1:]))"
                cmd=[py,'-c',boot,'-q',*[str(repo/'tests'/name) for name in ('test_gtoc12_gpu_fleet.py','test_gtoc12_gpu_cli.py','test_gtoc12_cooperative.py','test_gtoc12_collectdp.py','test_gtoc12_completion_costs.py')]]
            elif mode=='nsys':
                cmd=['nsys','profile','--trace=cuda','--sample=none','--cpuctxsw=none','-o',str(root/'retained'),py,str(root/'case.py')]
            else:
                cmd=[cuda+'/bin/compute-sanitizer','--tool',mode,'--error-exitcode','86',*(['--leak-check','full'] if mode=='memcheck' else []),py,str(root/'case.py')]
            with (root/(mode+'.log')).open('x') as log:
                child=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT);report.update(stage=mode,child_pid=child.pid);save();code=child.wait()
            report['stages'].append(dict(name=mode,code=code,seconds=time.perf_counter()-started,command=cmd));save()
            if code:raise RuntimeError(mode)
        report['success']=True
except BaseException:report['error']=traceback.format_exc()
report['complete']=True;save()
