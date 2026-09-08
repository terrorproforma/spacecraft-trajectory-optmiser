from pathlib import Path
import cProfile,fcntl,io,json,os,pstats,sys,time
home=Path.home()
sys.meta_path=[f for f in sys.meta_path if f.__class__.__module__!='_editable_skbc_spacepdhcg']
sys.path.insert(0,str(home/'spacepdhcg-fleet-v727/repo/src'))
os.environ['SPACEPDHCG_GTOC12_CUDA_LIBRARY']=str(home/'spacepdhcg-fleet-v727/final/libspacepdhcg_cuda.so')
from spacepdhcg.gtoc12.cooperative import FleetColumn
from spacepdhcg.gtoc12.gpu_fleet import solve_fleet_cuda
p=json.loads((home/'spacepdhcg-fleet-pool-v716/pool.json').read_text())
cols=[FleetColumn(r['identifier'],r['ship_id'],r['label'],*({int(k):v for k,v in r[key].items()} for key in ('deploys','collects','foreign','mass')),True) for r in p['rows']]
warm=tuple(c for c in cols if c.identifier in p['warm']);weights={int(k):v for k,v in p['weights'].items()}
with (home/'.spacepdhcg-gpu.lock').open('a') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    solve_fleet_cuda(cols,weights=weights,incumbent=warm,node_cap=0)
    profile=cProfile.Profile();profile.enable()
    for _ in range(5):
        result=solve_fleet_cuda(cols,weights=weights,incumbent=warm,node_cap=0)
    profile.disable()
    out=io.StringIO();pstats.Stats(profile,stream=out).sort_stats('cumulative').print_stats(20)
    print(out.getvalue());print(json.dumps(dict(native_seconds=result.native_seconds,score=result.objective)))
