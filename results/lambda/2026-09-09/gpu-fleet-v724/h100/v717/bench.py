from pathlib import Path
import fcntl,hashlib,json,os,sys,time,traceback
home=Path.home(); root=home/'spacepdhcg-fleet-v717';root.mkdir(exist_ok=True)
sys.meta_path=[f for f in sys.meta_path if f.__class__.__module__!='_editable_skbc_spacepdhcg']
sys.path.insert(0,str(home/'spacepdhcg-fleet-v713/repo/src'))
os.environ['SPACEPDHCG_GTOC12_CUDA_LIBRARY']=str(home/'spacepdhcg-fleet-v713/final/libspacepdhcg_cuda.so')
from spacepdhcg.gtoc12.cooperative import FleetColumn,solve_fleet_master
from spacepdhcg.gtoc12.gpu_fleet import solve_fleet_cuda
pool=Path(sys.argv[1]);backend=sys.argv[2];cap=int(sys.argv[3]);bits=int(sys.argv[4]) if len(sys.argv)>4 else 8
p=json.loads(pool.read_text());cols=[]
for r in p['rows']:
    cols.append(FleetColumn(r['identifier'],r['ship_id'],r['label'],*({int(k):v for k,v in r[key].items()} for key in ('deploys','collects','foreign','mass')),True))
warm=tuple(c for c in cols if c.identifier in p['warm']);weights={int(k):v for k,v in p['weights'].items()}
report=dict(pid=os.getpid(),backend=backend,node_cap=cap,prefix_bits=bits,pool_sha256=hashlib.sha256(pool.read_bytes()).hexdigest(),columns=len(cols),complete=False)
out=root/(pool.parent.name+'-'+backend+'-'+str(cap)+'-'+str(bits)+'.json')
def save():
    tmp=out.with_suffix('.tmp');tmp.write_text(json.dumps(report,indent=2));tmp.replace(out)
save()
try:
    lock=(home/'.spacepdhcg-gpu.lock').open('a')
    if backend=='cuda':fcntl.flock(lock,fcntl.LOCK_EX)
    start=time.perf_counter()
    kw=dict(weights=weights,incumbent=warm,node_cap=cap,max_ships=100)
    result=solve_fleet_cuda(cols,prefix_bits=bits,**kw) if backend=='cuda' else solve_fleet_master(cols,**kw)
    report.update(seconds=time.perf_counter()-start,summary=result.summary(),selected=[c.identifier for c in result.selected],missing_sources=[r['identifier'] for r in p['rows'] if r['identifier'] in {c.identifier for c in result.selected} and not r.get('source_available',True)],success=True)
except BaseException:report.update(error=traceback.format_exc(),success=False)
report['complete']=True;save();print(json.dumps(report))
