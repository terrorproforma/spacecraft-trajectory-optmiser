from pathlib import Path
import fcntl,hashlib,json,os,statistics,sys,time,traceback
home=Path.home();root=home/'spacepdhcg-fleet-v721';root.mkdir(exist_ok=False)
sys.meta_path=[f for f in sys.meta_path if f.__class__.__module__!='_editable_skbc_spacepdhcg'];sys.path.insert(0,str(home/'spacepdhcg-fleet-v720/repo/src'))
from spacepdhcg.gtoc12.cooperative import FleetColumn
from spacepdhcg.gtoc12.gpu_fleet import solve_fleet_cuda
pool=home/'spacepdhcg-fleet-pool-v716/pool.json';p=json.loads(pool.read_text());cols=[]
for r in p['rows']:cols.append(FleetColumn(r['identifier'],r['ship_id'],r['label'],*({int(k):v for k,v in r[key].items()} for key in ('deploys','collects','foreign','mass')),True))
warm=tuple(c for c in cols if c.identifier in p['warm']);weights={int(k):v for k,v in p['weights'].items()}
report=dict(pid=os.getpid(),complete=False,success=False,pool_sha256=hashlib.sha256(pool.read_bytes()).hexdigest(),rows=[])
def save():
    tmp=root/'report.tmp';tmp.write_text(json.dumps(report,indent=2));tmp.replace(root/'report.json')
save()
try:
    with (home/'.spacepdhcg-gpu.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        variants=['v713','v718','v720'] if (home/'spacepdhcg-fleet-v718/final').exists() else ['v713','v720']
        specs=[(v,200000,8,repeat) for repeat in range(5) for v in (variants if repeat%2==0 else variants[::-1])]
        specs += [('v720',cap,bits,0) for cap,bits in [(2000000,8),(20000000,8),(200000,0),(200000,10)]]
        for version,cap,bits,repeat in specs:
            lib=home/('spacepdhcg-fleet-'+version)/'final/libspacepdhcg_cuda.so';os.environ['SPACEPDHCG_GTOC12_CUDA_LIBRARY']=str(lib)
            report['running']=dict(version=version,cap=cap,bits=bits,repeat=repeat);save();start=time.perf_counter()
            result=solve_fleet_cuda(cols,weights=weights,incumbent=warm,node_cap=cap,prefix_bits=bits)
            s=result.summary();report['rows'].append(dict(**report['running'],seconds=time.perf_counter()-start,library_sha256=hashlib.sha256(lib.read_bytes()).hexdigest(),stats={k:v for k,v in s.items() if not isinstance(v,(dict,list))},selected=[c.identifier for c in result.selected]));save()
    report['success']=True
except BaseException:report['error']=traceback.format_exc()
report['complete']=True;save();print(json.dumps({k:v for k,v in report.items() if k!='rows'}))
