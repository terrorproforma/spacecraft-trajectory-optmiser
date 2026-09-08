from pathlib import Path
import fcntl,hashlib,json,os,sys,time,traceback
home=Path.home();root=home/'spacepdhcg-fleet-v728';root.mkdir()
sys.meta_path=[f for f in sys.meta_path if f.__class__.__module__!='_editable_skbc_spacepdhcg'];sys.path.insert(0,str(home/'spacepdhcg-fleet-v727/repo/src'))
os.environ['SPACEPDHCG_GTOC12_CUDA_LIBRARY']=str(home/'spacepdhcg-fleet-v727/final/libspacepdhcg_cuda.so')
from spacepdhcg.gtoc12.cooperative import FleetColumn
from spacepdhcg.gtoc12.gpu_fleet import solve_fleet_cuda
p=json.loads((home/'spacepdhcg-fleet-pool-v716/pool.json').read_text());cols=[]
for r in p['rows']:cols.append(FleetColumn(r['identifier'],r['ship_id'],r['label'],*({int(k):v for k,v in r[key].items()} for key in ('deploys','collects','foreign','mass')),True))
warm=tuple(c for c in cols if c.identifier in p['warm']);weights={int(k):v for k,v in p['weights'].items()}
report=dict(pid=os.getpid(),complete=False,success=False,rows=[])
def save():
    tmp=root/'report.tmp';tmp.write_text(json.dumps(report,indent=2));tmp.replace(root/'report.json')
save()
try:
    with (home/'.spacepdhcg-gpu.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        for rounds,cap,repeat in [(r,0,i) for i in range(5) for r in ([0,16] if i%2==0 else [16,0])]+[(16,200000,0),(100,0,0)]:
            report['running']=dict(rounds=rounds,cap=cap,repeat=repeat);save();start=time.perf_counter()
            result=solve_fleet_cuda(cols,weights=weights,incumbent=warm,node_cap=cap,exchange_rounds=rounds)
            seconds=time.perf_counter()-start;s=result.summary();selected=[c.identifier for c in result.selected]
            report['rows'].append(dict(**report['running'],seconds=seconds,stats={k:v for k,v in s.items() if not isinstance(v,(dict,list))},selected=selected,missing_sources=[r['identifier'] for r in p['rows'] if r['identifier'] in selected and not r.get('source_available',True)]));save()
    report['success']=True
except BaseException:report['error']=traceback.format_exc()
report['complete']=True;save();print(json.dumps({k:v for k,v in report.items() if k!='rows'}))
