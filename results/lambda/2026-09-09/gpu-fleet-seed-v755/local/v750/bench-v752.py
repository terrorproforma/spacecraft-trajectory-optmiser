from pathlib import Path
import fcntl,hashlib,importlib.util,json,os,statistics,sys,time,traceback
home=Path.home();root=home/'spacepdhcg-fleet-workspace-v752';root.mkdir(exist_ok=False)
sys.meta_path=[f for f in sys.meta_path if f.__class__.__module__!='_editable_skbc_spacepdhcg']
sys.path.insert(0,str(home/'spacepdhcg-fleet-v750/repo/src'))
from spacepdhcg.gtoc12.cooperative import FleetColumn
from spacepdhcg.gtoc12.gpu_fleet import CudaFleetWorkspace,solve_fleet_cuda
spec=importlib.util.spec_from_file_location('spacepdhcg.gtoc12.gpu_fleet_v742',home/'spacepdhcg-fleet-v742/repo/src/spacepdhcg/gtoc12/gpu_fleet.py')
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
pool=home/'spacepdhcg-fleet-pool-v716/pool.json';p=json.loads(pool.read_text())
cols=[FleetColumn(r['identifier'],r['ship_id'],r['label'],*({int(k):v for k,v in r[key].items()} for key in ('deploys','collects','foreign','mass')),True) for r in p['rows']]
warm=tuple(c for c in cols if c.identifier in p['warm']);weights={int(k):v for k,v in p['weights'].items()}
report=dict(pid=os.getpid(),complete=False,success=False,pool_sha256=hashlib.sha256(pool.read_bytes()).hexdigest(),rows=[],libraries={})
for version in ('v742','v750'):
    path=home/f'spacepdhcg-fleet-{version}/final/libspacepdhcg_cuda.so'
    report['libraries'][version]=dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
def save():
    tmp=root/'report.tmp';tmp.write_text(json.dumps(report,indent=2));tmp.replace(root/'report.json')
save()
try:
    with (home/'.spacepdhcg-gpu.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        os.environ['SPACEPDHCG_GTOC12_CUDA_LIBRARY']=report['libraries']['v742']['path']
        old.solve_fleet_cuda(cols,weights=weights,incumbent=warm,node_cap=0)
        os.environ['SPACEPDHCG_GTOC12_CUDA_LIBRARY']=report['libraries']['v742']['path']
        legacy=old.CudaFleetWorkspace(cols,weights=weights)
        os.environ['SPACEPDHCG_GTOC12_CUDA_LIBRARY']=report['libraries']['v750']['path']
        with CudaFleetWorkspace(cols,weights=weights) as workspace:
            report['setup_seconds']=workspace.setup_seconds;report['legacy_setup_seconds']=legacy.setup_seconds;save()
            for cap in (0,200000):
                reference=None
                for repeat in range(7):
                    modes=['old_one_shot','new_one_shot','old_retained','retained']
                    modes=modes[repeat%4:]+modes[:repeat%4]
                    for mode in modes:
                        os.environ['SPACEPDHCG_GTOC12_CUDA_LIBRARY']=report['libraries']['v742' if mode=='old_one_shot' else 'v750']['path']
                        started=time.perf_counter()
                        if mode in ('retained','old_retained'):
                            result=(legacy if mode=='old_retained' else workspace).solve(incumbent=warm,node_cap=cap)
                        else:
                            fn=old.solve_fleet_cuda if mode=='old_one_shot' else solve_fleet_cuda
                            result=fn(cols,weights=weights,incumbent=warm,node_cap=cap)
                        seconds=time.perf_counter()-started
                        summary=result.summary()
                        row=dict(mode=mode,cap=cap,repeat=repeat,seconds=seconds,stats={k:v for k,v in summary.items() if not isinstance(v,(dict,list))},selected=[c.identifier for c in result.selected])
                        signature=(row['selected'],result.objective,result.nodes,result.exchange_proposals,result.exchange_moves,result.exchange_rounds,result.exhaustive,result.upper_bound)
                        if reference is None:reference=signature
                        assert signature==reference,(mode,cap,signature,reference)
                        report['rows'].append(row);save()
        legacy.close()
        report['medians']={}
        for cap in (0,200000):
            report['medians'][str(cap)]={mode:{metric:statistics.median(r['seconds'] if metric=='seconds' else r['stats']['native_seconds'] for r in report['rows'] if r['cap']==cap and r['mode']==mode and r['repeat']>0) for metric in ('seconds','native_seconds')} for mode in ('old_one_shot','new_one_shot','old_retained','retained')}
        report['success']=True
except BaseException:report['error']=traceback.format_exc()
report['complete']=True;save();print(json.dumps({k:v for k,v in report.items() if k!='rows'}))
