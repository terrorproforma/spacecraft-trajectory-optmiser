from pathlib import Path
import os,sys,json,time,fcntl,hashlib,dataclasses
sys.meta_path=[f for f in sys.meta_path if f.__class__.__module__!='_editable_skbc_spacepdhcg']
repo=Path.cwd();sys.path.insert(0,str(repo/'src'))
os.environ.update(SPACEPDHCG_GTOC12_CUDA_LIBRARY='/home/angus/build-spacepdhcg-retime-dp-v219/final/libspacepdhcg_cuda.so',SPACEPDHCG_GTOC12_DATA='/home/angus/worktrees/spacepdhcg-release/benchmarks/gtoc12/data')
from spacepdhcg.gtoc12.data import load_catalogue,load_bonus_table
from spacepdhcg.gtoc12.pipeline import plan_from_route_summary
from spacepdhcg.gtoc12.retiming import Retimer,weighted_collected
from spacepdhcg.gtoc12.bundles import ClusterPricingSettings,cluster_search_settings,cluster_retime_settings
from spacepdhcg.gtoc12.lambert import using_lambert_backend
out=repo/'build/performance/fleet-retime-v227';out.mkdir(exist_ok=False)
fleet=json.loads((repo/'results/lambda/2026-09-06/fleet_master_v11/run_report.json').read_text())
cat=load_catalogue();bonus=load_bonus_table();weights={int(i):float(bonus.coefficient[i-1]) for i in cat.ids}
lock=open(Path.home()/'.spacepdhcg-gpu.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
rows=[];start=time.perf_counter()
with using_lambert_backend('cuda') as gpu:
 for ship,item in enumerate(fleet['master']['selected'],1):
  if ship not in [18,20]:continue
  index=int(item['label'].split('_')[0][1:])-10000;slot=int(item['label'].split('_s')[1]);group=fleet['groups'][index]['name']
  matches=[]
  for source in (repo/'results/gtoc12/runs'/group/f'ship_{slot:02d}').rglob('route_summary.json'):
   summary=json.loads(source.read_text())
   if summary.get('certified') and abs(summary['total_collected_kg']-item['collected_kg'])<1e-5:matches.append((source,summary))
  assert matches,ship
  source,summary=sorted(matches,key=lambda x:str(x[0]))[0];plan=plan_from_route_summary(summary)
  for calibrated in [False]:
   settings=ClusterPricingSettings();search=cluster_search_settings(settings,len(cat.ids));retimer=Retimer(cat,search,dataclasses.replace(cluster_retime_settings(settings,last=True),step_days=5.0),weights=weights);retimer.protect_earth_leg(plan)
   used=0
   if calibrated:
    for leg in plan.legs:
     if leg.role=='camp' or leg.delta_v_proxy_km_s<=0:continue
     flown=next((x for x in summary['legs'] if x['from']==leg.from_id and x['to']==leg.to_id and abs(x['t0']-leg.departure_epoch)<1e-5 and abs(x['tf']-leg.arrival_epoch)<1e-5),None)
     if flown is None or not flown['certified']:continue
     retimer.calibrate(leg.from_id,leg.to_id,flown['delta_v_km_s']/leg.delta_v_proxy_km_s,authority_ratio=retimer.authority_ratio(leg.delta_v_proxy_km_s,flown['mass_before'],leg.tof_days),tof_days=leg.tof_days);used+=1
   result=retimer.retime(plan);row=dict(ship=ship,label=item['label'],source=str(source.relative_to(repo)),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),calibrated=calibrated,calibrated_legs=used,result=result.summary())
   if result.plan is not None:
    row['weighted_gain']=weighted_collected(result.plan,weights)-weighted_collected(plan,weights);row['raw_gain']=result.plan.total_collected_kg-plan.total_collected_kg
    path=out/f'ship-{ship:02d}-cal-{int(calibrated)}.json';path.write_text(json.dumps(result.plan.summary(),indent=2));row['plan']=str(path.relative_to(repo))
   rows.append(row);(out/'report.json').write_text(json.dumps(dict(complete=False,rows=rows,seconds=time.perf_counter()-start),indent=2));print(json.dumps({k:v for k,v in row.items() if k not in ['source','source_sha256','result'] }|{'failure':result.failure}),flush=True)
 telemetry=dict(gpu.telemetry)
(out/'report.json').write_text(json.dumps(dict(complete=True,rows=rows,seconds=time.perf_counter()-start,telemetry=telemetry,source_module=Retimer.__module__,settings=dataclasses.asdict(search)),indent=2))

