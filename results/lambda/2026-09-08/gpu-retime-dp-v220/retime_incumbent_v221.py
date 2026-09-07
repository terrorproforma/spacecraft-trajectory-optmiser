from pathlib import Path
import json,os,time,fcntl,hashlib
from spacepdhcg.gtoc12.data import load_catalogue,load_bonus_table
from spacepdhcg.gtoc12.pipeline import plan_from_route_summary
from spacepdhcg.gtoc12.retiming import Retimer,weighted_collected
from spacepdhcg.gtoc12.search import SearchSettings
from spacepdhcg.gtoc12.lambert import using_lambert_backend
root=Path('build/performance/incumbent-retime-v221');root.mkdir(exist_ok=False)
report=json.loads(Path('results/lambda/2026-09-06/fleet_master_v11/run_report.json').read_text())
cat=load_catalogue();bonus=load_bonus_table();weights={int(i):float(bonus.coefficient[i-1]) for i in cat.ids}
lock=open(Path.home()/'.spacepdhcg-gpu.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
rows=[];start=time.perf_counter()
with using_lambert_backend('cuda') as gpu:
 for ship,item in enumerate(report['master']['selected'],1):
  index=int(item['label'].split('_')[0][1:])-10000;slot=int(item['label'].split('_s')[1])
  group=report['groups'][index]['name'];directory=Path('results/gtoc12/runs')/group/f'ship_{slot:02d}'
  candidates=[]
  for source in directory.rglob('route_summary.json'):
   summary=json.loads(source.read_text())
   if summary.get('certified') and abs(summary['total_collected_kg']-item['collected_kg'])<1e-5:candidates.append((source,summary))
  row=dict(ship=ship,label=item['label'],group=group,original_kg=item['collected_kg'])
  if not candidates:row['skipped']='No exact-mass matching source archive'
  else:
   source,summary=sorted(candidates,key=lambda x:str(x[0]))[0]
   original=plan_from_route_summary(summary);retimer=Retimer(cat,SearchSettings(),weights=weights)
   retimer.protect_earth_leg(original)
   before=weighted_collected(original,weights);result=retimer.retime(original)
   row.update(source=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),weighted_before=before,result=result.summary())
   if result.plan is not None:
    row.update(weighted_after=weighted_collected(result.plan,weights),raw_after=result.plan.total_collected_kg)
    row['weighted_gain']=row['weighted_after']-before
    row['eligible']=row['weighted_gain']>1e-6 and row['raw_after']>=item['collected_kg']-1e-6
    (root/f'ship-{ship:02d}-plan.json').write_text(json.dumps(result.plan.summary(),indent=2))
  rows.append(row);(root/'report.json').write_text(json.dumps(dict(rows=rows,seconds=time.perf_counter()-start,complete=False),indent=2))
  print(json.dumps({k:v for k,v in row.items() if k!='result'}),flush=True)
 telemetry=dict(gpu.telemetry)
(root/'report.json').write_text(json.dumps(dict(rows=rows,seconds=time.perf_counter()-start,complete=True,telemetry=telemetry),indent=2))
