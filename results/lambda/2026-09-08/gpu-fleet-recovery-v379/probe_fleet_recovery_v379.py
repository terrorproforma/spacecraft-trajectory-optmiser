from pathlib import Path
import os,json,time,fcntl,hashlib
from spacepdhcg.cli import build_parser
from spacepdhcg.gtoc12.cli import catalogue_pool,_scvx_settings
from spacepdhcg.gtoc12.data import load_catalogue,load_bonus_table
from spacepdhcg.gtoc12.search import RouteSearch,SearchSettings,RoutePlan
from spacepdhcg.gtoc12.pipeline import refine_route,write_route_artifacts
from spacepdhcg.gtoc12.gpu_execution import using_gpu_execution
from spacepdhcg.gtoc12.lambert import using_lambert_backend
from spacepdhcg.gtoc12.verifier import Gtoc12Verifier
from spacepdhcg.gtoc12.official import run_official_verifier
root=Path('build/performance/fleet-recovery-v379');root.mkdir(exist_ok=False)
old=json.loads(Path('results/lambda/2026-09-08/gpu-fleet-v374/spacepdhcg-fleet-search-v374/output/run_report.json').read_text())
args=build_parser().parse_args(['gtoc12','run','--run-id','recovery379','--output',str(root),'--full-catalogue','--ships','1','--beam-width','32','--max-deploys','10','--neighbours','48','--no-cooperative','--node-days','2','--scvx-iterations','40','--screening-backend','cuda','--seed-backend','cuda','--discretisation-backend','cuda','--assembly-backend','cuda','--convex-solver','qoco','--outer-loop-backend','cuda','--gpu-execution','graph'])
report=dict(pid=os.getpid(),complete=False,attempts=[],scope='Diagnostic reordering only. Original GPU solver and physics gates. A failure is not an infeasibility proof.')
def save():(root/'report.json').write_text(json.dumps(report,indent=2))
save();catalogue=load_catalogue();bonus=load_bonus_table();weights={int(a):float(bonus.coefficient[a-1]) for a in catalogue.ids}
prefix=lambda p:tuple((l.from_id,l.to_id,l.departure_epoch,l.arrival_epoch) for l in p.legs[:3])
failed_prefix=prefix(RoutePlan.from_summary(old['ships'][3]['refinements'][0]['plan']))
with open('/home/angus/.spacepdhcg-gpu.lock','a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 with using_gpu_execution(args),using_lambert_backend('cuda') as gpu:
  start=time.perf_counter()
  search=RouteSearch(catalogue,catalogue_pool(catalogue,args),SearchSettings(beam_width=32,max_deploys=10,neighbours=48,time_budget_seconds=120),excluded=set(old['fleet']['asteroids']),weights=weights)
  result=search.run();report['search_seconds']=time.perf_counter()-start;report['candidates']=len(result.candidates)
  (root/'candidates.json').write_text(json.dumps([p.summary() for p in result.candidates]))
  seen={failed_prefix};selected=[(0,result.candidates[0],'baseline_prefix')]
  for rank,plan in enumerate(result.candidates):
   key=prefix(plan)
   if key in seen:continue
   selected.append((rank,plan,'different_prefix'));seen.add(key)
   if len(selected)==6:break
  report['selected_ranks']=[r for r,_,_ in selected];report['baseline_prefix_matches']=prefix(result.candidates[0])==failed_prefix;save();print('Candidates',len(result.candidates),'selected',report['selected_ranks'],flush=True)
  for rank,plan,mode in selected:
   start=time.perf_counter();refined=refine_route(plan,catalogue,scvx=_scvx_settings(args))
   entry=dict(rank=rank,mode=mode,seconds=time.perf_counter()-start,refined=refined.summary())
   if refined.certified:
    artifacts=write_route_artifacts(refined,catalogue,root/f'candidate-{rank:03d}');solution=Path(artifacts['solution'])
    entry['official']=run_official_verifier(solution).summary();entry['independent']=Gtoc12Verifier(catalogue,bonus=bonus).verify_file(solution).summary();entry['artifacts']=artifacts
   report['attempts'].append(entry);save();print(rank,mode,refined.certified,refined.summary().get('failures'),flush=True)
   if mode=='different_prefix' and refined.certified:break
  report['screening']=gpu.telemetry;report['complete']=True;save()
