import sys,time,json,dataclasses,hashlib
from pathlib import Path
sys.meta_path=[f for f in sys.meta_path if f.__class__.__module__!='_editable_skbc_spacepdhcg']
from spacepdhcg.gtoc12 import collectdp,search
from spacepdhcg.cli import main

root=Path('/home/ubuntu/spacepdhcg-collect-profile-v272')
original=collectdp._solve_collect_dp
rows=[];fixtures={}
def solve(*args,**kwargs):
    start=time.perf_counter()
    try:return original(*args,**kwargs)
    finally:
        rows.append(dict(asteroids=len(args[1]),epochs=len(args[4]),seconds=time.perf_counter()-start,
            price=args[10],burn_per_hop=args[12]))
collectdp._solve_collect_dp=solve
outer=collectdp.plan_collect_tour
def capture(table,deployed,camp,camp_epoch,mass_after_deploys,**kwargs):
    start=time.perf_counter()
    result=outer(table,deployed,camp,camp_epoch,mass_after_deploys,**kwargs)
    elapsed=time.perf_counter()-start
    k=len(deployed)
    if k not in fixtures or elapsed>fixtures[k]['seconds']:
        fixtures[k]=dict(settings=dataclasses.asdict(table.settings),deployed=deployed,camp=camp,
            camp_epoch=camp_epoch,mass_after_deploys=mass_after_deploys,kwargs=kwargs,
            expected=dataclasses.asdict(result) if result is not None else None,seconds=elapsed,
            return_sweeps=bool(table.return_sweeps) if hasattr(table,'return_sweeps') else None)
    return result
collectdp.plan_collect_tour=search.plan_collect_tour=capture
start=time.perf_counter()
try:
    code=main()
finally:
    result=dict(total_seconds=time.perf_counter()-start,dp_seconds=sum(x['seconds'] for x in rows),
        dp_passes=len(rows),passes=rows,fixtures=fixtures,
        scope='Direct perf_counter around collection DP passes, including their CUDA table construction. No cProfile. Largest timed input per asteroid count retained.')
    (root/'timing.json').write_text(json.dumps(result,indent=2,default=lambda x:sorted(x) if isinstance(x,(set,frozenset)) else x.item()))
raise SystemExit(code)
