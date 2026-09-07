from pathlib import Path
import os, sys, time, json, hashlib, fcntl, ctypes as ct
sys.meta_path = [f for f in sys.meta_path if f.__class__.__module__ != '_editable_skbc_spacepdhcg']
import numpy as np
from spacepdhcg.gtoc12.data import load_catalogue
from spacepdhcg.gtoc12.search import RoutePlan, SearchSettings
from spacepdhcg.gtoc12.retiming import Retimer, build_visits, orders_of
from spacepdhcg.gtoc12.returnsweep import ReturnSweep
from spacepdhcg.gtoc12 import lambert

out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
source, sweep_source = Path(sys.argv[2]), Path(sys.argv[3])
plan = RoutePlan.from_summary(json.loads(source.read_text())[0]['plan'])
flown = json.loads(sweep_source.read_text())['legs'][-1]; assert flown['certified']
cat = load_catalogue()
lock = open(Path.home()/'.spacepdhcg-gpu.lock', 'a'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

def timer():
    t = Retimer(cat, SearchSettings())
    t.set_return_sweep(ReturnSweep(flown['from'], flown['mass_before'], np.array([flown['t0']]),
        np.array([flown['tf']-flown['t0']]), np.ones((1,1), bool), np.ones((1,1), bool),
        np.array([[flown['delta_v_km_s']]]), np.array([[flown['propellant_kg']]])))
    return t

def graph_stats(gpu):
    b,l = ct.c_uint64(),ct.c_uint64(); ws = gpu.retime_workspace
    ws._check(ws.graph_stats(ws.handle,ct.byref(b),ct.byref(l)))
    return dict(builds=b.value, launches=l.value)

rows=[]; reference=None
for repeat in range(8):
    for enabled in ([False,True] if repeat%2==0 else [True,False]):
        with lambert.using_lambert_backend('cuda') as gpu:
            gpu.retime_cuda_graph=enabled
            start=time.perf_counter(); t=timer(); result=t.retime(plan); elapsed=time.perf_counter()-start
            assert result.plan is not None
            schedule=[(x.from_id,x.to_id,x.departure_epoch,x.arrival_epoch) for x in result.plan.legs]
            if reference is None: reference=schedule; objective=result.objective_after
            assert schedule==reference and abs(result.objective_after-objective)<1e-9
            rows.append(dict(repeat=repeat,graph=enabled,seconds=elapsed,result=result.summary(),
                telemetry=dict(gpu.telemetry),graph_stats=graph_stats(gpu)))
            (out/'plan.json').write_text(json.dumps(result.plan.summary(),indent=2))

cached=[]
with lambert.using_lambert_backend('cuda') as gpu:
    t=timer(); visits=build_visits(*orders_of(plan)); masses=t._plan_masses(plan)
    for repeat in range(42):
        price=[0.,0.01,0.03,0.1,0.15,0.5][repeat%6]; pair=[]
        for enabled in ([False,True] if repeat%2==0 else [True,False]):
            gpu.retime_cuda_graph=enabled
            start=time.perf_counter(); r=t._dp(visits,masses,price); elapsed=time.perf_counter()-start
            pair.append(r); cached.append(dict(repeat=repeat,graph=enabled,seconds=elapsed,price=price))
        assert pair[0]==pair[1]
    final_stats=graph_stats(gpu)
medians={name:{str(enabled):float(np.median([r['seconds'] for r in data if r['graph']==enabled and r['repeat']>0])) for enabled in [False,True]} for name,data in [('full',rows),('cached',cached)]}
report=dict(rows=rows,cached=cached,medians=medians,graph_stats=final_stats,
    library_sha256=hashlib.sha256(Path(os.environ['SPACEPDHCG_GTOC12_CUDA_LIBRARY']).read_bytes()).hexdigest(),
    input_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [source,sweep_source]})
(out/'benchmark.json').write_text(json.dumps(report,indent=2));print(json.dumps(dict(medians=medians,graph_stats=final_stats),indent=2))
