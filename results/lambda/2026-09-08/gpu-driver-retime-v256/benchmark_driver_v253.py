from pathlib import Path
import os,sys,time,json,fcntl,hashlib
sys.meta_path=[f for f in sys.meta_path if f.__class__.__module__!='_editable_skbc_spacepdhcg']
import numpy as np
from spacepdhcg.gtoc12.data import load_catalogue
from spacepdhcg.gtoc12.search import RoutePlan
from spacepdhcg.gtoc12.retiming import Retimer
from spacepdhcg.gtoc12.returnsweep import ReturnSweep
from spacepdhcg.gtoc12 import lambert
out=Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=True)
source,sweep_source=Path(sys.argv[2]),Path(sys.argv[3]);plan=RoutePlan.from_summary(json.loads(source.read_text())[0]['plan'])
flown=json.loads(sweep_source.read_text())['legs'][-1];cat=load_catalogue()
lock=open(Path.home()/'.spacepdhcg-gpu.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
def timer():
 t=Retimer(cat);t.set_return_sweep(ReturnSweep(flown['from'],flown['mass_before'],np.array([flown['t0']]),np.array([flown['tf']-flown['t0']]),np.ones((1,1),bool),np.ones((1,1),bool),np.array([[flown['delta_v_km_s']]]),np.array([[flown['propellant_kg']]])));return t
rows=[];reference=None
with lambert.using_lambert_backend('cuda') as gpu:
 for scope in ['fresh','cached']:
  t=timer()
  for repeat in range(12):
   for enabled in ([False,True] if repeat%2==0 else [True,False]):
    gpu.retime_cuda_driver=enabled
    if scope=='fresh':t=timer()
    before=gpu.retime_workspace.driver_calls if gpu.retime_workspace else 0
    start=time.perf_counter();r=t.retime(plan);elapsed=time.perf_counter()-start
    assert r.plan is not None
    schedule=[(x.from_id,x.to_id,x.departure_epoch,x.arrival_epoch) for x in r.plan.legs]
    if reference is None:reference=schedule;objective=r.objective_after
    assert schedule==reference and abs(r.objective_after-objective)<1e-9
    calls=gpu.retime_workspace.driver_calls-before
    assert (calls>0)==enabled
    rows.append(dict(scope=scope,repeat=repeat,gpu_driver=enabled,seconds=elapsed,result=r.summary(),driver_calls=calls))
medians={scope:{str(mode):float(np.median([r['seconds'] for r in rows if r['scope']==scope and r['gpu_driver']==mode and r['repeat']>0])) for mode in [False,True]} for scope in ['fresh','cached']}
(out/'benchmark.json').write_text(json.dumps(dict(medians=medians,rows=rows,library_sha256=hashlib.sha256(Path(os.environ['SPACEPDHCG_GTOC12_CUDA_LIBRARY']).read_bytes()).hexdigest()),indent=2));print(json.dumps(medians,indent=2))
