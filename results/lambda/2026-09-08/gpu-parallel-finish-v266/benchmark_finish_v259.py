from pathlib import Path
import os,sys,time,json,fcntl,hashlib
sys.meta_path=[f for f in sys.meta_path if f.__class__.__module__!='_editable_skbc_spacepdhcg']
import numpy as np
from spacepdhcg.gtoc12.data import load_catalogue
from spacepdhcg.gtoc12.search import RoutePlan
from spacepdhcg.gtoc12.retiming import Retimer
from spacepdhcg.gtoc12.returnsweep import ReturnSweep
from spacepdhcg.gtoc12.gpu_lambert import GpuLambert
from spacepdhcg.gtoc12 import lambert
out=Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=True)
source,sweep_source=Path(sys.argv[2]),Path(sys.argv[3]);cores=dict(old=sys.argv[4],candidate=sys.argv[5])
plan=RoutePlan.from_summary(json.loads(source.read_text())[0]['plan'])
flown=json.loads(sweep_source.read_text())['legs'][-1];cat=load_catalogue()
lock=open(Path.home()/'.spacepdhcg-gpu.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
def timer():
 t=Retimer(cat);t.set_return_sweep(ReturnSweep(flown['from'],flown['mass_before'],np.array([flown['t0']]),np.array([flown['tf']-flown['t0']]),np.ones((1,1),bool),np.ones((1,1),bool),np.array([[flown['delta_v_km_s']]]),np.array([[flown['propellant_kg']]])));return t
workers={};rows=[];reference=None
try:
 for mode,core in cores.items():
  os.environ['SPACEPDHCG_GTOC12_CUDA_LIBRARY']=core;workers[mode]=GpuLambert()
 for scope in ['fresh','cached']:
  timers={mode:timer() for mode in cores}
  for repeat in range(12):
   for mode in (['old','candidate'] if repeat%2==0 else ['candidate','old']):
    gpu=workers[mode];token=lambert._GPU_BACKEND.set(gpu)
    try:
     if scope=='fresh':timers[mode]=timer()
     before=gpu.retime_workspace.driver_calls if gpu.retime_workspace else 0
     start=time.perf_counter();r=timers[mode].retime(plan);elapsed=time.perf_counter()-start
     assert r.plan is not None
     schedule=[(x.from_id,x.to_id,x.departure_epoch,x.arrival_epoch) for x in r.plan.legs]
     if reference is None:reference=schedule;objective=r.objective_after
     assert schedule==reference and r.objective_after==objective
     assert gpu.retime_workspace.driver_calls-before==1
     rows.append(dict(scope=scope,repeat=repeat,mode=mode,seconds=elapsed,result=r.summary()))
    finally:lambert._GPU_BACKEND.reset(token)
finally:
 for gpu in workers.values():gpu.close()
medians={scope:{mode:float(np.median([r['seconds'] for r in rows if r['scope']==scope and r['mode']==mode and r['repeat']>0])) for mode in cores} for scope in ['fresh','cached']}
(out/'benchmark.json').write_text(json.dumps(dict(medians=medians,rows=rows,libraries={mode:dict(path=p,sha256=hashlib.sha256(Path(p).read_bytes()).hexdigest()) for mode,p in cores.items()}),indent=2));print(json.dumps(medians,indent=2))
