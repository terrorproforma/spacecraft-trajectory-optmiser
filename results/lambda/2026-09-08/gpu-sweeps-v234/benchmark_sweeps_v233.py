from pathlib import Path
import os,sys,time,json,hashlib,fcntl
sys.meta_path=[f for f in sys.meta_path if f.__class__.__module__!='_editable_skbc_spacepdhcg']
import numpy as np
from spacepdhcg.gtoc12.data import load_catalogue
from spacepdhcg.gtoc12.search import RoutePlan,SearchSettings
from spacepdhcg.gtoc12.retiming import Retimer
from spacepdhcg.gtoc12.returnsweep import ReturnSweep
from spacepdhcg.gtoc12 import lambert
out=Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=True)
source=Path(sys.argv[2]);plan=RoutePlan.from_summary(json.loads(source.read_text())[0]['plan'])
sweep_source=Path(sys.argv[3]);flown=json.loads(sweep_source.read_text())['legs'][-1];assert flown['certified']
cat=load_catalogue();dispatch=lambert.cuda_retime_dp
lock=open(Path.home()/'.spacepdhcg-gpu.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
rows=[];reference=None
for repeat in range(6):
 for native in ([False,True] if repeat%2==0 else [True,False]):
  pass
  with lambert.using_lambert_backend('cuda') as gpu:
   gpu.resident_retime_tables=native
   start=time.perf_counter();timer=Retimer(cat,SearchSettings());timer.set_return_sweep(ReturnSweep(flown['from'],flown['mass_before'],np.array([flown['t0']]),np.array([flown['tf']-flown['t0']]),np.ones((1,1),dtype=bool),np.ones((1,1),dtype=bool),np.array([[flown['delta_v_km_s']]]),np.array([[flown['propellant_kg']]])));result=timer.retime(plan);elapsed=time.perf_counter()-start
   assert result.plan is not None
   (out/'plan.json').write_text(json.dumps(result.plan.summary(),indent=2))
   schedule=[(x.from_id,x.to_id,x.departure_epoch,x.arrival_epoch) for x in result.plan.legs]
   if reference is None:reference=schedule;objective=result.objective_after
   assert reference==schedule
   assert abs(result.objective_after-objective)<1e-8
   rows.append(dict(repeat=repeat,native=native,seconds=elapsed,result=result.summary(),telemetry=dict(gpu.telemetry)))
   print(json.dumps({k:v for k,v in rows[-1].items() if k not in ['result','telemetry']}),flush=True)
lambert.cuda_retime_dp=dispatch
(out/'benchmark.json').write_text(json.dumps(dict(rows=rows,schedule=reference,sweep_source_sha256=hashlib.sha256(sweep_source.read_bytes()).hexdigest(),sweep_sample=flown,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),library_sha256=hashlib.sha256(Path(os.environ['SPACEPDHCG_GTOC12_CUDA_LIBRARY']).read_bytes()).hexdigest(),medians={str(native):float(np.median([x['seconds'] for x in rows if x['native']==native and x['repeat']>0])) for native in [False,True]}),indent=2))
