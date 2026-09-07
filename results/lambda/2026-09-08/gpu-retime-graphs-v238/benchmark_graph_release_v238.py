from pathlib import Path
import os,sys,time,json,hashlib,fcntl,importlib.util
sys.meta_path=[f for f in sys.meta_path if f.__class__.__module__!='_editable_skbc_spacepdhcg']
import numpy as np
from spacepdhcg.gtoc12.data import load_catalogue
from spacepdhcg.gtoc12.search import RoutePlan,SearchSettings
from spacepdhcg.gtoc12.retiming import Retimer,build_visits,orders_of
from spacepdhcg.gtoc12.returnsweep import ReturnSweep
from spacepdhcg.gtoc12 import lambert,gpu_retime
from spacepdhcg.gtoc12.gpu_lambert import GpuLambert
out=Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=True)
source,sweep_source,legacy_source=map(Path,sys.argv[2:5]);old_core,new_core=sys.argv[5:7]
spec=importlib.util.spec_from_file_location('spacepdhcg.gtoc12.legacy_gpu_retime',legacy_source)
legacy=importlib.util.module_from_spec(spec);spec.loader.exec_module(legacy)
classes={'old':legacy.GpuRetime,'graph':gpu_retime.GpuRetime};cores={'old':old_core,'graph':new_core}
plan=RoutePlan.from_summary(json.loads(source.read_text())[0]['plan'])
flown=json.loads(sweep_source.read_text())['legs'][-1];assert flown['certified']
cat=load_catalogue();lock=open(Path.home()/'.spacepdhcg-gpu.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
def timer():
 t=Retimer(cat,SearchSettings());t.set_return_sweep(ReturnSweep(flown['from'],flown['mass_before'],np.array([flown['t0']]),np.array([flown['tf']-flown['t0']]),np.ones((1,1),bool),np.ones((1,1),bool),np.array([[flown['delta_v_km_s']]]),np.array([[flown['propellant_kg']]])));return t
rows=[];reference=None
for repeat in range(8):
 for mode in (['old','graph'] if repeat%2==0 else ['graph','old']):
  gpu_retime.GpuRetime=classes[mode];os.environ['SPACEPDHCG_GTOC12_CUDA_LIBRARY']=cores[mode]
  with lambert.using_lambert_backend('cuda') as gpu:
   start=time.perf_counter();t=timer();r=t.retime(plan);elapsed=time.perf_counter()-start
   assert r.plan is not None
   schedule=[(x.from_id,x.to_id,x.departure_epoch,x.arrival_epoch) for x in r.plan.legs]
   if reference is None:reference=schedule;objective=r.objective_after
   assert schedule==reference and abs(r.objective_after-objective)<1e-9
   rows.append(dict(repeat=repeat,mode=mode,seconds=elapsed,objective=r.objective_after,telemetry=dict(gpu.telemetry)))
cached=[];workers={}
try:
 for mode in ['old','graph']:
  os.environ['SPACEPDHCG_GTOC12_CUDA_LIBRARY']=cores[mode]
  gpu=GpuLambert();gpu.retime_workspace=classes[mode](gpu.library,gpu.device_id,gpu)
  t=timer();workers[mode]=(gpu,t,build_visits(*orders_of(plan)),t._plan_masses(plan))
 for repeat in range(42):
  price=[0.,.01,.03,.1,.15,.5][repeat%6];pair=[]
  for mode in (['old','graph'] if repeat%2==0 else ['graph','old']):
   gpu,t,visits,masses=workers[mode]
   start=time.perf_counter();r=gpu.retime_dp(t,visits,masses,price);elapsed=time.perf_counter()-start
   pair.append(r);cached.append(dict(repeat=repeat,mode=mode,seconds=elapsed,price=price))
  assert pair[0]==pair[1]
finally:
 for gpu,*_ in workers.values():gpu.close()
medians={name:{mode:float(np.median([r['seconds'] for r in data if r['mode']==mode and r['repeat']>0])) for mode in ['old','graph']} for name,data in [('full',rows),('cached',cached)]}
report=dict(medians=medians,rows=rows,cached=cached,source_sha256=hashlib.sha256(legacy_source.read_text().encode()).hexdigest(),libraries={mode:dict(path=p,sha256=hashlib.sha256(Path(p).read_bytes()).hexdigest()) for mode,p in cores.items()})
(out/'benchmark.json').write_text(json.dumps(report,indent=2));print(json.dumps(medians,indent=2))
