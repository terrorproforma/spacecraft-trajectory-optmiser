from pathlib import Path
import os,sys,time,json,hashlib,fcntl
sys.meta_path=[f for f in sys.meta_path if f.__class__.__module__!='_editable_skbc_spacepdhcg']
import numpy as np
from spacepdhcg.gtoc12.data import load_catalogue
from spacepdhcg.gtoc12.search import RoutePlan
from spacepdhcg.gtoc12.retiming import Retimer,RetimeSettings,build_visits,orders_of
from spacepdhcg.gtoc12.gpu_lambert import GpuLambert
out=Path(sys.argv[1]);source=Path(sys.argv[2]);cores=dict(old=sys.argv[3],candidate=sys.argv[4])
plan=RoutePlan.from_summary(json.loads(source.read_text())[0]['plan']);cat=load_catalogue()
lock=open(Path.home()/'.spacepdhcg-gpu.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
rows=[];builds=[]
for step in [30.,15.,5.]:
 workers={}
 try:
  for mode in ['old','candidate']:
   os.environ['SPACEPDHCG_GTOC12_CUDA_LIBRARY']=cores[mode]
   gpu=GpuLambert();timer=Retimer(cat,settings=RetimeSettings(step_days=step))
   visits=build_visits(*orders_of(plan));masses=timer._plan_masses(plan)
   workers[mode]=(gpu,timer,visits,masses)
  for repeat in range(25):
   pair=[];price=[.01,.03,.1,.15,.5][repeat%5]
   for mode in (['old','candidate'] if repeat%2==0 else ['candidate','old']):
    gpu,timer,visits,masses=workers[mode]
    start=time.perf_counter();r=gpu.retime_dp(timer,visits,masses,price);elapsed=time.perf_counter()-start
    pair.append(r);rows.append(dict(step=step,repeat=repeat,mode=mode,seconds=elapsed,price=price,feasible=r is not None))
   assert pair[0]==pair[1],(step,repeat,pair)
  for mode,(gpu,timer,visits,masses) in workers.items():
   builds.append(dict(step=step,mode=mode,epochs=timer.lattice.count,max_tofs=max(len(timer._tofs(v.role_out)) for v in visits[:-1]),telemetry=dict(gpu.telemetry)))
 finally:
  for gpu,*_ in workers.values():gpu.close()
medians={str(step):{mode:float(np.median([r['seconds'] for r in rows if r['step']==step and r['mode']==mode and r['repeat']>0])) for mode in cores} for step in [30.,15.,5.]}
report=dict(medians=medians,rows=rows,builds=builds,libraries={mode:dict(path=p,sha256=hashlib.sha256(Path(p).read_bytes()).hexdigest()) for mode,p in cores.items()})
out.write_text(json.dumps(report,indent=2));print(json.dumps(medians,indent=2))
