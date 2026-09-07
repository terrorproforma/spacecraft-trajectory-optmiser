from pathlib import Path
import os,sys,time,json,fcntl
sys.meta_path=[f for f in sys.meta_path if f.__class__.__module__!='_editable_skbc_spacepdhcg']
import numpy as np
from spacepdhcg.gtoc12.data import load_catalogue
from spacepdhcg.gtoc12.retiming import Retimer,orders_of,build_visits
from spacepdhcg.gtoc12.search import RoutePlan,SearchSettings
from spacepdhcg.gtoc12 import lambert
out=Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=True)
plan=RoutePlan.from_summary(json.loads(Path(sys.argv[2]).read_text())[0]['plan'])
lock=open(Path.home()/'.spacepdhcg-gpu.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
dispatch=lambert.cuda_retime_dp;rows=[]
with lambert.using_lambert_backend('cuda') as gpu:
 retimer=Retimer(load_catalogue(),SearchSettings());visits=build_visits(*orders_of(plan));masses=retimer._plan_masses(plan)
 retimer._dp(visits,masses,.15)
 for repeat in range(30):
  price=[.01,.03,.1,.15,.3,.6][repeat%6];pair={}
  for native in ([False,True] if repeat%2==0 else [True,False]):
   lambert.cuda_retime_dp=dispatch if native else lambda *args:NotImplemented
   start=time.perf_counter();result=retimer._dp(visits,masses,price);elapsed=time.perf_counter()-start
   assert result is not None;pair[native]=result
   rows.append(dict(repeat=repeat,price=price,native=native,seconds=elapsed,objective=result[2]))
  assert pair[False][:2]==pair[True][:2]
  assert abs(pair[False][2]-pair[True][2])<1e-9
 telemetry=dict(gpu.telemetry)
lambert.cuda_retime_dp=dispatch
report=dict(rows=rows,telemetry=telemetry,medians={str(native):float(np.median([r['seconds'] for r in rows if r['native']==native])) for native in [False,True]})
(out/'cached-dp.json').write_text(json.dumps(report,indent=2));print(report['medians']);print(telemetry)
