from pathlib import Path
import os,sys,json,time,fcntl,hashlib,traceback
root=Path('/home/ubuntu/spacepdhcg-sweep-cert-v235');repo=Path('/home/ubuntu/spacepdhcg-sweeps-v234/repo')
sys.meta_path=[f for f in sys.meta_path if f.__class__.__module__!='_editable_skbc_spacepdhcg'];sys.path.insert(0,str(repo/'src'))
from spacepdhcg.gtoc12.data import load_catalogue,load_bonus_table
from spacepdhcg.gtoc12.search import RoutePlan,SearchSettings
from spacepdhcg.gtoc12.retiming import Retimer
from spacepdhcg.gtoc12.returnsweep import ReturnSweep
import numpy as np
from spacepdhcg.gtoc12.lambert import using_lambert_backend
from spacepdhcg.gtoc12.pipeline import refine_route,write_route_artifacts
from spacepdhcg.gtoc12.low_thrust import ScvxSettings
from spacepdhcg.gtoc12.verifier import Gtoc12Verifier
from spacepdhcg.gtoc12.official import run_official_verifier
lock=open('/home/ubuntu/.spacepdhcg-gpu.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
report=dict(pid=os.getpid(),start=time.time(),complete=False,rows=[],source_sha256={n:hashlib.sha256((repo/n).read_bytes()).hexdigest() for n in ['src/spacepdhcg/gtoc12/retiming.py','cpp/cuda/src/native_qoco_adapter.cpp']})
try:
 cat=load_catalogue();original=RoutePlan.from_summary(json.loads(Path('/home/ubuntu/spacepdhcg-native-campaign-v209/output/ship_01/refinements.json').read_text())[0]['plan'])
 with using_lambert_backend('cuda') as gpu:
  flown=json.loads((repo/'input-return.json').read_text())['legs'][-1];assert flown['certified']
  timer=Retimer(cat,SearchSettings());timer.set_return_sweep(ReturnSweep(flown['from'],flown['mass_before'],np.array([flown['t0']]),np.array([flown['tf']-flown['t0']]),np.ones((1,1),dtype=bool),np.ones((1,1),dtype=bool),np.array([[flown['delta_v_km_s']]]),np.array([[flown['propellant_kg']]])))
  result=timer.retime(original)
 report['retime_gpu']=dict(gpu.telemetry);report['retiming']=result.summary();assert result.improved
 (root/'plan.json').write_text(json.dumps(result.plan.summary(),indent=2))
 settings=ScvxSettings(node_days=2.,max_iterations=40,seed_backend='cuda',discretisation_backend='cuda',assembly_backend='cuda',convex_solver_backend='qoco',outer_loop_backend='cuda')
 route=refine_route(result.plan,cat,scvx=settings);report['refinement']=route.summary()
 if route.certified:
  artifacts=write_route_artifacts(route,cat,root/'output');solution=Path(artifacts['solution'])
  report['independent']=Gtoc12Verifier(cat,bonus=load_bonus_table()).verify_file(solution).summary()
  report['official']=run_official_verifier(solution).summary()
 report['complete']=True
except Exception:report['error']=traceback.format_exc()
report['end']=time.time();(root/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='refinement'}),flush=True)
