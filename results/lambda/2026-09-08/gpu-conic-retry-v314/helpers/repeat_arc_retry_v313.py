from pathlib import Path
import os,sys,json,time,fcntl,hashlib
sys.meta_path=[f for f in sys.meta_path if f.__class__.__module__!='_editable_skbc_spacepdhcg'];sys.path.insert(0,str(Path('src').resolve()))
import numpy as np
from spacepdhcg.gtoc12.low_thrust import LegBoundary,ScvxSettings,solve_leg,certify_leg
from spacepdhcg.gtoc12.data import load_catalogue
from spacepdhcg.gtoc12.ephemeris import earth_state,asteroid_state
root=Path('build/performance/arc-retry-v313');root.mkdir(exist_ok=False)
catalogue=load_catalogue();r0,v0=earth_state(64343.);rf,vf=asteroid_state(catalogue,57530,64808.)
boundary=LegBoundary(64343.,r0,v0,64808.,rf,vf,3000.,free_departure_vinf=True)
settings=ScvxSettings(node_days=2.,max_iterations=40,discretisation_backend='cuda',assembly_backend='cuda',convex_solver_backend='qoco',outer_loop_backend='cuda',seed_backend='cuda')
report=dict(pid=os.getpid(),complete=False,boundary_sha256=hashlib.sha256(np.r_[r0,v0,rf,vf].tobytes()).hexdigest(),rows=[])
with open('/home/angus/.spacepdhcg-gpu.lock','a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 for repeat in range(24):
  sol=solve_leg(boundary,settings);cert=certify_leg(sol)
  row=dict(repeat=repeat,status=sol.status,qualified=cert.within_tolerance,seconds=sol.solve_seconds,iterations=sol.iterations,accepted=sol.accepted_iterations,virtual_inf=sol.virtual_inf,defect=sol.max_defect,retries=sum(bool(x.get('retry_unchanged')) for x in sol.history),history=sol.history,conic_reports=sol.solver_reports)
  report['rows'].append(row);(root/'report.json').write_text(json.dumps(report,indent=2));print({k:v for k,v in row.items() if k not in ['history','conic_reports']},flush=True)
report['complete']=True;(root/'report.json').write_text(json.dumps(report,indent=2))
