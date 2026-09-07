from pathlib import Path
import os,sys,json,time,fcntl,hashlib,ast
root=Path('/home/ubuntu/spacepdhcg-arc-snapshot-v307');repo=Path('/home/ubuntu/spacepdhcg-collect-resident-v298/repo')
sys.meta_path=[f for f in sys.meta_path if f.__class__.__module__!='_editable_skbc_spacepdhcg'];sys.path.insert(0,str(repo/'src'))
for key in list(os.environ):
 if key.startswith('SPACEPDHCG_TEST_'):del os.environ[key]
core=Path('/home/ubuntu/spacepdhcg-collect-resident-v298/core-build/cuda/libspacepdhcg_cuda.so');qoco=Path('/home/ubuntu/spacepdhcg-diagnose-v174/final/libqoco.so')
os.environ['LD_LIBRARY_PATH']=str(core.parent)+':'+str(qoco.parent)+':/home/ubuntu/spacepdhcg-recovery-v152/cudss/lib:/usr/local/cuda/lib64'
os.environ.update(SPACEPDHCG_GTOC12_CUDA_LIBRARY=str(core),SPACEPDHCG_QOCO_LIBRARY=str(qoco),SPACEPDHCG_GTOC12_DATA='/home/ubuntu/spacepdhcg/gtoc12/benchmarks/gtoc12/data',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
# cuDSS is dynamically opened by the adapter using its configured absolute paths.
os.environ['SPACEPDHCG_QOCO_SNAPSHOT_DIRECTORY']=str(root/'snapshots')
import numpy as np
from spacepdhcg.gtoc12.low_thrust import LegBoundary,ScvxSettings,solve_leg,certify_leg
from spacepdhcg.gtoc12.data import load_catalogue
from spacepdhcg.gtoc12.ephemeris import earth_state,asteroid_state
catalogue=load_catalogue();r0,v0=earth_state(64343.);rf,vf=asteroid_state(catalogue,57530,64808.)
boundary=LegBoundary(64343.,r0,v0,64808.,rf,vf,3000.,free_departure_vinf=True)
report=dict(pid=os.getpid(),start=time.time(),boundary=dict(t0=64343.,tf=64808.,asteroid=57530,initial_mass=3000.,minimum_final_mass=500.,sha256=hashlib.sha256(np.r_[r0,v0,rf,vf].tobytes()).hexdigest()),rows=[])
lock=open('/home/ubuntu/.spacepdhcg-gpu.lock','a');fcntl.flock(lock,fcntl.LOCK_EX)
try:
 for origin,ruiz in [(False,0)]:

  settings=ScvxSettings(node_days=2.,max_iterations=40,discretisation_backend='cuda',assembly_backend='cuda',convex_solver_backend='qoco',outer_loop_backend='cuda',seed_backend='cuda',qoco_ruiz_iterations=ruiz)
  for repeat in range(1):
   solution=solve_leg(boundary,settings);certificate=certify_leg(solution)
   row=dict(origin=origin,ruiz=ruiz,repeat=repeat,status=solution.status,diagnostic=solution.diagnostic,seconds=solution.solve_seconds,iterations=solution.iterations,accepted=solution.accepted_iterations,virtual_inf=solution.virtual_inf,defect=solution.max_defect,final_mass=solution.final_mass_kg,qualified=certificate.within_tolerance,position_error_km=certificate.position_error_km,velocity_error_km_s=certificate.velocity_error_km_s,maximum_thrust_n=certificate.maximum_thrust_n,history=solution.history,conic_reports=solution.solver_reports)
   report['rows'].append(row);(root/'report.json').write_text(json.dumps(report,indent=2))
   print(json.dumps({k:v for k,v in row.items() if k not in ['history','conic_reports']}),flush=True)
 report['complete']=True
except Exception as e:report.update(complete=False,error=str(e))
report['end']=time.time();(root/'report.json').write_text(json.dumps(report,indent=2))
