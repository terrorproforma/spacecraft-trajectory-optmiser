from pathlib import Path
import os,sys,json,time,fcntl,hashlib,ast
root=Path('/home/ubuntu/spacepdhcg-hop-v210');integrated=Path('/home/ubuntu/spacepdhcg-regularization-v205');repo=integrated/'repo'
sys.meta_path=[f for f in sys.meta_path if f.__class__.__module__!='_editable_skbc_spacepdhcg'];sys.path.insert(0,str(repo/'src'))
for key in list(os.environ):
 if key.startswith('SPACEPDHCG_TEST_'):del os.environ[key]
core=integrated/'core-build/cuda/libspacepdhcg_cuda.so';qoco=Path('/home/ubuntu/spacepdhcg-diagnose-v174/final/libqoco.so')
os.environ.update(SPACEPDHCG_GTOC12_CUDA_LIBRARY=str(core),SPACEPDHCG_QOCO_LIBRARY=str(qoco),SPACEPDHCG_GTOC12_DATA='/home/ubuntu/spacepdhcg/gtoc12/benchmarks/gtoc12/data',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
# cuDSS is dynamically opened by the adapter using its configured absolute paths.
import numpy as np
from spacepdhcg.gtoc12.low_thrust import LegBoundary,ScvxSettings,solve_leg,certify_leg
from spacepdhcg.gtoc12.data import load_catalogue
from spacepdhcg.gtoc12.ephemeris import earth_state,asteroid_state
catalogue=load_catalogue();r0,v0=asteroid_state(catalogue,45738,65428.);rf,vf=asteroid_state(catalogue,25792,65608.)
boundary=LegBoundary(65428.,r0,v0,65608.,rf,vf,2172.7365942276165)
report=dict(pid=os.getpid(),start=time.time(),boundary=dict(t0=65428.,tf=65608.,from_asteroid=45738,to_asteroid=25792,initial_mass=2172.7365942276165,minimum_final_mass=500.,sha256=hashlib.sha256(np.r_[r0,v0,rf,vf].tobytes()).hexdigest()),rows=[])
lock=open('/home/ubuntu/.spacepdhcg-gpu.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
try:
 for origin,ruiz in [(False,0),(True,0),(False,5),(True,5)]:
  os.environ['SPACEPDHCG_TEST_GTOC12_STATE_ORIGIN']=str(int(origin))
  settings=ScvxSettings(node_days=2.,max_iterations=40,discretisation_backend='cuda',assembly_backend='cuda',convex_solver_backend='qoco',outer_loop_backend='cuda',seed_backend='cuda',qoco_ruiz_iterations=ruiz)
  for repeat in range(2):
   solution=solve_leg(boundary,settings);certificate=certify_leg(solution)
   row=dict(origin=origin,ruiz=ruiz,repeat=repeat,status=solution.status,diagnostic=solution.diagnostic,seconds=solution.solve_seconds,iterations=solution.iterations,accepted=solution.accepted_iterations,virtual_inf=solution.virtual_inf,defect=solution.max_defect,final_mass=solution.final_mass_kg,qualified=certificate.within_tolerance,position_error_km=certificate.position_error_km,velocity_error_km_s=certificate.velocity_error_km_s,maximum_thrust_n=certificate.maximum_thrust_n,history=solution.history,conic_reports=solution.solver_reports)
   report['rows'].append(row);(root/'report.json').write_text(json.dumps(report,indent=2))
   print(json.dumps({k:v for k,v in row.items() if k not in ['history','conic_reports']}),flush=True)
 report['complete']=True
except Exception as e:report.update(complete=False,error=str(e))
report['end']=time.time();(root/'report.json').write_text(json.dumps(report,indent=2))
