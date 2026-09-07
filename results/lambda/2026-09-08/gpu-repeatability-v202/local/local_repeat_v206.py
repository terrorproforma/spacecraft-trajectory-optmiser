from pathlib import Path
import os,sys,json,time,fcntl,hashlib,ast
root=Path('build/performance/local-repeat-v206');root.mkdir(exist_ok=False);repo=Path.cwd()
sys.meta_path=[f for f in sys.meta_path if f.__class__.__module__!='_editable_skbc_spacepdhcg'];sys.path.insert(0,str(repo/'src'))
for key in list(os.environ):
 if key.startswith('SPACEPDHCG_TEST_'):del os.environ[key]
core=Path('/home/angus/build-spacepdhcg-regularization-v205/final/libspacepdhcg_cuda.so');qoco=Path('/home/angus/build-qoco-gpu-device-ir-v137/final/libqoco.so')
os.environ['LD_LIBRARY_PATH']=str(core.parent)+':'+str(qoco.parent)+':/home/angus/cudss-isolated-0.8.0.10/nvidia/cu12/lib:/usr/local/cuda-12.8/lib64'
os.environ.update(SPACEPDHCG_GTOC12_CUDA_LIBRARY=str(core),SPACEPDHCG_QOCO_LIBRARY=str(qoco),SPACEPDHCG_GTOC12_DATA='/home/angus/worktrees/spacepdhcg-release/benchmarks/gtoc12/data',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
# cuDSS is dynamically opened by the adapter using its configured absolute paths.
import numpy as np
from spacepdhcg.gtoc12.low_thrust import LegBoundary,ScvxSettings,solve_leg,certify_leg
from spacepdhcg.gtoc12.data import load_catalogue
from spacepdhcg.gtoc12.ephemeris import earth_state,asteroid_state
catalogue=load_catalogue();r0,v0=earth_state(64328.);rf,vf=asteroid_state(catalogue,57530,64828.)
boundary=LegBoundary(64328.,r0,v0,64828.,rf,vf,3000.,free_departure_vinf=True)
report=dict(pid=os.getpid(),start=time.time(),boundary=dict(t0=64328.,tf=64828.,asteroid=57530,initial_mass=3000.,minimum_final_mass=500.,sha256=hashlib.sha256(np.r_[r0,v0,rf,vf].tobytes()).hexdigest()),rows=[])
lock=open('/home/angus/.spacepdhcg-gpu.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
try:
 for origin,ruiz in [(False,0),(True,0)]:
  os.environ['SPACEPDHCG_TEST_GTOC12_STATE_ORIGIN']=str(int(origin))
  settings=ScvxSettings(node_days=2.,max_iterations=40,discretisation_backend='cuda',assembly_backend='cuda',convex_solver_backend='qoco',outer_loop_backend='cuda',seed_backend='cuda',qoco_ruiz_iterations=ruiz)
  for repeat in range(16):
   solution=solve_leg(boundary,settings);certificate=certify_leg(solution)
   row=dict(origin=origin,ruiz=ruiz,repeat=repeat,status=solution.status,diagnostic=solution.diagnostic,seconds=solution.solve_seconds,iterations=solution.iterations,accepted=solution.accepted_iterations,virtual_inf=solution.virtual_inf,defect=solution.max_defect,final_mass=solution.final_mass_kg,qualified=certificate.within_tolerance,position_error_km=certificate.position_error_km,velocity_error_km_s=certificate.velocity_error_km_s,maximum_thrust_n=certificate.maximum_thrust_n,history=solution.history,conic_reports=solution.solver_reports)
   report['rows'].append(row);(root/'report.json').write_text(json.dumps(report,indent=2))
   print(json.dumps({k:v for k,v in row.items() if k not in ['history','conic_reports']}),flush=True)
 report['complete']=True
except Exception as e:report.update(complete=False,error=str(e))
report['end']=time.time();(root/'report.json').write_text(json.dumps(report,indent=2))
