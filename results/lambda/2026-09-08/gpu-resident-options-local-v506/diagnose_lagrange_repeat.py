from pathlib import Path
import dataclasses,json,os,sys,time
import numpy as np
from types import SimpleNamespace
sys.meta_path=[f for f in sys.meta_path if f.__class__.__module__!='_editable_skbc_spacepdhcg']
sys.path.insert(0,str(Path('tests').resolve()))
from test_gtoc12_gpu_discretisation import synthetic_boundary
from spacepdhcg.gtoc12.low_thrust import ScvxSettings,solve_leg,certify_leg
from spacepdhcg.gtoc12.gpu_execution import using_gpu_execution
root=Path(sys.argv[1]);root.mkdir(exist_ok=False);rows=[]
a=synthetic_boundary();rotation=np.array([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]])
b=dataclasses.replace(a,departure_position=rotation@a.departure_position,departure_velocity=rotation@a.departure_velocity,arrival_position=rotation@a.arrival_position,arrival_velocity=rotation@a.arrival_velocity,arrival_epoch=a.arrival_epoch-.5,initial_mass=a.initial_mass-50)
policy=SimpleNamespace(gpu_execution='graph',outer_loop_backend='cuda',workers=1)
def encode(x):
 if isinstance(x,np.ndarray):return x.tolist()
 if isinstance(x,np.generic):return x.item()
 raise TypeError(type(x).__name__)
def save():(root/'solves.json').write_text(json.dumps(rows,indent=2,default=encode))
with using_gpu_execution(policy):
 for precision in ['default','tighter']:
  settings=ScvxSettings(discretisation_backend='cuda',assembly_backend='cuda',convex_solver_backend='qoco',outer_loop_backend='cuda',hold='lagrange',max_iterations=40,time_limit_s=30)
  if precision=='tighter':settings=dataclasses.replace(settings,step_tolerance=1e-9,objective_tolerance=1e-8)
  for origin in [0,1]:
   os.environ['SPACEPDHCG_TEST_GTOC12_STATE_ORIGIN']=str(origin)
   os.environ['SPACEPDHCG_TEST_QOCO_EARLY_OUTER_GRAPH']='0'
   for pool in [0,1]:
    os.environ['SPACEPDHCG_TEST_GTOC12_QOCO_POOL']=str(pool)
    for trial in range(4):
     for name,boundary in [('A',a),('B',b)]:
      start=time.perf_counter();solution=solve_leg(boundary,settings);certificate=certify_leg(solution)
      rows.append(dict(precision=precision,origin=origin,pool=pool,trial=trial,boundary=name,status=solution.status,iterations=solution.iterations,certificate=dataclasses.asdict(certificate),mass=certificate.final_mass_kg,seconds=time.perf_counter()-start,history=solution.history,reports=solution.solver_reports));save()
print(len(rows),'solves recorded',flush=True)
