from pathlib import Path
import dataclasses,json,os,sys,time
import numpy as np
from types import SimpleNamespace
from scipy.integrate import solve_ivp
sys.meta_path=[f for f in sys.meta_path if f.__class__.__module__!='_editable_skbc_spacepdhcg']
sys.path.insert(0,str(Path('tests').resolve()))
from test_gtoc12_gpu_discretisation import synthetic_boundary
from spacepdhcg.gtoc12.low_thrust import ScvxSettings,solve_leg,certify_leg
from spacepdhcg.gtoc12.gpu_execution import using_gpu_execution
from spacepdhcg.gtoc12.verifier import LagrangeThrust,_thrust_dynamics
from spacepdhcg.gtoc12 import constants as C
root=Path(sys.argv[1]);root.mkdir(exist_ok=False);rows=[]
a=synthetic_boundary();rotation=np.array([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]])
b=dataclasses.replace(a,departure_position=rotation@a.departure_position,departure_velocity=rotation@a.departure_velocity,arrival_position=rotation@a.arrival_position,arrival_velocity=rotation@a.arrival_velocity,arrival_epoch=a.arrival_epoch-.5,initial_mass=a.initial_mass-50)
settings=ScvxSettings(discretisation_backend='cuda',assembly_backend='cuda',convex_solver_backend='qoco',outer_loop_backend='cuda',hold='lagrange',max_iterations=40,time_limit_s=30)
with using_gpu_execution(SimpleNamespace(gpu_execution='graph',outer_loop_backend='cuda',workers=1)):
 for origin in [0,1]:
  os.environ['SPACEPDHCG_TEST_GTOC12_STATE_ORIGIN']=str(origin)
  os.environ['SPACEPDHCG_TEST_QOCO_EARLY_OUTER_GRAPH']='0'
  for pool in [0,1]:
   os.environ['SPACEPDHCG_TEST_GTOC12_QOCO_POOL']=str(pool)
   for trial in range(4):
    solution=solve_leg(b,settings);certificate=certify_leg(solution);assert solution.converged and certificate.within_tolerance
    arc=solution.burn_arcs()[0];epochs,thrust=arc.interior_arrays();times=(epochs-b.departure_epoch)*C.DAY_S
    interpolant=LagrangeThrust(times,thrust,C.THRUST_INTERPOLATION_ORDER)
    y0=np.concatenate((b.departure_position,solution.departure_ship_velocity_km_s(),[b.initial_mass]))
    np.savez_compressed(root/(str(len(rows))+'.npz'),times=times,thrust=thrust,y0=y0,target_position=b.arrival_position,target_velocity=solution.arrival_ship_velocity_km_s(),gpu_final_state=solution.states_scaled[-1])
    masses=[]
    for points in [16,32]:
     x,w=np.polynomial.legendre.leggauss(points);impulse=0.
     for lo,hi in zip(times[:-1],times[1:]):
      impulse+=(hi-lo)*.5*sum(weight*np.linalg.norm(interpolant((lo+hi)*.5+(hi-lo)*.5*node)) for node,weight in zip(x,w))
     masses.append(b.initial_mass-C.MASS_FLOW_PER_NEWTON_KG_S*impulse)
    start=time.perf_counter();state=y0.copy();rhs=_thrust_dynamics(interpolant)
    for lo,hi in zip(times[:-1],times[1:]):
     integrated=solve_ivp(rhs,(lo,hi),state,method='DOP853',rtol=1e-12,atol=np.array([1e-7]*3+[1e-10]*3+[1e-9]));assert integrated.success
     state=integrated.y[:,-1]
    rows.append(dict(origin=origin,pool=pool,trial=trial,old_mass=certificate.final_mass_kg,old_position_km=certificate.position_error_km,segmented_mass=float(state[6]),segmented_position_km=float(np.linalg.norm(state[:3]-b.arrival_position)),quadrature16_mass=float(masses[0]),quadrature32_mass=float(masses[1]),gpu_mass=float(solution.states_scaled[-1,6]*b.initial_mass),segmented_seconds=time.perf_counter()-start))
    (root/'analysis.json').write_text(json.dumps(rows,indent=2))
print(json.dumps(dict(rows=len(rows),max_old_mass_error=max(abs(r['old_mass']-r['quadrature32_mass']) for r in rows),max_segmented_mass_error=max(abs(r['segmented_mass']-r['quadrature32_mass']) for r in rows),max_quadrature_difference=max(abs(r['quadrature16_mass']-r['quadrature32_mass']) for r in rows)),indent=2))
