from pathlib import Path
from types import SimpleNamespace
import os,sys,json,time,dataclasses
import numpy as np
from spacepdhcg.gtoc12.low_thrust import LegBoundary,ScvxSettings,solve_leg,certify_leg
from spacepdhcg.gtoc12.gpu_execution import using_gpu_execution

mode=sys.argv[1];root=Path(sys.argv[2]);root.mkdir(parents=True,exist_ok=False)
indices=[int(v) for v in sys.argv[3].split(',')] if len(sys.argv)>3 else list(range(225))
source=Path('build/performance/grid-cache-fleet-v403/scvx-calls.json')
rows=json.loads(source.read_text());output=[]
os.environ['SPACEPDHCG_TEST_GTOC12_STATIONARY_FAILURE']=mode
with using_gpu_execution(SimpleNamespace(gpu_execution='graph',outer_loop_backend='cuda',workers=1)):
 for index in indices:
  row=rows[index];kw=dict(row['boundary'])
  for key in ['departure_position','departure_velocity','arrival_position','arrival_velocity']:kw[key]=np.array(kw[key],dtype=float)
  boundary=LegBoundary(**kw);settings=ScvxSettings(**row['settings'])
  start=time.perf_counter();solution=solve_leg(boundary,settings);seconds=time.perf_counter()-start
  result=dict(index=index,prior_status=row['status'],status=solution.status,diagnostic=solution.diagnostic,seconds=seconds,iterations=solution.iterations,accepted=solution.accepted_iterations,history=solution.history,reports=solution.solver_reports)
  if solution.status=='converged':
   start=time.perf_counter();certificate=certify_leg(solution)
   result.update(certificate=dataclasses.asdict(certificate),certificate_seconds=time.perf_counter()-start,certified=certificate.within_tolerance,mass_margin_kg=certificate.final_mass_kg-boundary.minimum_final_mass)
   np.savez_compressed(root/f'leg-{index:03d}.npz',states=solution.states_scaled,thrust=solution.thrust_n,epochs=solution.node_epochs_mjd)
  output.append(result)
  (root/'results.json').write_text(json.dumps(output,indent=2,default=lambda v:v.tolist()))
  print(json.dumps({k:v for k,v in result.items() if k not in ['history','reports','certificate']}),flush=True)
(root/'summary.json').write_text(json.dumps(dict(complete=True,cases=len(output),seconds=sum(v['seconds'] for v in output),converged=sum(v['status']=='converged' for v in output),stationary_failures=sum('stationary penalized' in str(v['diagnostic']) for v in output),lost_prior=[v['index'] for v in output if v['prior_status']=='converged' and v['status']!='converged']),indent=2))
