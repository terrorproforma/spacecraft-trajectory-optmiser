import sys,runpy,json,dataclasses,hashlib
from pathlib import Path
import numpy as np
sys.meta_path=[f for f in sys.meta_path if f.__class__.__module__!='_editable_skbc_spacepdhcg']
from spacepdhcg.gtoc12 import gpu_scvx
original=gpu_scvx.solve_native
def captured(boundary,settings,*args,**kwargs):
 solution=original(boundary,settings,*args,**kwargs)
 if boundary.departure_epoch==64343.0 and boundary.arrival_epoch==64808.0:
  data=dict(settings=dataclasses.asdict(settings),boundary={k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in dataclasses.asdict(boundary).items()},status=solution.status,diagnostic=solution.diagnostic,history=solution.history,conic_reports=solution.solver_reports,iterations=solution.iterations,accepted=solution.accepted_iterations,virtual_inf=solution.virtual_inf,defect=solution.max_defect,seconds=solution.solve_seconds)
  Path('/home/ubuntu/spacepdhcg-resident-capture-v306/arc.json').write_text(json.dumps(data,indent=2))
 return solution
gpu_scvx.solve_native=captured
runpy.run_module('spacepdhcg',run_name='__main__')
