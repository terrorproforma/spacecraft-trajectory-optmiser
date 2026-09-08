from pathlib import Path
import dataclasses,os,time,json,runpy
import numpy as np
from spacepdhcg.gtoc12 import gpu_scvx
root=Path(os.environ['SPACEPDHCG_PHASE_OUTPUT']);rows=[];original=gpu_scvx.solve_native
def measured(*args,**kwargs):
    start=time.perf_counter();value=original(*args,**kwargs)
    rows.append(dict(seconds=time.perf_counter()-start,boundary=dataclasses.asdict(value.boundary),settings=dataclasses.asdict(args[1]),nodes=len(value.node_epochs_mjd),status=value.status,diagnostic=value.diagnostic,iterations=value.iterations,accepted=value.accepted_iterations,solver_reports=value.solver_reports))
    return value
gpu_scvx.solve_native=measured
try:runpy.run_module('spacepdhcg',run_name='__main__')
finally:
    def encode(value):
        if isinstance(value,np.ndarray):return value.tolist()
        if isinstance(value,np.generic):return value.item()
        raise TypeError(type(value).__name__)
    (root/'calls.json').write_text(json.dumps(rows,indent=2,default=encode))
