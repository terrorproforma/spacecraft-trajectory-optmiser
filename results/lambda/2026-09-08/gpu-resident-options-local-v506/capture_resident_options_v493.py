from pathlib import Path
import dataclasses,hashlib,json,runpy,time
import numpy as np
from spacepdhcg.gtoc12 import gpu_collection
root=Path('build/performance/resident-options-capture-v493')
original=gpu_collection.GpuCollection.select
tables={};queries=[]
def capture(self,options,mass,epoch,settings,penalty_scale=1.0,max_span=np.inf,*,first=False):
    rows=np.ascontiguousarray(options,dtype=np.float64).reshape(-1,3)
    key=hashlib.sha256(rows.tobytes()).hexdigest()
    if key not in tables:tables[key]=rows.copy()
    start=time.perf_counter()
    result=original(self,options,mass,epoch,settings,penalty_scale,max_span,first=first)
    queries.append(dict(key=key,count=len(rows),mass=float(mass),epoch=float(epoch),penalty_scale=float(penalty_scale),max_span=float(max_span),first=bool(first),settings={name:getattr(settings,name) for name in ['hop_inflation_slope','earth_return_authority_ratio','hop_authority_ratio','hop_inflation','hop_inflation_floor','wait_penalty']},expected=result,native_wrapper_seconds=time.perf_counter()-start))
    return result
gpu_collection.GpuCollection.select=capture
try:runpy.run_module('spacepdhcg',run_name='__main__')
finally:
    np.savez_compressed(root/'tables.npz',**tables)
    (root/'queries.json').write_text(json.dumps(queries,indent=2))
    (root/'capture-summary.json').write_text(json.dumps(dict(scope='Instrumented selection input/output capture, not a throughput benchmark. Host input bytes count option rows uploaded by each existing selection call, including repeated cached tables.',queries=len(queries),unique_tables=len(tables),unique_bytes=sum(x.nbytes for x in tables.values()),selection_upload_bytes=sum(q['count']*24 for q in queries),wrapper_seconds=sum(q['native_wrapper_seconds'] for q in queries)),indent=2))
