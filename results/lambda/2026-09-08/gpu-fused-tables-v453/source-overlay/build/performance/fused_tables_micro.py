"""Replay captured real collection grids; measure construction, verify every cell."""
from pathlib import Path
from types import SimpleNamespace
import hashlib, json, os, time, sys
import numpy as np
from spacepdhcg.gtoc12.data import load_catalogue
from spacepdhcg.gtoc12.gpu_collect_tables import GpuCollectTable
from spacepdhcg.gtoc12.lambert import using_lambert_backend
from spacepdhcg.gtoc12.gpu_lambert import HopElements, body_elements
from spacepdhcg.gtoc12 import constants as C

trace=Path(sys.argv[1]); output=Path(sys.argv[2]); catalogue=load_catalogue()
queries=json.loads(trace.read_text()); rows=[]; reference=[]
for q in queries:
    q['epochs']=np.asarray(q['epochs'],dtype=np.float64)
    q['tofs']=np.asarray(q['tofs'],dtype=np.float64)
    elements=HopElements(body_elements(catalogue,q['source']),body_elements(catalogue,q['target']),C.MU_SUN_KM3_S2,0.,C.MAX_VINF_EARTH_KM_S if q['target']==0 else 0.)
    assert hashlib.sha256(bytes(elements)+q['epochs'].tobytes()+q['tofs'].tobytes()).hexdigest()==q['key']
with using_lambert_backend('cuda',maximum_batch_size=16384) as gpu:
    for name,candidate in [('warm_old',False),('warm_fused',True),('old0',False),('fused0',True),('fused1',True),('old1',False)]:
        os.environ['SPACEPDHCG_TEST_GTOC12_FUSED_TABLES']='1' if candidate else '0'
        seconds=0.; retained=[]
        for i,q in enumerate(queries):
            table=SimpleNamespace(catalogue=catalogue,epochs=q['epochs'],lambert_evaluations=0)
            start=time.perf_counter()
            value=GpuCollectTable(gpu,table,q['source'],q['target'],q['tofs'],q['end'])
            seconds+=time.perf_counter()-start
            retained.append(value)
            digest=hashlib.sha256(value.read().tobytes()).hexdigest()
            if name=='warm_old': reference.append(digest)
            else: assert digest==reference[i],(name,i,q['source'],q['target'])
        for value in retained:value.close()
        rows.append(dict(name=name,candidate=candidate,seconds=seconds,tables=len(queries)))
        output.write_text(json.dumps(dict(complete=False,rows=rows),indent=2))
output.write_text(json.dumps(dict(complete=True,scope='Sum of actual GpuCollectTable constructor wall time for captured full-campaign grids; excludes validation downloads, hashing, destruction, and other mission work. Retains each pass tables until the end; separate full warmup per mode; every table bitwise identical.',rows=rows,trace_sha256=hashlib.sha256(trace.read_bytes()).hexdigest(),tables=len(queries),cells=sum(q['n']*q['nt'] for q in queries),table_sha256=reference),indent=2))
