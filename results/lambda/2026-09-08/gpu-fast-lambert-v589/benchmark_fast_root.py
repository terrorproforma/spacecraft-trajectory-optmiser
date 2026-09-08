"""Paired identical-request host API timings; includes transfers and sync."""
import json
import os
import sys
import time
from pathlib import Path

sys.meta_path = [f for f in sys.meta_path if f.__class__.__module__ != '_editable_skbc_spacepdhcg']
import numpy as np
from spacepdhcg.gtoc12 import constants as C
from spacepdhcg.gtoc12.gpu_lambert import GpuLambert, HOP_RESULT

rows=[]
for count in (31, 1024, 16384, 16385, 65536):
 rng=np.random.default_rng(585)
 r1,r2=(rng.normal(size=(count,3)) for _ in range(2))
 for positions in (r1,r2):
  positions *= (C.AU_KM*rng.uniform(1,3,count)/np.linalg.norm(positions,axis=1))[:,None]
 days=rng.uniform(150,800,count)
 with GpuLambert(count) as gpu:
  gpu.screen_hops(r1,[0,0,0],r2,[0,0,0],64328,days)
  output=np.empty(count,dtype=HOP_RESULT)
  def evaluate():
   gpu._check(gpu.evaluate_hops(gpu.handle,gpu.hops.ctypes.data,count,output.ctypes.data,count))
  baseline=None
  for name,enabled in [('A0',0),('B0',1),('B1',1),('A1',0)]:
   os.environ['SPACEPDHCG_TEST_GTOC12_FAST_LAMBERT_ROOT']=str(enabled)
   for _ in range(5):evaluate()
   times=[]
   for _ in range(30):
    start=time.perf_counter();evaluate();times.append(time.perf_counter()-start)
   if baseline is None:baseline=output.copy()
   for field in ('v1','v2','dep','arr'):
    np.testing.assert_allclose(output[field],baseline[field],atol=1e-8,rtol=1e-9)
   np.testing.assert_array_equal(output['feasible'],baseline['feasible'])
   rows.append(dict(count=count,name=name,candidate=bool(enabled),seconds=times,median_seconds=float(np.median(times)),hops_per_second=count/float(np.median(times)),feasible=int(output['feasible'].sum())))
Path(sys.argv[1]).write_text(json.dumps(dict(complete=True,timing_scope='native host API including input/output transfer and synchronization; excludes request construction',rows=rows),indent=2))
