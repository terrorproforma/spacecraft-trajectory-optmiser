from pathlib import Path
import functools,time,inspect,runpy,json,hashlib,collections
import numpy as np
from spacepdhcg.gtoc12 import search,collectdp,gpu_collect_dp,gpu_collect_tables,gpu_lambert,gpu_collection,gpu_scvx

root=Path('build/performance/search-profile-v414');totals={};stack=[];grids=[]
def timed(name,fn):
 @functools.wraps(fn)
 def call(*args,**kwargs):
  frame=[time.perf_counter(),0.0];stack.append(frame)
  try:return fn(*args,**kwargs)
  finally:
   elapsed=time.perf_counter()-frame[0];stack.pop()
   if stack:stack[-1][1]+=elapsed
   r=totals.setdefault(name,dict(calls=0,inclusive_seconds=0.0,exclusive_seconds=0.0,max_seconds=0.0));r['calls']+=1;r['inclusive_seconds']+=elapsed;r['exclusive_seconds']+=elapsed-frame[1];r['max_seconds']=max(r['max_seconds'],elapsed)
 return call

native_init=gpu_collect_tables.GpuCollectTable.__init__
def traced_init(self,gpu,table,source,target,tofs,end):
 from spacepdhcg.gtoc12 import constants as C
 elements=gpu_lambert.HopElements(gpu_lambert.body_elements(table.catalogue,source),gpu_lambert.body_elements(table.catalogue,target),C.MU_SUN_KM3_S2,0.,C.MAX_VINF_EARTH_KM_S if target==0 else 0.)
 axes=np.ascontiguousarray(table.epochs,dtype=np.float64).tobytes()+np.ascontiguousarray(tofs,dtype=np.float64).tobytes()
 grids.append(dict(key=hashlib.sha256(bytes(elements)+axes).hexdigest(),n=len(table.epochs),nt=len(tofs),source=int(source),target=int(target),end=float(end)))
 return native_init(self,gpu,table,source,target,tofs,end)
gpu_collect_tables.GpuCollectTable.__init__=traced_init
for cls in [search.RouteSearch,collectdp.CollectPairTable,gpu_collect_dp.GpuCollectDP,gpu_collect_tables.GpuCollectTable,gpu_collection.GpuCollection]:
 for name,value in list(vars(cls).items()):
  if name.startswith('__') and name!='__init__':continue
  if inspect.isfunction(value):setattr(cls,name,timed(cls.__name__+'.'+name,value))
  elif isinstance(value,staticmethod):setattr(cls,name,staticmethod(timed(cls.__name__+'.'+name,value.__func__)))
for name in ['screen_hops','screen_elements','_prepare']:
 if hasattr(gpu_lambert.GpuLambert,name):setattr(gpu_lambert.GpuLambert,name,timed('GpuLambert.'+name,getattr(gpu_lambert.GpuLambert,name)))
gpu_scvx.solve_native=timed('SCvx.solve_native',gpu_scvx.solve_native)
start=time.perf_counter()
try:runpy.run_module('spacepdhcg',run_name='__main__')
finally:
 elapsed=time.perf_counter()-start
 (root/'timers.json').write_text(json.dumps(dict(seconds=elapsed,scope='Direct nested wall timers. Exclusive time subtracts only instrumented children; native time includes synchronization. Grid trace hashes immutable elements and axes, excluding the subsequent end-date mask.',timers=dict(sorted(totals.items(),key=lambda kv:-kv[1]['exclusive_seconds']))),indent=2))
 (root/'collect-grids.json').write_text(json.dumps(grids,indent=2))
