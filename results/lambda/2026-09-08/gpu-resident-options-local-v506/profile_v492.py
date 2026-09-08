from pathlib import Path
import functools,time,inspect,runpy,json,hashlib,collections
import numpy as np
from spacepdhcg.gtoc12 import search,collectdp,gpu_collect_dp,gpu_collect_tables,gpu_lambert,gpu_collection,gpu_scvx,gpu_beam,gpu_retime,retiming,verifier,low_thrust

root=Path('build/performance/pipeline-profile-v492');totals={};stack=[];grids=[]
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

for cls in [search.RouteSearch,collectdp.CollectPairTable,gpu_collect_dp.GpuCollectDP,gpu_collect_tables.GpuCollectTable,gpu_collection.GpuCollection,gpu_retime.GpuRetime,retiming.Retimer,verifier.Gtoc12Verifier]:
 for name,value in list(vars(cls).items()):
  if name.startswith('__') and name!='__init__':continue
  if inspect.isfunction(value):setattr(cls,name,timed(cls.__name__+'.'+name,value))
  elif isinstance(value,staticmethod):setattr(cls,name,staticmethod(timed(cls.__name__+'.'+name,value.__func__)))
for name in ['screen_hops','paired_hops','paired_options','screen_elements','_prepare']:
 if hasattr(gpu_lambert.GpuLambert,name):setattr(gpu_lambert.GpuLambert,name,timed('GpuLambert.'+name,getattr(gpu_lambert.GpuLambert,name)))
gpu_beam.earth_beam=timed('GpuBeam.earth_beam',gpu_beam.earth_beam)
gpu_scvx.solve_native=timed('SCvx.solve_native',gpu_scvx.solve_native)
low_thrust.certify_leg=timed('Verification.certify_leg',low_thrust.certify_leg)
start=time.perf_counter()
try:runpy.run_module('spacepdhcg',run_name='__main__')
finally:
 elapsed=time.perf_counter()-start
 (root/'timers.json').write_text(json.dumps(dict(seconds=elapsed,scope='Direct nested wall timers. Exclusive time subtracts only instrumented children; native time includes synchronization. Includes independent verification timers; no extra CUDA synchronization is inserted.',timers=dict(sorted(totals.items(),key=lambda kv:-kv[1]['exclusive_seconds']))),indent=2))
 (root/'collect-grids.json').write_text(json.dumps(grids,indent=2))
