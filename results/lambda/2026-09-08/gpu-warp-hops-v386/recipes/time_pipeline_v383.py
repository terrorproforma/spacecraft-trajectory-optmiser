from pathlib import Path
import runpy,json,time,functools,collections
from spacepdhcg.gtoc12 import gpu_lambert,gpu_scvx,gpu_retime,gpu_collect_tables,gpu_collection,search,verifier,cooperative
root=Path('build/performance/pipeline-timers-v383');totals={};sizes=collections.Counter()
def timed(name,fn):
 @functools.wraps(fn)
 def call(*args,**kwargs):
  start=time.perf_counter()
  try:return fn(*args,**kwargs)
  finally:
   elapsed=time.perf_counter()-start;row=totals.setdefault(name,dict(calls=0,seconds=0.0,max_seconds=0.0));row['calls']+=1;row['seconds']+=elapsed;row['max_seconds']=max(row['max_seconds'],elapsed)
 return call
initial=gpu_lambert.GpuLambert.__init__
def initialise(self,*args,**kwargs):
 initial(self,*args,**kwargs);native=self.evaluate_hops
 def evaluate(*a):
  sizes[int(a[2])]+=1
  return native(*a)
 self.evaluate_hops=timed('native_hop_screening_including_sync',evaluate)
gpu_lambert.GpuLambert.__init__=initialise
for obj,name in [(gpu_lambert.GpuLambert,'screen_hops'),(gpu_scvx,'solve_native'),(gpu_retime.GpuRetime,'solve'),(gpu_collect_tables.GpuCollectTable,'__init__'),(gpu_collect_tables.GpuCollectTable,'minimum_propellant'),(gpu_collection.GpuCollection,'select'),(search.RouteSearch,'run'),(search.RouteSearch,'_return_options'),(search.RouteSearch,'_collect_hop_options'),(verifier.Gtoc12Verifier,'verify_file'),(cooperative,'solve_fleet_master')]:
 label=getattr(obj,'__name__',str(obj))+'.'+name;setattr(obj,name,timed(label,getattr(obj,name)))
start=time.perf_counter()
try:runpy.run_module('spacepdhcg',run_name='__main__')
finally:
 (root/'timers.json').write_text(json.dumps(dict(scope='Inclusive direct wall timers; nested rows must not be summed. Native screening includes CUDA synchronization.',wall_seconds=time.perf_counter()-start,timers=totals,screening_batch_sizes=dict(sorted(sizes.items()))),indent=2))
