from pathlib import Path
import runpy,json,time,functools,collections
from spacepdhcg.gtoc12 import gpu_lambert,gpu_scvx,gpu_retime,gpu_collect_tables,gpu_collection,search,verifier,cooperative
root=Path('build/performance/pipeline-timers-v394');totals={};sizes=collections.Counter()
def timed(name,fn):
 @functools.wraps(fn)
 def call(*args,**kwargs):
  start=time.perf_counter()
  try:return fn(*args,**kwargs)
  finally:
   elapsed=time.perf_counter()-start;row=totals.setdefault(name,dict(calls=0,seconds=0.0,max_seconds=0.0));row['calls']+=1;row['seconds']+=elapsed;row['max_seconds']=max(row['max_seconds'],elapsed)
 return call
grid_requests=[]
retime_init=gpu_retime.GpuRetime.__init__
def initialise_retime(self,*args,**kwargs):
 retime_init(self,*args,**kwargs)
 original_grid=self.create_elements
 def capture_grid(*a):
  import ctypes as ct,hashlib,numpy as np
  n,stages=a[1],a[2];epochs=ct.string_at(a[5],n*8)
  params=np.frombuffer(ct.string_at(a[8],stages*gpu_retime.STAGE.itemsize),dtype=gpu_retime.STAGE)
  for j,p in enumerate(params):
   nt=int(p['tofs']);tofs=ct.string_at(a[6]+8*int(p['tof_offset']),8*nt)
   elements=ct.string_at(a[9]+136*j,136)
   grid_requests.append(dict(key=hashlib.sha256(epochs+tofs+elements).hexdigest(),cells=n*nt,bytes=n*nt*9))
  return original_grid(*a)
 self.create_elements=capture_grid
 for name in ['create','create_elements','destroy','evaluate','evaluate_forward','evaluate_order','set_graph']:
  fn=getattr(self,name,None)
  if fn is not None:setattr(self,name,timed('native_retime.'+name,fn))
gpu_retime.GpuRetime.__init__=initialise_retime
scvx_calls=[]
scvx_native=gpu_scvx.solve_native
def track_scvx(*args,**kwargs):
 import dataclasses
 start=time.perf_counter();result=scvx_native(*args,**kwargs);elapsed=time.perf_counter()-start
 scvx_calls.append(dict(seconds=elapsed,nodes=len(result.node_epochs_mjd),boundary=dataclasses.asdict(result.boundary),settings=dataclasses.asdict(args[1]),status=result.status,diagnostic=result.diagnostic,iterations=result.iterations,accepted=result.accepted_iterations,history=result.history,reports=result.solver_reports))
 return result
gpu_scvx.solve_native=track_scvx
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
 (root/'scvx-calls.json').write_text(json.dumps(scvx_calls,indent=2,default=lambda x:x.tolist()))
 (root/'grid-requests.json').write_text(json.dumps(grid_requests,indent=2))
