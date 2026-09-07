import sys,runpy,time,json,threading,functools
from pathlib import Path
sys.meta_path=[f for f in sys.meta_path if f.__class__.__module__!='_editable_skbc_spacepdhcg']
from spacepdhcg.gtoc12 import collectdp,gpu_collect_dp
stats={};local=threading.local()
def instrument(owner,name,label):
 original=getattr(owner,name)
 @functools.wraps(original)
 def wrapped(*args,**kwargs):
  if not hasattr(local,'stack'):local.stack=[]
  frame=[time.perf_counter(),0.0];local.stack.append(frame)
  try:return original(*args,**kwargs)
  finally:
   elapsed=time.perf_counter()-frame[0];local.stack.pop()
   if local.stack:local.stack[-1][1]+=elapsed
   row=stats.setdefault(label,dict(calls=0,inclusive_seconds=0.,exclusive_seconds=0.))
   row['calls']+=1;row['inclusive_seconds']+=elapsed;row['exclusive_seconds']+=elapsed-frame[1]
 setattr(owner,name,wrapped)
 # Preserve instrumentation for aliases imported during module initialization.
 for module in list(sys.modules.values()):
  if module is not None and getattr(module,'__name__','').startswith('spacepdhcg.'):
   for key,value in list(vars(module).items()):
    if value is original:setattr(module,key,wrapped)
instrument(collectdp,'plan_collect_tour','plan_collect_tour')
instrument(collectdp,'_solve_collect_dp','solve_collect_dp')
instrument(gpu_collect_dp.GpuCollectDP,'__init__','workspace_create')
instrument(gpu_collect_dp.GpuCollectDP,'solve','workspace_solve_and_format')
for name in ['hop','earth_return','return_override','pair_geometry','phase_deg','phase_penalty']:
 instrument(collectdp.CollectPairTable,name,'table_'+name)
started=time.perf_counter()
try:runpy.run_module('spacepdhcg',run_name='__main__')
finally:Path('/home/ubuntu/spacepdhcg-collect-profile-v288/timing.json').write_text(json.dumps(dict(total_seconds=time.perf_counter()-started,spans=stats,scope='Direct nested wall timers. Exclusive spans subtract only instrumented child calls; include native execution and synchronization. Instrumented run is not a speed benchmark.'),indent=2))
