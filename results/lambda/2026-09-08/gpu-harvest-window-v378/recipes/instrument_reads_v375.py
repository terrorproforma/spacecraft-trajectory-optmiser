import atexit,json,traceback,time,runpy
from pathlib import Path
from spacepdhcg.gtoc12.gpu_collect_tables import GpuCollectTable
rows={};original=GpuCollectTable.read
def observed(self):
 frames=traceback.extract_stack()[:-1]
 key=' > '.join(f'{Path(f.filename).name}:{f.name}:{f.lineno}' for f in frames[-6:])
 start=time.perf_counter();result=original(self)
 row=rows.setdefault(key,dict(calls=0,bytes=0,seconds=0.0))
 row['calls']+=1;row['bytes']+=result.nbytes;row['seconds']+=time.perf_counter()-start
 return result
GpuCollectTable.read=observed
@atexit.register
def save():Path('build/performance/read-profile-v375/reads.json').write_text(json.dumps(rows,indent=2))
runpy.run_module('spacepdhcg',run_name='__main__')
