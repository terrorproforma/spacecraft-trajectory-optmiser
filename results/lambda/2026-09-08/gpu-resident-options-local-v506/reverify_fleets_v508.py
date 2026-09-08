from pathlib import Path
import dataclasses,hashlib,json,os,sys,time
sys.meta_path=[f for f in sys.meta_path if f.__class__.__module__!='_editable_skbc_spacepdhcg']
from spacepdhcg.gtoc12.data import load_catalogue,load_bonus_table
from spacepdhcg.gtoc12.verifier import Gtoc12Verifier
root=Path('build/performance/verifier-fleets-v508');root.mkdir(exist_ok=False)
sources=['src/spacepdhcg/gtoc12/verifier.py','src/spacepdhcg/gtoc12/solution.py','src/spacepdhcg/gtoc12/ephemeris.py']
r=dict(pid=os.getpid(),complete=False,source_sha256={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sources},cases=[])
def save():(root/'report.json').write_text(json.dumps(r,indent=2))
save()
try:
 verifier=Gtoc12Verifier(load_catalogue(),bonus=load_bonus_table())
 for name,path in [('incumbent',Path('results/lambda/2026-09-06/fleet_master_v11/fleet/Result.txt')),('gpu_large',Path('results/lambda/2026-09-08/gpu-fleet-recovery-v381/fleet/Result.txt')),('gpu_wider',Path('build/performance/resident-options-fleet-v499/output/fleet/Result.txt'))]:
  r['stage']=name;save();start=time.perf_counter();report=verifier.verify_file(path)
  row=dict(name=name,path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),seconds=time.perf_counter()-start,summary=report.summary())
  r['cases'].append(row);save();assert report.ok,row
  if name=='incumbent':assert abs(report.weighted_score_fixed_bonus_kg-12805.194102488575)<1e-6 and report.ship_count==23
 r['complete']=True
except Exception as e:r['error']=repr(e)
save();print(json.dumps(r,indent=2))
