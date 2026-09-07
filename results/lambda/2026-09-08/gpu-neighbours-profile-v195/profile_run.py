import sys
sys.path.pop(0)
import os,json,cProfile,pstats,fcntl
from pathlib import Path
import numpy as np
root=Path('/home/ubuntu/spacepdhcg-neighbours-v194')
lock=open('/home/ubuntu/.spacepdhcg-gpu.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
sys.meta_path=[f for f in sys.meta_path if f.__class__.__module__!='_editable_skbc_spacepdhcg']
sys.path.insert(0,str(root/'repo/src'))
os.environ.update(SPACEPDHCG_GTOC12_CUDA_LIBRARY=str(root/'libscreening.so'),SPACEPDHCG_GTOC12_DATA=str(root/'data'),OPENBLAS_NUM_THREADS='1')
from spacepdhcg.gtoc12 import constants as C
from spacepdhcg.gtoc12.data import load_catalogue
from spacepdhcg.gtoc12.lambert import using_lambert_backend
from spacepdhcg.gtoc12.search import RouteSearch,SearchSettings
catalogue=load_catalogue();ids=np.array(json.loads((root/'routes.json').read_text())['asteroid_ids'])
settings=SearchSettings(beam_width=8,max_deploys=4,neighbours=24,launch_epochs=tuple(float(x) for x in C.MISSION_START_MJD+np.arange(0,731,90)),earth_leg_tofs=(450.,600.,750.,900.),hop_tofs=(90.,180.,270.,360.),harvest_substitution=False)
with using_lambert_backend('cuda'):
 RouteSearch(catalogue,ids,settings).run()
 profile=cProfile.Profile();profile.enable();result=RouteSearch(catalogue,ids,settings).run();profile.disable()
stats=pstats.Stats(profile)
rows=[dict(file=key[0],line=key[1],function=key[2],calls=value[1],self_seconds=value[2],cumulative_seconds=value[3]) for key,value in stats.stats.items()]
rows.sort(key=lambda r:-r['cumulative_seconds'])
(Path('/home/ubuntu/spacepdhcg-neighbours-profile-v195')/'profile.json').write_text(json.dumps(dict(total_seconds=stats.total_tt,branches=result.lambert_evaluations,candidates=len(result.candidates),rows=rows[:100]),indent=2))
print(json.dumps(dict(total_seconds=stats.total_tt,branches=result.lambert_evaluations,candidates=len(result.candidates),rows=rows[:16]),indent=2))
