from pathlib import Path
import json,struct
from types import SimpleNamespace
import numpy as np
from spacepdhcg.gtoc12.gpu_collection import Query
from spacepdhcg.gtoc12 import constants as C
p=Path('build/performance');root=p/'resident-options-capture-v493'
queries=json.loads((root/'queries.json').read_text());tables=np.load(root/'tables.npz');keys=list(tables.files)
out=p/'resident-options-fixture.bin'
with out.open('wb') as f:
 f.write(struct.pack('<II',len(keys),len(queries)))
 for key in keys:
  a=np.ascontiguousarray(tables[key],dtype=np.float64);f.write(struct.pack('<I',len(a)));f.write(a.tobytes())
 for row in queries:
  s=SimpleNamespace(**row['settings']);first=row['first']
  q=Query(int(first),int(s.hop_inflation_slope is not None),row['mass'],row['epoch'],row['max_span'],s.earth_return_authority_ratio if first else s.hop_authority_ratio,s.hop_inflation,s.hop_inflation_floor,s.hop_inflation_slope or 0.,s.wait_penalty,row['penalty_scale'],C.THRUST_MAX_N,C.DAY_S,C.YEAR_DAYS,C.MINING_RATE_KG_PER_YEAR,C.ISP_S*C.G0_M_S2*1e-3)
  f.write(struct.pack('<I',keys.index(row['key'])));f.write(bytes(q));cost,winner=row['expected']
  f.write(struct.pack('<dI3d',cost,int(winner is not None),*(winner if winner else [0.,0.,0.])))
print(out,len(keys),len(queries),out.stat().st_size)
