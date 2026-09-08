from pathlib import Path
import json,sys
home=Path.home();sys.meta_path=[f for f in sys.meta_path if f.__class__.__module__!='_editable_skbc_spacepdhcg'];sys.path.insert(0,str(home/'spacepdhcg-fleet-v727/repo/src'))
from spacepdhcg.gtoc12.cooperative import FleetColumn,fleet_feasible
from spacepdhcg.gtoc12.gpu_fleet import solve_fleet_cuda
p=json.loads((home/'spacepdhcg-fleet-pool-v716/pool.json').read_text());cols=[]
for r in p['rows']:cols.append(FleetColumn(r['identifier'],r['ship_id'],r['label'],*({int(k):v for k,v in r[key].items()} for key in ('deploys','collects','foreign','mass')),True))
warm=tuple(c for c in cols if c.identifier in p['warm']);weights={int(k):v for k,v in p['weights'].items()}
r=solve_fleet_cuda(cols,weights=weights,incumbent=warm,node_cap=0)
assert not fleet_feasible(r.selected) and [c.identifier for c in r.selected]==json.loads((home/'spacepdhcg-fleet-refine-v729/input/selected.json').read_text())
assert r.nodes==0 and abs(r.objective-12842.970672270907)<1e-8
print(json.dumps(dict(nodes=r.nodes,objective=r.objective,native_seconds=r.native_seconds)))
