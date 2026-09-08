from pathlib import Path
import json,sys
home=Path.home();sys.meta_path=[f for f in sys.meta_path if f.__class__.__module__!='_editable_skbc_spacepdhcg'];sys.path.insert(0,str(home/'spacepdhcg-fleet-v742/repo/src'))
from spacepdhcg.gtoc12.cooperative import FleetColumn,fleet_feasible
from spacepdhcg.gtoc12.gpu_fleet import CudaFleetWorkspace
p=json.loads((home/'spacepdhcg-fleet-pool-v716/pool.json').read_text())
cols=[FleetColumn(r['identifier'],r['ship_id'],r['label'],*({int(k):v for k,v in r[key].items()} for key in ('deploys','collects','foreign','mass')),True) for r in p['rows']]
warm=tuple(c for c in cols if c.identifier in p['warm']);weights={int(k):v for k,v in p['weights'].items()}
expected=json.loads((home/'spacepdhcg-fleet-refine-v729/input/selected.json').read_text())
for repeat in range(2):
    with CudaFleetWorkspace(cols,weights=weights) as workspace:
        for cap,rounds,ships,incumbent in [(0,16,100,warm),(0,0,0,None),(200000,16,100,warm)]:
            r=workspace.solve(incumbent=incumbent,node_cap=cap,exchange_rounds=rounds,max_ships=ships)
            assert not fleet_feasible(r.selected)
            if ships:
                assert [c.identifier for c in r.selected]==expected
                assert r.nodes==(35145 if cap else 0) and abs(r.objective-12842.970672270907)<1e-8
                assert r.exchange_proposals==179205 and r.exchange_moves==2
            else:assert not r.selected and r.objective==0
            print(json.dumps(dict(repeat=repeat,cap=cap,rounds=rounds,nodes=r.nodes,objective=r.objective,native_seconds=r.native_seconds)))
