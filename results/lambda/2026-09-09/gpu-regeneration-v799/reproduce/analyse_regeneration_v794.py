from pathlib import Path
import json,sys,math
home=Path.home();root=home/'spacepdhcg-regeneration-v792'
sys.meta_path=[f for f in sys.meta_path if f.__class__.__module__!='_editable_skbc_spacepdhcg'];sys.path.insert(0,str(home/'spacepdhcg-grid-v788/repo/src'))
from spacepdhcg.gtoc12.solution import Solution
from spacepdhcg.gtoc12.search import RoutePlan
from spacepdhcg.gtoc12.data import load_bonus_table
from spacepdhcg.gtoc12 import constants as C
fleet=Solution.read(root/'fleet.txt');bonus=load_bonus_table()
allrows=[]
for ship in fleet.ships:
    raw=sum(max(0,e.after.mass-e.before.mass) for e in ship.asteroid_visits())
    score=sum(bonus.coefficient[e.event_id-1]*max(0,e.after.mass-e.before.mass) for e in ship.asteroid_visits())
    plans=[RoutePlan.from_summary(x) for x in json.loads((root/f'ship-{ship.ship_id:02d}/plans.json').read_text())]
    rows=[dict(ship=ship.ship_id,rank=i,raw=sum(p.collected_mass.values()),score=sum(bonus.coefficient[a-1]*m for a,m in p.collected_mass.items())) for i,p in enumerate(plans)]
    for r in rows:r.update(raw_gain=r['raw']-raw,score_gain=r['score']-score)
    # Analytical filtering only: every point with a raw-mass gain is retained.
    positive=[r for r in rows if r['raw_gain']>0]
    pareto=[r for r in positive if not any(t['raw']>=r['raw'] and t['score']>=r['score'] and (t['raw']>r['raw'] or t['score']>r['score']) for t in positive)]
    allrows.extend(rows)
    print(json.dumps(dict(ship=ship.ship_id,current_raw=raw,positive=len(positive),pareto=pareto)))
print(json.dumps(dict(total_candidates=len(allrows),expansions=sum(r['expansions'] for r in json.loads((root/'report.json').read_text())['routes']))))
