from pathlib import Path
import json
root=Path('results/lambda/2026-09-08/gpu-fleet-v374/spacepdhcg-fleet-search-v374/output')
rows=[]
for e in json.loads((root/'ship_04/refinements.json').read_text()):
 r=e['refined'];rows.append(dict(rank=e['rank'],failures=r['failures'],legs=[{k:v for k,v in leg.items() if k in ['source','target','departure_mjd','arrival_mjd','certified','status','failure','reason']} for leg in r['legs']]))
print(json.dumps(rows,indent=2))
Path('build/performance/fleet-failures-v374.json').write_text(json.dumps(rows,indent=2))
