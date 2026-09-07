from pathlib import Path
import json
from audit_helper import problem,audit
qp=next(Path('/home/ubuntu/spacepdhcg-capture-v202/qp-0').glob('*000000.txt'));d=problem(qp)
for version in ['frozen-v203','regularization-v204']:
 root=Path('/home/ubuntu/spacepdhcg-'+version)
 assert json.loads((root/'report.json').read_text())['complete']
 audits={}
 for p in root.glob('*.log'):
  if p.name=='runner.log':continue
  rows=[json.loads(line[10:]) for line in p.read_text().splitlines() if line.startswith('QP_REPLAY ')]
  audits[p.stem]=[audit(d,row) for row in rows]
 (root/'audit.json').write_text(json.dumps(audits,indent=2))
 print(version,{k:dict(passed=sum(r['qualified'] for r in v),total=len(v),maxgap=max((r['gap'] for r in v),default=0)) for k,v in audits.items()})
