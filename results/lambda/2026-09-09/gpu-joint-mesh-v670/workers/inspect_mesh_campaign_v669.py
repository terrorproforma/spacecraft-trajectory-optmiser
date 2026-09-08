from pathlib import Path
import json
root=Path('/home/angus/spacepdhcg-joint-mesh-v662/campaign-v667')
report=json.loads((root/'report.json').read_text())
for run in report['runs']:
 solves=[json.loads(line) for line in (root/run['name']/'native-solves.jsonl').read_text().splitlines()]
 print(run['name'],'process',run['process_seconds'],'native',sum(s['seconds'] for s in solves),'iterations',sum(s.get('iterations',0) for s in solves),'returns',[(s['solve'],s.get('iterations'),s['seconds']) for s in solves if s['arrival_epoch']>=69800])
