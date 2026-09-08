from pathlib import Path
import json,sys
root=Path(sys.argv[1]);r=json.loads((root/'report.json').read_text())
assert r['complete'] and not r.get('error')
rows=[]
for c in r['campaigns']:
 calls=json.loads((root/c['name']/'calls.json').read_text())
 creations=sum(max(p['workspace_creations'] for p in call['solver_reports']) for call in calls)
 assert len(calls)==47 and creations==47
 rows.append(dict(label=c['name'],pool_enabled=False,seconds=c['seconds'],score=c['score'],workspace_creations=creations,native_seconds=sum(call['seconds'] for call in calls)))
result=dict(scope='Baseline repeats only: runner changed the early-graph flag to zero in both branches and never enabled workspace pooling. Candidate labels in raw reports are incorrect; these runs do not measure pool performance. Test matrix still explicitly exercises pool on/off.',rows=rows)
(root/'analysis.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
