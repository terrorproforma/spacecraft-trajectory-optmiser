import sys
sys.argv=['analysis','/home/ubuntu/spacepdhcg-workspace-pool-v484']
exec("from pathlib import Path\nimport json,sys\nroot=Path(sys.argv[1]);r=json.loads((root/'report.json').read_text())\nassert r['complete'] and not r.get('error')\nrows=[]\nfor c in r['campaigns']:\n calls=json.loads((root/c['name']/'calls.json').read_text())\n creations=sum(max(p['workspace_creations'] for p in call['solver_reports']) for call in calls)\n assert len(calls)==47 and creations==47\n rows.append(dict(label=c['name'],pool_enabled=False,seconds=c['seconds'],score=c['score'],workspace_creations=creations,native_seconds=sum(call['seconds'] for call in calls)))\nresult=dict(scope='Baseline repeats only: runner changed the early-graph flag to zero in both branches and never enabled workspace pooling. Candidate labels in raw reports are incorrect; these runs do not measure pool performance. Test matrix still explicitly exercises pool on/off.',rows=rows)\n(root/'analysis.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))\n")
from pathlib import Path
import json,hashlib,tarfile,shutil
root=Path('/home/ubuntu/spacepdhcg-workspace-pool-v484')
r=json.loads((root/'report.json').read_text());assert r['complete'] and not r.get('error')
for name,sha in r['source_sha256'].items():
 p=root/'repo'/name;assert hashlib.sha256(p.read_bytes()).hexdigest()==sha
 target=root/'source-overlay'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
for c in r['campaigns']:
 f=json.loads((root/c['name']/'output/run_report.json').read_text());assert f['best']['accepted'] and f['best']['official']['ok'] and f['best']['independent']['ok']
paths=[p for p in root.rglob('*') if p.is_file() and p.relative_to(root).parts[0] not in ['repo','core-build','compact-options-probe'] and p.name not in ['files-sha256.json','workspace-pool-probe']]
manifest={p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}
(root/'files-sha256.json').write_text(json.dumps(manifest,indent=2))
archive=root.with_suffix('.tar.gz')
with tarfile.open(archive,'w:gz') as t:
 for name in list(manifest)+['files-sha256.json']:t.add(root/name,arcname=name,recursive=False)
print(json.dumps(dict(path=str(archive),sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),bytes=archive.stat().st_size,files=len(manifest))))
