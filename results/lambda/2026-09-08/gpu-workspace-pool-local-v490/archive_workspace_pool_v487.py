import sys
sys.argv=['analysis','/home/ubuntu/spacepdhcg-workspace-pool-v487']
exec("from pathlib import Path\nimport json,sys,statistics\nroot=Path(sys.argv[1]);report=json.loads((root/'report.json').read_text())\nassert report['complete'] and not report.get('error')\nrows=[];reference=None;plans=None\nfor c in report['campaigns']:\n folder=root/c['name'];r=json.loads((folder/'output/run_report.json').read_text());b=r['best']\n assert b['accepted'] and b['official']['ok'] and b['independent']['ok']\n if reference is None:reference=r['screening'];plans=r['ships'][0]['search']['top_candidates']\n assert reference==r['screening'] and plans==r['ships'][0]['search']['top_candidates']\n calls=json.loads((folder/'calls.json').read_text())\n cold=lambda key:sum(max((p[key] or 0 for p in call['solver_reports']),default=0) for call in calls)\n row=dict(name=c['name'],candidate=c['candidate'],seconds=r['wall_seconds_total'],score=b['independent']['weighted_score_fixed_bonus_kg'],native_seconds=sum(call['seconds'] for call in calls),native_calls=len(calls),workspace_creations=cold('workspace_creations'),setup_seconds=cold('setup_seconds'),priming=sum(p['solve_seconds'] is not None for call in calls for p in call['solver_reports']),outer_attempts=sum(call['iterations'] for call in calls),inner_iterations=sum(p['iterations'] for call in calls for p in call['solver_reports']))\n row['process_seconds']=next(s['seconds'] for s in report['stages'] if s['name']==c['name'])\n assert len(calls)==47\n if c['candidate']:assert row['workspace_creations']<47\n rows.append(row)\nbefore=statistics.median(r['seconds'] for r in rows if not r['candidate']);after=statistics.median(r['seconds'] for r in rows if r['candidate'])\nresult=dict(rows=rows,baseline_median=before,candidate_median=after,less_time_percent=100*(1-after/before),scope='ABBA two observations per mode, same binary, pool off/on. Exact initial plans and screening counts; both mission checkers pass. Setup timings are cumulative per call, counted once.')\nbefore_process=statistics.median(r['process_seconds'] for r in rows if not r['candidate'])\nafter_process=statistics.median(r['process_seconds'] for r in rows if r['candidate'])\nresult.update(baseline_process_median=before_process,candidate_process_median=after_process,less_process_time_percent=100*(1-after_process/before_process))\n(root/'analysis.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))\n")
from pathlib import Path
import json,hashlib,tarfile,shutil
root=Path('/home/ubuntu/spacepdhcg-workspace-pool-v487')
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
