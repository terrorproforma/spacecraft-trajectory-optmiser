from pathlib import Path
import json,sys,statistics
root=Path(sys.argv[1]);report=json.loads((root/'report.json').read_text())
assert report['complete'] and not report.get('error')
rows=[];reference=None;plans=None
for c in report['campaigns']:
 folder=root/c['name'];r=json.loads((folder/'output/run_report.json').read_text());b=r['best']
 assert b['accepted'] and b['official']['ok'] and b['independent']['ok']
 if reference is None:reference=r['screening'];plans=r['ships'][0]['search']['top_candidates']
 assert reference==r['screening'] and plans==r['ships'][0]['search']['top_candidates']
 calls=json.loads((folder/'calls.json').read_text())
 cold=lambda key:sum(max((p[key] or 0 for p in call['solver_reports']),default=0) for call in calls)
 row=dict(name=c['name'],candidate=c['candidate'],seconds=r['wall_seconds_total'],score=b['independent']['weighted_score_fixed_bonus_kg'],native_seconds=sum(call['seconds'] for call in calls),native_calls=len(calls),workspace_creations=cold('workspace_creations'),setup_seconds=cold('setup_seconds'),priming=sum(p['solve_seconds'] is not None for call in calls for p in call['solver_reports']),outer_attempts=sum(call['iterations'] for call in calls),inner_iterations=sum(p['iterations'] for call in calls for p in call['solver_reports']))
 row['process_seconds']=next(s['seconds'] for s in report['stages'] if s['name']==c['name'])
 assert len(calls)==47
 if c['candidate']:assert row['workspace_creations']<47
 rows.append(row)
before=statistics.median(r['seconds'] for r in rows if not r['candidate']);after=statistics.median(r['seconds'] for r in rows if r['candidate'])
result=dict(rows=rows,baseline_median=before,candidate_median=after,less_time_percent=100*(1-after/before),scope='ABBA two observations per mode, same binary, pool off/on. Exact initial plans and screening counts; both mission checkers pass. Setup timings are cumulative per call, counted once.')
before_process=statistics.median(r['process_seconds'] for r in rows if not r['candidate'])
after_process=statistics.median(r['process_seconds'] for r in rows if r['candidate'])
result.update(baseline_process_median=before_process,candidate_process_median=after_process,less_process_time_percent=100*(1-after_process/before_process))
(root/'analysis.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
