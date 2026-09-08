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
    priming=sum(p['solve_seconds'] is not None for call in calls for p in call['solver_reports'])
    row=dict(name=c['name'],candidate=c['candidate'],seconds=r['wall_seconds_total'],score=b['independent']['weighted_score_fixed_bonus_kg'],native_seconds=sum(call['seconds'] for call in calls),native_calls=len(calls),priming=priming,outer_attempts=sum(call['iterations'] for call in calls),inner_iterations=sum(p['iterations'] for call in calls for p in call['solver_reports']))
    assert len(calls)==47 and priming==(94 if c['candidate'] else 141)
    rows.append(row)
before=statistics.median(r['seconds'] for r in rows if not r['candidate']);after=statistics.median(r['seconds'] for r in rows if r['candidate'])
result=dict(rows=rows,baseline_median=before,candidate_median=after,less_time_percent=100*(1-after/before),scope='ABBA, two observations per mode, same binary with readiness switch. Full initial plans and screening counts match; both mission checkers pass. Nonlinear iteration counts vary and complete-runtime gains are not universal.')
(root/'analysis.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
