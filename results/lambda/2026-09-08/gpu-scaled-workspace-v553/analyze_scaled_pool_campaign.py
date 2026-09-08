from pathlib import Path
import json
import statistics
import sys

root = Path(sys.argv[1])
report = json.loads((root / 'report.json').read_text())
assert report['complete'] and not report.get('error')
rows, plans, counts = [], None, None
for c in report['campaigns']:
    result = json.loads((root / c['name'] / 'output/run_report.json').read_text())
    best = result['best']
    assert best['accepted'] and best['official']['ok'] and best['independent']['ok']
    calls = json.loads((root / c['name'] / 'calls.json').read_text())
    assert all(call['settings']['qoco_ruiz_iterations'] == (2 if c['candidate'] else 0) for call in calls)
    current = result['ships'][0]['search']['top_candidates']
    actual = {k: v for k, v in result['screening'].items() if k.startswith('completed_')}
    if plans is None:
        plans, counts = current, actual
    elapsed = c.get('process_seconds')
    if elapsed is None:
        elapsed = next(s['seconds'] for s in report['stages'] if s['name'] == c['name'])
    rows.append(dict(name=c['name'], candidate=c['candidate'], process_seconds=elapsed,
                     cli_seconds=result['wall_seconds_total'],
                     score=best['independent']['weighted_score_fixed_bonus_kg'],
                     same_initial_plans=current == plans, same_logical_counts=actual == counts,
                     calls=len(calls), solver_seconds=sum(x['seconds'] for x in calls),
                     converged=sum(x['status'] == 'converged' for x in calls),
                     workspace_creations=sum(x['solver_reports'][-1]['workspace_creations']
                                             for x in calls if x['solver_reports'])))
before, after = [statistics.median(r['process_seconds'] for r in rows if r['candidate'] == mode)
                 for mode in (False, True)]
output = dict(scope='ABBA two runs per mode, identical binary. Zero Ruiz versus two Ruiz passes with objective magnitude preserved; candidate enables compatible scaled workspace reuse; zero-Ruiz baseline uses existing reuse. All runs pass both mission checkers.',
              baseline_process_median=before, candidate_process_median=after,
              less_process_time_percent=100 * (1 - after / before), rows=rows)
(root / 'analysis.json').write_text(json.dumps(output, indent=2))
print(json.dumps(output, indent=2))
