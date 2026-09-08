from pathlib import Path
import json
import sys

root = Path(sys.argv[1])
report = json.loads((root / 'report.json').read_text())
assert report['complete'] and not report.get('error')
baseline = json.loads((root / 'origin0-ruiz0/results.json').read_text())
output = []
for origin, ruiz in [(0, 0), (1, 0), (0, 2), (1, 2), (0, 5), (1, 5)]:
    rows = json.loads((root / f'origin{origin}-ruiz{ruiz}/results.json').read_text())
    assert [r['index'] for r in rows] == [r['index'] for r in baseline]
    lost, deltas = [], []
    for old, new in zip(baseline, rows, strict=True):
        assert all(old['settings'][k] == new['settings'][k] for k in old['settings']
                   if k != 'qoco_ruiz_iterations')
        if new['status'] == 'converged':
            assert new['certified']
        if old.get('certified'):
            if not new.get('certified'):
                lost.append(old['index'])
            else:
                deltas.append(abs(old['certificate']['final_mass_kg'] - new['certificate']['final_mass_kg']))
    output.append(dict(origin=origin, ruiz=ruiz, seconds=sum(r['seconds'] for r in rows),
                       certified=sum(r.get('certified', False) for r in rows), lost_baseline=lost,
                       max_mass_difference_kg=max(deltas),
                       inner_iterations=sum(p['iterations'] for r in rows for p in r['reports'])))
result = dict(scope='Diagnostic eight-case selection, not complete-workload throughput. Every mode uses a 30-second per-leg pilot budget, fresh workspaces and unchanged accuracy limits.', rows=output)
(root / 'analysis.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
