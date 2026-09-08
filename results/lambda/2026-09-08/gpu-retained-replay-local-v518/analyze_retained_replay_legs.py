from pathlib import Path
import collections
import json
import sys

root = Path(sys.argv[1])
report = json.loads((root / 'report.json').read_text())
assert report['complete'] and not report.get('error')
groups = {name: json.loads((root / name / 'results.json').read_text())
          for name in ['baseline', 'candidate']}
result = {}
for name, rows in groups.items():
    assert [row['index'] for row in rows] == list(range(225))
    assert all(row.get('certified', False) for row in rows if row['status'] == 'converged')
    result[name] = dict(
        seconds=sum(row['seconds'] for row in rows),
        converged=sum(row['status'] == 'converged' for row in rows),
        priming=sum(row['priming'] for row in rows),
        workspaces=sum(row['workspace_creations'] for row in rows),
        statuses=dict(collections.Counter(row['status'] for row in rows)),
    )
lost = []
differences = []
for old, new in zip(groups['baseline'], groups['candidate'], strict=True):
    if old['status'] == 'converged':
        if new['status'] != 'converged':
            lost.append(old['index'])
        else:
            differences.append(dict(index=old['index'], delta_kg=abs(
                old['certificate']['final_mass_kg'] - new['certificate']['final_mass_kg'])))
result.update(
    scope='One full pass per mode with the same binary; cumulative solver time excludes certification. All trajectory acceptance tolerances are unchanged.',
    lost_baseline=lost,
    max_certified_mass_delta=max(differences, key=lambda x: x['delta_kg']),
    less_time_percent=100 * (1 - result['candidate']['seconds'] / result['baseline']['seconds']),
    largest_timing_changes=sorted([
        dict(index=a['index'], baseline_seconds=a['seconds'], candidate_seconds=b['seconds'],
             difference=b['seconds'] - a['seconds'], baseline_iterations=a['iterations'],
             candidate_iterations=b['iterations'])
        for a, b in zip(groups['baseline'], groups['candidate'], strict=True)
    ], key=lambda row: -abs(row['difference']))[:10],
)
result['diagnostic_subsets'] = {}
for name, condition in [
    ('reused_in_both', lambda a, b: a['workspace_creations'] == b['workspace_creations'] == 0),
    ('fresh_in_both', lambda a, b: a['workspace_creations'] == b['workspace_creations'] == 1),
    ('converged_in_both', lambda a, b: a['status'] == b['status'] == 'converged'),
]:
    pairs = [(a, b) for a, b in zip(groups['baseline'], groups['candidate'], strict=True)
             if condition(a, b)]
    before, after = [sum(pair[i]['seconds'] for pair in pairs) for i in (0, 1)]
    result['diagnostic_subsets'][name] = dict(
        cases=len(pairs), baseline_seconds=before, candidate_seconds=after,
        less_time_percent=100 * (1 - after / before))
(root / 'analysis.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
