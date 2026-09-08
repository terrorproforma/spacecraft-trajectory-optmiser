"""Verify final analytic evidence and index exact bytes; no GPU calls."""
from pathlib import Path
import hashlib
import json
import tarfile

root = Path(__file__).resolve().parent
live = root.parents[2]
digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
manifest = json.loads((root / 'manifest.json').read_text())
assert manifest['complete']
for name in manifest['owned_paths']:
    assert digest(live / name) == manifest['source_sha256'][name]
with tarfile.open(root / 'source.tar.gz', 'r:gz') as archive:
    for name in manifest['owned_paths']:
        assert hashlib.sha256(archive.extractfile(name).read()).hexdigest() == manifest['source_sha256'][name]
validation = json.loads((root / 'json-validation.json').read_text())
assert validation['complete'] and len(validation['cases']) == 10
for case in validation['cases']:
    for line in (root / (case['name'] + '-validate-no-gpu.log')).read_text().splitlines():
        json.loads(line.split(' ', 1)[1])
complete = json.loads((root / 'tiny-wait/report.json').read_text())
assert complete['complete'] and complete['qualified_calls'] == 6 and len(complete['cases']) == 6
assert complete['executable_sha256'] == manifest['executable_sha256']
assert complete['core_sha256'] == manifest['immutable_core_sha256']
assert complete['manifest_sha256'] == digest(root / 'manifest.json')
totals = {'completed_executable_invocations': 6, 'native_solve_api_calls': 0,
          'actual_requested_solve_iterations': 0, 'bootstrap_iterations': 0,
          'independently_qualified_final_vectors': 6, 'executable_wall_seconds_sum': 0.0}
for case in complete['cases']:
    records = {}
    for line in (root / 'tiny-wait' / (case['name'] + '.log')).read_text().splitlines():
        prefix, payload = line.split(' ', 1)
        assert prefix not in records
        records[prefix] = json.loads(payload)
    meta = records['PERSISTENT_REPLAY_META']
    final = records['PERSISTENT_REPLAY']
    assert meta['input_sha256'] == case['snapshot_sha256']
    assert meta['source_sha256'] == manifest['compiled_snapshot_source_sha256']
    assert final['qualified_original'] and final['termination'] == 1
    assert case['independent_qualified_count'] == 1
    if case['point_sha256']:
        assert meta['initial_point_sha256'] == case['point_sha256']
        pre = records['PERSISTENT_REPLAY_PRESTEP']
        assert pre['termination'] == 0 and pre['seeded_iterations'] == 0
        assert pre['native_seed_verified_unchanged']
    totals['native_solve_api_calls'] += 1 + len(case['bootstrap'])
    totals['actual_requested_solve_iterations'] += final['iterations']
    totals['bootstrap_iterations'] += sum(record['iterations'] for record in case['bootstrap'])
    totals['executable_wall_seconds_sum'] += case['seconds']
initial = json.loads((root / 'initial-attempt-vector-audit.json').read_text())
busy = json.loads((root / 'busy-v606c/report.json').read_text())
assert not busy['complete'] and not busy['cases'] and 'BlockingIOError' in busy['failure']
combined = {key: totals[key] + initial[key] for key in (
    'completed_executable_invocations', 'native_solve_api_calls', 'actual_requested_solve_iterations',
    'bootstrap_iterations', 'independently_qualified_final_vectors')}
work = {'scope': 'completed requested solves and bootstrap work only; CPU audits and waiting are not solver work',
        'completed_v606c': totals, 'initial_v606b': {key: initial[key] for key in combined},
        'zero_call_busy_v606c': {'completed_executable_invocations': 0, 'native_solve_api_calls': 0},
        'combined': combined, 'tiny_report_sha256': digest(root / 'tiny-wait/report.json'),
        'executable_wall_scope': 'sum of native executable invocations, excluding queue wait and independent audit subprocesses'}
(root / 'work-counts.json').write_text(json.dumps(work, indent=2))
index = {path.relative_to(root).as_posix(): {'bytes': path.stat().st_size, 'sha256': digest(path)}
         for path in sorted(root.rglob('*')) if path.is_file() and path.name != 'sha256.json'}
assert not any(name.endswith(('.so', '.exe', '.o', '.pyc')) for name in index)
(root / 'sha256.json').write_text(json.dumps(index, indent=2))
print(json.dumps({'indexed_files': len(index), 'indexed_bytes': sum(row['bytes'] for row in index.values()),
                  'combined_work': combined, 'index_sha256': digest(root / 'sha256.json')}))
