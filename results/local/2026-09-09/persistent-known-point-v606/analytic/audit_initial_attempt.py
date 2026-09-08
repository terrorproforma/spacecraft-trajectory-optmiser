"""Audit completed vectors from preserved v606b logs, without rewriting malformed metadata."""
from pathlib import Path
import hashlib
import importlib.util
import json
import re

root = Path(__file__).resolve().parent
attempt = root / 'initial-v606b'
digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
spec = importlib.util.spec_from_file_location('frozen_auditor', attempt / 'independent_auditor.py')
auditor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(auditor)
report = {'scope': 'CPU re-audit of two completed final-vector records from the preserved partial attempt; raw metadata is not repaired',
          'cases': [], 'gpu_calls_launched_by_this_script': 0,
          'auditor_sha256': digest(attempt / 'independent_auditor.py')}
for row in json.loads((attempt / 'report.json').read_text())['cases']:
    path = attempt / (row['name'] + '.log')
    lines = path.read_text().splitlines()
    snapshot = root / 'fixtures' / Path(row['command'][1]).name
    assert digest(snapshot) == row['snapshot_sha256']
    # The broken fields occur after these two complete, unchanged identity fields.
    meta = next(line for line in lines if line.startswith('PERSISTENT_REPLAY_META '))
    assert re.search(r'"input_sha256":"([0-9a-f]{64})"', meta).group(1) == digest(snapshot)
    assert '"coordinate_system":"original"' in meta
    final = [json.loads(line[len('PERSISTENT_REPLAY '):]) for line in lines if line.startswith('PERSISTENT_REPLAY ')]
    bootstrap = [json.loads(line[len('PERSISTENT_REPLAY_BOOTSTRAP '):]) for line in lines if line.startswith('PERSISTENT_REPLAY_BOOTSTRAP ')]
    assert len(final) == 1
    result = auditor.audit(auditor.load_snapshot(snapshot), final[0], backend='persistent', coordinates='original')
    assert result['qualified']
    report['cases'].append({'name': row['name'], 'log_sha256': digest(path), 'snapshot_sha256': digest(snapshot),
                            'point_sha256': row['point_sha256'], 'process_returncode': row['returncode'],
                            'actual_iterations': final[0]['iterations'], 'native_solve_api_calls': 1 + len(bootstrap),
                            'bootstrap_iterations': sum(x['iterations'] for x in bootstrap),
                            'native_qualified': final[0]['qualified_original'], 'independent_audit': result})
report['completed_executable_invocations'] = len(report['cases'])
report['native_solve_api_calls'] = sum(x['native_solve_api_calls'] for x in report['cases'])
report['actual_requested_solve_iterations'] = sum(x['actual_iterations'] for x in report['cases'])
report['bootstrap_iterations'] = sum(x['bootstrap_iterations'] for x in report['cases'])
report['independently_qualified_final_vectors'] = sum(x['independent_audit']['qualified'] for x in report['cases'])
with (root / 'initial-attempt-vector-audit.json').open('x') as output:
    json.dump(report, output, indent=2)
print(json.dumps({key: value for key, value in report.items() if key != 'cases'}))
