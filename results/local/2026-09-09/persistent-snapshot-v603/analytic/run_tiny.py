"""Bounded analytic validation; run in WSL only, under the shared GPU lock."""
from pathlib import Path
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time

root = Path('/home/angus/spacepdhcg-persistent-replay-v603')
live = Path('/mnt/c/Users/Angus/Desktop/projects/spacecraft-trajectory-optmiser')
manifest = json.loads((root / 'manifest.json').read_text())
assert manifest['complete']
output_name = sys.argv[1] if len(sys.argv) == 2 else 'tiny'
assert len(sys.argv) <= 2 and Path(output_name).name == output_name and output_name not in ('.', '..')
out = root / output_name
out.mkdir(exist_ok=False)
binary = root / 'build/cuda-tests/persistent_snapshot_replay'
for name in ['build/cuda/libspacepdhcg_cuda.so', 'build/cuda-tests/persistent_snapshot_replay']:
    assert hashlib.sha256((root / name).read_bytes()).hexdigest() == manifest[name]['sha256']
shutil.copy2(live / 'scripts/gpu/audit_persistent_snapshot.py', out / 'independent_auditor.py')
env = {key: value for key, value in os.environ.items()
       if not key.startswith(('SPACEPDHCG_', 'QOCO_', 'PDHCG_', 'CUDA_VISIBLE_DEVICES', 'LD_LIBRARY_PATH'))}
env['CUDA_VISIBLE_DEVICES'] = '0'
report = {'complete': False, 'scope': 'analytic diagnostic adapter checks, no trajectory qualification',
          'frozen_commit': manifest['frozen_commit'], 'cases': [], 'validation': []}
def record():
    (out / 'report.json').write_text(json.dumps(report, indent=2))
def call(name, command, expected=0, environment=env):
    start = time.perf_counter()
    with (out / (name + '.log')).open('x') as log:
        result = subprocess.run(command, env=environment, stdout=log, stderr=subprocess.STDOUT, timeout=30)
    row = {'name': name, 'command': command, 'returncode': result.returncode,
           'seconds': time.perf_counter() - start}
    assert result.returncode == expected, (name, result.returncode)
    return row

# A linked CUDA executable must reject malformed inputs, and validate good ones,
# with all GPUs hidden. This branch never constructs the CUDA storage helper.
hidden = dict(env, CUDA_VISIBLE_DEVICES='')
report['validation'].append(call('validate-only', [str(binary), str(root / 'fixtures/mixed.txt'), '--validate-only'], environment=hidden))
malformed = out / 'malformed.txt'
malformed.write_text((root / 'fixtures/mixed.txt').read_text() + 'trailing-data\n')
report['validation'].append(call('reject-before-cuda', [str(binary), str(malformed), '--tolerance', '1e-10', '--iterations', '100000', '--deadline-seconds', '10'], expected=1, environment=hidden))
record()
with Path('/home/angus/.spacepdhcg-gpu.lock').open('a+') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    for name, fixture, mode, repeats in [
        ('cold-original', 'mixed.txt', 'cold', 1),
        ('cold-shifted', 'mixed-shifted.txt', 'cold', 1),
        ('reuse-original', 'mixed.txt', 'reuse', 2),
        ('retained-shifted', 'mixed-shifted.txt', 'full-retained', 2),
    ]:
        snapshot = root / 'fixtures' / fixture
        command = [str(binary), str(snapshot), '--tolerance', '1e-10', '--iterations', '100000',
                   '--deadline-seconds', '10', '--mode', mode, '--repeats', str(repeats)]
        row = call(name, command)
        records = [json.loads(line[len('PERSISTENT_REPLAY '):])
                   for line in (out / (name + '.log')).read_text().splitlines()
                   if line.startswith('PERSISTENT_REPLAY ')]
        row['records'] = [{key: value for key, value in item.items()
                           if key not in ('x', 'x_solver', 'y', 'z', 's')} for item in records]
        report['cases'].append(row)
        record()
        assert len(records) == repeats and all(item['qualified_original'] for item in records), name
        for item in records:
            assert max(abs(item['x'][0] - 1), abs(item['x'][1])) < 1e-8
        audit_path = out / (name + '-audit.json')
        call(name + '-audit', ['/home/angus/worktrees/spacepdhcg-literature-venv/bin/python',
             str(out / 'independent_auditor.py'), str(snapshot), str(out / (name + '.log')),
             '--output', str(audit_path), '--backend', 'persistent', '--coordinates', 'original',
             '--record-prefix', 'PERSISTENT_REPLAY'])
        row['independent_qualified_count'] = json.loads(audit_path.read_text())['qualified_count']
        record()
report['complete'] = True
report['qualified_calls'] = sum(case['independent_qualified_count'] for case in report['cases'])
record()
print(json.dumps({'complete': True, 'qualified_calls': report['qualified_calls'], 'output': str(out)}))
