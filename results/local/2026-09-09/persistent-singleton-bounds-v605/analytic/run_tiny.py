"""Bounded exact-singleton importer test; WSL shared GPU lock is mandatory."""
from pathlib import Path
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time

root = Path('/home/angus/spacepdhcg-persistent-replay-v604b')
live = Path('/mnt/c/Users/Angus/Desktop/projects/spacecraft-trajectory-optmiser')
manifest = json.loads((root / 'manifest.json').read_text())
assert manifest['complete']
name = sys.argv[1] if len(sys.argv) == 2 else 'tiny'
assert len(sys.argv) <= 2 and Path(name).name == name and name not in ('.', '..')
out = root / name
out.mkdir(exist_ok=False)
binary = root / 'build/cuda-tests/persistent_snapshot_replay'
assert hashlib.sha256(binary.read_bytes()).hexdigest() == manifest['executable_sha256']
assert hashlib.sha256(Path(manifest['immutable_core_path']).read_bytes()).hexdigest() == manifest['immutable_core_sha256']
shutil.copy2(live / 'scripts/gpu/audit_persistent_snapshot.py', out / 'independent_auditor.py')
env = {key: value for key, value in os.environ.items()
       if not key.startswith(('SPACEPDHCG_', 'QOCO_', 'PDHCG_', 'CUDA_VISIBLE_DEVICES', 'LD_LIBRARY_PATH'))}
env['CUDA_VISIBLE_DEVICES'] = '0'
report = {'complete': False, 'scope': 'optional exact singleton-bound diagnostic, not trajectory certification',
          'frozen_commit': manifest['frozen_commit'], 'cases': []}
def save():
    (out / 'report.json').write_text(json.dumps(report, indent=2))
def call(name, command, environment=env):
    start = time.perf_counter()
    with (out / (name + '.log')).open('x') as log:
        result = subprocess.run(command, env=environment, stdout=log, stderr=subprocess.STDOUT, timeout=30)
    row = {'name': name, 'command': command, 'returncode': result.returncode,
           'seconds': time.perf_counter() - start}
    assert result.returncode == 0, row
    return row
report['validation'] = call('validate-no-gpu', [str(binary), str(root / 'fixtures/bounds-duplicate.txt'),
    '--fold-singleton-bounds', '--validate-only'], dict(env, CUDA_VISIBLE_DEVICES=''))
save()
with Path('/home/angus/.spacepdhcg-gpu.lock').open('a+') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    for name, fixture, folded, mode, repeats in [
        ('default-mixed', 'mixed.txt', False, 'cold', 1),
        ('default-duplicates', 'bounds-duplicate.txt', False, 'cold', 1),
        ('folded-mixed', 'mixed.txt', True, 'cold', 1),
        ('folded-shifted', 'mixed-shifted.txt', True, 'cold', 1),
        ('folded-duplicates', 'bounds-duplicate.txt', True, 'cold', 1),
        ('folded-fixed-retained', 'bounds-fixed.txt', True, 'full-retained', 2),
    ]:
        snapshot = root / 'fixtures' / fixture
        command = [str(binary), str(snapshot), '--tolerance', '1e-10', '--iterations', '100000',
                   '--deadline-seconds', '10', '--mode', mode, '--repeats', str(repeats)]
        if folded:
            command.append('--fold-singleton-bounds')
        row = call(name, command)
        lines = (out / (name + '.log')).read_text().splitlines()
        meta = [json.loads(line[len('PERSISTENT_REPLAY_META '):]) for line in lines
                if line.startswith('PERSISTENT_REPLAY_META ')]
        records = [json.loads(line[len('PERSISTENT_REPLAY '):]) for line in lines
                   if line.startswith('PERSISTENT_REPLAY ')]
        assert len(meta) == 1 and meta[0]['library_sha256'] == manifest['immutable_core_sha256']
        row['metadata'] = meta[0]
        row['records'] = [{key: value for key, value in item.items()
                          if key not in ('x', 'x_solver', 'y', 'z', 's')} for item in records]
        report['cases'].append(row)
        save()
        assert len(records) == repeats and all(item['qualified_original'] for item in records), name
        assert all(item['folded_dual_reconstruction_supported'] for item in records), name
        audit_path = out / (name + '-audit.json')
        call(name + '-audit', ['/home/angus/worktrees/spacepdhcg-literature-venv/bin/python',
             str(out / 'independent_auditor.py'), str(snapshot), str(out / (name + '.log')),
             '--output', str(audit_path), '--backend', 'persistent', '--coordinates', 'original',
             '--record-prefix', 'PERSISTENT_REPLAY'])
        row['independent_qualified_count'] = json.loads(audit_path.read_text())['qualified_count']
        save()
report['complete'] = True
report['qualified_calls'] = sum(case['independent_qualified_count'] for case in report['cases'])
save()
print(json.dumps({'complete': True, 'qualified_calls': report['qualified_calls'], 'output': str(out)}))
