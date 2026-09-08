"""Six approved bounded known-point checks, with actual shared GPU lock."""
from pathlib import Path
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time

root = Path('/home/angus/spacepdhcg-persistent-known-point-v606c')
live = Path('/mnt/c/Users/Angus/Desktop/projects/spacecraft-trajectory-optmiser')
manifest = json.loads((root / 'manifest.json').read_text())
assert manifest['complete']
name = sys.argv[1] if len(sys.argv) == 2 else 'tiny'
assert len(sys.argv) <= 2 and Path(name).name == name and name not in ('.', '..')
out = root / name
out.mkdir(exist_ok=False)
binary = root / 'build/cuda-tests/persistent_snapshot_replay'
digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
assert digest(binary) == manifest['executable_sha256'] == '7d248cf1361cf8df83232951fee07cf9d274af06a6abf6e186823f6e9aae84d3'
assert digest(Path(manifest['immutable_core_path'])) == manifest['immutable_core_sha256'] == 'd4b0bea9672bb0cea612b7d4e8db5d448f1b79b9831be5710723276edd128633'
shutil.copy2(__file__, out / 'run_tiny.py')
shutil.copy2(live / 'scripts/gpu/audit_persistent_snapshot.py', out / 'independent_auditor.py')
env = {key: value for key, value in os.environ.items()
       if not key.startswith(('SPACEPDHCG_', 'QOCO_', 'PDHCG_', 'CUDA_VISIBLE_DEVICES', 'LD_LIBRARY_PATH'))}
env['CUDA_VISIBLE_DEVICES'] = '0'
report = {'complete': False, 'scope': 'known-qualified point input, zero-step residual measurement and actual bounded updates',
          'frozen_commit': manifest['frozen_commit'], 'manifest_sha256': digest(root / 'manifest.json'),
          'executable_sha256': digest(binary), 'core_sha256': manifest['immutable_core_sha256'],
          'runner_sha256': digest(Path(__file__)), 'cases': []}
def save():
    (out / 'report.json').write_text(json.dumps(report, indent=2))
def call(name, command):
    start = time.perf_counter()
    with (out / (name + '.log')).open('x') as log:
        result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=35)
    return {'name': name, 'command': command, 'returncode': result.returncode,
            'seconds': time.perf_counter() - start}
save()
try:
    with Path('/home/angus/.spacepdhcg-gpu.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for name, snapshot_name, point_name, folded in [
            ('unseeded-control', 'mixed.txt', None, False),
            ('original-generic', 'mixed.txt', 'mixed-initial-original.txt', False),
            ('translated-folded', 'mixed.txt', 'mixed-initial-translated.txt', True),
            ('shifted-original-generic', 'mixed-shifted.txt', 'mixed-shifted-initial-original.txt', False),
            ('shifted-translated-folded', 'mixed-shifted.txt', 'mixed-shifted-initial-translated.txt', True),
            ('weak-bound-folded', 'bounds-duplicate.txt', 'bounds-weak-initial.txt', True),
        ]:
            snapshot = root / 'fixtures' / snapshot_name
            command = [str(binary), str(snapshot), '--tolerance', '1e-9', '--iterations',
                       '1' if point_name else '100', '--deadline-seconds', '10', '--mode', 'cold']
            if folded:
                command.append('--fold-singleton-bounds')
            point = root / 'fixtures' / point_name if point_name else None
            if point:
                command.extend(['--initial-point', str(point)])
            row = call(name, command)
            row['snapshot_sha256'] = digest(snapshot)
            row['point_sha256'] = digest(point) if point else None
            report['cases'].append(row)
            save()
            assert row['returncode'] == 0, row
            lines = (out / (name + '.log')).read_text().splitlines()
            def records(prefix):
                prefix += ' '
                return [json.loads(line[len(prefix):]) for line in lines if line.startswith(prefix)]
            meta = records('PERSISTENT_REPLAY_META')
            final = records('PERSISTENT_REPLAY')
            pre = records('PERSISTENT_REPLAY_PRESTEP')
            bootstrap = records('PERSISTENT_REPLAY_BOOTSTRAP')
            initial = records('PERSISTENT_REPLAY_INITIAL_POINT')
            assert len(meta) == len(final) == 1 and meta[0]['library_sha256'] == manifest['immutable_core_sha256']
            assert final[0]['qualified_original'] and final[0]['iterations'] <= (1 if point else 100)
            assert final[0]['initial_point_applied'] == bool(point)
            if point:
                assert len(pre) == len(bootstrap) == len(initial) == 1
                assert bootstrap[0]['iterations'] <= 1 and bootstrap[0]['recovery_iterations'] == 0
                assert pre[0]['seeded_iterations'] == 0 and pre[0]['termination'] == 0
                assert pre[0]['native_seed_verified_unchanged'] and pre[0]['solve_epoch'] > 0
                assert pre[0]['warm_start_mode'] == 2 and pre[0]['warm_start_accepted']
                assert initial[0]['supplied_qualified'] and initial[0]['mapped_reference_qualified']
                assert initial[0]['strict_reconstructed_qualified'] == (name != 'weak-bound-folded')
            else:
                assert not pre and not bootstrap and not initial
            row['final'] = [{key: value for key, value in record.items()
                             if key not in ('x', 'x_solver', 'y', 'z', 's')} for record in final]
            row['prestep'] = pre
            row['bootstrap'] = bootstrap
            row['initial'] = [{key: value for key, value in record.items()
                               if key not in ('x', 'x_solver', 'dual_solver', 'y', 'z', 's')} for record in initial]
            save()
            audit_path = out / (name + '-audit.json')
            audited = call(name + '-audit', ['/home/angus/worktrees/spacepdhcg-literature-venv/bin/python',
                str(out / 'independent_auditor.py'), str(snapshot), str(out / (name + '.log')),
                '--output', str(audit_path), '--backend', 'persistent', '--coordinates', 'original',
                '--record-prefix', 'PERSISTENT_REPLAY'])
            assert audited['returncode'] == 0, audited
            row['independent_qualified_count'] = json.loads(audit_path.read_text())['qualified_count']
            save()
    report['complete'] = True
    report['qualified_calls'] = sum(case['independent_qualified_count'] for case in report['cases'])
    save()
except BaseException as error:
    report['failure'] = repr(error)
    save()
    raise
print(json.dumps({'complete': True, 'qualified_calls': report['qualified_calls'], 'output': str(out)}))
