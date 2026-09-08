"""One bounded native reference comparison under the shared local GPU lock."""
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

root = Path(__file__).resolve().parent / 'upstream-snapshot-v608'
out = root / 'run'
out.mkdir(exist_ok=False)
shutil.copyfile(__file__, out / 'run.py')
manifest = json.loads((root / 'manifest.json').read_text())
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
assert manifest['complete'] and manifest['gpu_calls'] == 0
binary = root / 'upstream_snapshot_replay'
assert sha(binary) == manifest['executable_sha256'] == 'd3e4673f6974e59500cae6f9b072f7a1305a68fd0862f31ea95c6e8b7e9d52e7'
for name, expected in manifest['source_files_sha256'].items():
    assert sha(root / 'source' / name) == expected, name
for name, expected in manifest['fixtures_sha256'].items():
    assert sha(root / 'fixtures' / name) == expected, name
auditor_path = root / 'source/scripts/gpu/audit_persistent_snapshot.py'
spec = importlib.util.spec_from_file_location('independent_kkt', auditor_path)
auditor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(auditor)
env = {key: value for key, value in os.environ.items()
       if not key.startswith(('SPACEPDHCG_', 'QOCO_', 'PDHCG_', 'CUDA_VISIBLE_DEVICES', 'LD_LIBRARY_PATH'))}
env['CUDA_VISIBLE_DEVICES'] = '0'
env['LD_LIBRARY_PATH'] = '/usr/local/cuda-12.8/lib64'
report = {'complete': False, 'manifest_sha256': sha(root / 'manifest.json'),
          'runner_sha256': sha(Path(__file__)), 'source_sha256': manifest['source_sha256'],
          'executable_sha256': sha(binary), 'gpu_calls': 0, 'cases': [], 'cpu_checks': [],
          'scope': 'pinned upstream one-shot diagnostic, no fleet change or throughput claim',
          'maximum_native_calls': 8, 'maximum_real_cold_iterations_per_capture': 100000,
          'independent_status_adapter': 'upstream termination_code1 is accepted; original auditor formulas unchanged'}

def save():
    (out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')

def records(log, prefix):
    return [json.loads(line[len(prefix) + 1:]) for line in log.splitlines() if line.startswith(prefix + ' ')]

def call(name, command, limit, call_env):
    start = time.perf_counter()
    row = {'name': name, 'command': command, 'started_at_unix': time.time()}
    try:
        with (out / (name + '.log')).open('x') as log:
            run = subprocess.run(command, env=call_env, stdout=log, stderr=subprocess.STDOUT, timeout=limit)
        row['returncode'] = run.returncode
    except subprocess.TimeoutExpired:
        row.update(returncode=None, external_timeout=True)
    row['wall_seconds'] = time.perf_counter() - start
    return row

save()
try:
    hidden = env | {'CUDA_VISIBLE_DEVICES': '-1'}
    for name in ('mixed', 'mixed-shifted', 'conditioning', 'difficult'):
        row = call('validate-' + name, [str(binary), str(root / 'fixtures' / (name + '.txt')), '--validate-only'], 15, hidden)
        report['cpu_checks'].append(row)
        save()
        assert row['returncode'] == 0, row
    row = call('reject-duplicate', [str(binary), str(root / 'fixtures/mixed.txt'), '--validate-only', '--validate-only'], 15, hidden)
    report['cpu_checks'].append(row)
    save()
    assert row['returncode'] == 2, row
    with Path('/home/angus/.spacepdhcg-gpu.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = subprocess.check_output(['/usr/lib/wsl/lib/nvidia-smi', '--query-compute-apps=pid,process_name', '--format=csv,noheader'], text=True)
        report['gpu_processes_before'] = state
        report['gpu_identity'] = subprocess.check_output(['/usr/lib/wsl/lib/nvidia-smi', '--query-gpu=name,uuid,driver_version', '--format=csv,noheader'], text=True)
        save()
        if state.strip():
            raise RuntimeError('GPU has existing compute work; preserve this attempt without running')
        cases = [
            ('mixed-seeded', 'mixed', 'mixed-initial-original', 1, 10),
            ('shifted-seeded', 'mixed-shifted', 'mixed-shifted-initial-translated', 1, 10),
            ('mixed-cold', 'mixed', None, 10000, 10),
            ('shifted-cold', 'mixed-shifted', None, 10000, 10),
            ('conditioning-cold', 'conditioning', None, 100000, 60),
            ('difficult-cold', 'difficult', None, 100000, 60),
            ('conditioning-seeded', 'conditioning', 'conditioning-initial', 1, 10),
            ('difficult-seeded', 'difficult', 'difficult-initial', 1, 10),
        ]
        for name, fixture, point, iterations, deadline in cases:
            snapshot = root / 'fixtures' / (fixture + '.txt')
            command = [str(binary), str(snapshot), '--iterations', str(iterations), '--deadline-seconds', str(deadline)]
            if point:
                command += ['--initial-point', str(root / 'fixtures' / (point + '.txt'))]
            report['gpu_calls'] += 1
            report['starting_case'] = name
            save()
            row = call(name, command, deadline + 20, env)
            report['cases'].append(row)
            save()
            assert row['returncode'] == 0, row
            log = (out / (name + '.log')).read_text()
            meta = records(log, 'UPSTREAM_REPLAY_META')
            results = records(log, 'UPSTREAM_REPLAY_RESULT')
            done = records(log, 'UPSTREAM_REPLAY_DONE')
            assert len(meta) == len(results) == len(done) == 1
            result = results[0]
            assert meta[0]['input_sha256'] == result['input_sha256'] == sha(snapshot)
            assert meta[0]['executable_sha256'] == sha(binary)
            # Only termination-code plumbing is adapted; all original equations,
            # norms, gap and complementarity tests are imported without changes.
            audited = auditor.audit(auditor.load_snapshot(snapshot), result | {'termination': result['termination_code']},
                                    backend='persistent', coordinates='original')
            audited['backend'] = 'pinned_upstream_C_API'
            audited['termination_code_source'] = result['termination_code']
            assert audited['qualified'] == result['qualified']
            assert audited['passes_common_kkt_gate'] == result['passes_common_kkt_gate']
            row['independent_audit'] = audited
            row['result'] = {key: value for key, value in result.items() if key not in ('x', 'y', 'z', 's', 'x_solver', 'normal_dual_solver')}
            save()
            if fixture.startswith('mixed'):
                # Analytic mapping validation is separate from the strict qualification gate.
                assert max(abs(result['x'][0] - 1), abs(result['x'][1])) < 1e-6
                assert max(audited['primal'], audited['dual'], audited['gap']) < 1e-6
                if point:
                    assert audited['qualified'], 'exact analytic known point must remain qualified'
            print(json.dumps({'case': name, 'termination': result['termination'], 'iterations': result['iterations'],
                              'qualified': audited['qualified'], 'gap': audited['gap'], 'wall_seconds': row['wall_seconds']}), flush=True)
    report['complete'] = True
    report.pop('starting_case', None)
    report['qualified_calls'] = sum(case['independent_audit']['qualified'] for case in report['cases'])
    save()
except BaseException as error:
    report['failure'] = repr(error)
    save()
    raise
print(json.dumps({'complete': True, 'gpu_calls': report['gpu_calls'], 'qualified_calls': report['qualified_calls']}))
