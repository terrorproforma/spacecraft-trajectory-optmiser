"""Four automatic-grid cold comparisons, each with a fixed 100k/30s limit."""
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

live = Path(__file__).resolve().parents[2]
root = live / 'build/performance/auto-cold-v610'
root.mkdir(exist_ok=False)
out = root / 'run'
out.mkdir()
binary = Path('/home/angus/spacepdhcg-common-kkt-v609e/build/cuda-tests/persistent_snapshot_replay')
core = Path('/home/angus/spacepdhcg-common-kkt-v609d/build/cuda/libspacepdhcg_cuda.so')
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(binary) == '68683409e81b51acc30d9039349c7479b16b8320a16ea08967adcc2a5fd240ed'
assert sha(core) == 'a38d579579be2f47f9d8a8290a9440fd43f5beca2864773d9a5d6465a2ee61dd'
shutil.copyfile(__file__, root / 'run.py')
shutil.copyfile(live / 'scripts/gpu/audit_persistent_snapshot.py', root / 'independent_auditor.py')
spec = importlib.util.spec_from_file_location('audit_v610', root / 'independent_auditor.py')
auditor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(auditor)
assert sha(root / 'independent_auditor.py') == '0d944f768ad9c089499cc6e95cd1313677e153f650eced2c7d84b9815c00c7ff'
inputs = root / 'inputs'
inputs.mkdir()
for name, expected in [('conditioning', '1eb1b5a3979c5e86522337dfa94f264e354993163ed44129eff361371e3b49cf'),
                       ('difficult', '14fdc84e9b654df4ac16db71f0bce4fccfa62577f9620ae622ad2196e2f17080')]:
    shutil.copyfile(live / 'build/performance/known-point-replay-v606/inputs' / (name + '.txt'), inputs / (name + '.txt'))
    assert sha(inputs / (name + '.txt')) == expected
env = {k: v for k, v in os.environ.items() if not k.startswith(('SPACEPDHCG_', 'QOCO_', 'PDHCG_', 'CUDA_VISIBLE_DEVICES', 'LD_LIBRARY_PATH'))}
env['CUDA_VISIBLE_DEVICES'] = '0'
env['LD_LIBRARY_PATH'] = str(core.parent) + ':/usr/local/cuda-12.8/lib64'
report = {'complete': False, 'pid': os.getpid(), 'maximum_solve_calls': 4, 'iteration_cap_each': 100000,
          'deadline_seconds_each': 30, 'binary_sha256': sha(binary), 'core_sha256': sha(core),
          'runner_sha256': sha(root / 'run.py'), 'source_commit': 'f8b2ac7aeba9e29ea90f5814060bb8cfe5db22d7',
          'scope': 'automatic-grid comparison of existing stopping policies; no method change or SOTA claim', 'cases': []}
def save():
    (out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
def records(text, prefix):
    return [json.loads(line[len(prefix) + 1:]) for line in text.splitlines() if line.startswith(prefix + ' ')]
save()
try:
    with Path('/home/angus/.spacepdhcg-gpu.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        report['compute_processes_before'] = subprocess.check_output(['/usr/lib/wsl/lib/nvidia-smi', '--query-compute-apps=pid,process_name', '--format=csv,noheader'], text=True)
        report['gpu_identity'] = subprocess.check_output(['/usr/lib/wsl/lib/nvidia-smi', '--query-gpu=name,uuid,driver_version', '--format=csv,noheader'], text=True)
        save()
        assert not report['compute_processes_before'].strip(), 'GPU busy; no launch'
        for capture in ('conditioning', 'difficult'):
            for common in (False, True):
                name = capture + ('-common-auto' if common else '-natural-auto')
                snapshot = inputs / (capture + '.txt')
                command = [str(binary), str(snapshot), '--tolerance', '1e-9', '--iterations', '100000', '--deadline-seconds', '30', '--mode', 'cold']
                if common:
                    command.append('--common-kkt-stop')
                row = {'name': name, 'command': command, 'common': common}
                report['cases'].append(row)
                report['active_case'] = name
                save()
                start = time.perf_counter()
                try:
                    with (out / (name + '.log')).open('x') as log:
                        result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=45)
                    row['returncode'] = result.returncode
                except subprocess.TimeoutExpired:
                    row.update(returncode=None, external_timeout=True)
                row['wall_seconds'] = time.perf_counter() - start
                row['log_sha256'] = sha(out / (name + '.log'))
                save()
                assert row['returncode'] == 0, row
                text = (out / (name + '.log')).read_text()
                meta, final = records(text, 'PERSISTENT_REPLAY_META'), records(text, 'PERSISTENT_REPLAY')
                assert len(meta) == len(final) == 1
                meta, final = meta[0], final[0]
                assert meta['input_sha256'] == sha(snapshot) and meta['library_sha256'] == sha(core)
                assert meta['requested_execution_blocks'] is None and final['execution_blocks'] > 0
                assert not records(text, 'PERSISTENT_REPLAY_BOOTSTRAP')
                audit = auditor.audit(auditor.load_snapshot(snapshot), final, backend='persistent', coordinates='original')
                assert audit['qualified'] == final['qualified_original']
                if common:
                    gpu = final['gpu_common_kkt']
                    assert gpu['enabled']
                    if gpu['valid']:
                        assert gpu['passes'] == audit['passes_common_kkt_gate']
                    else:
                        assert not gpu['passes'] and final['termination'] == 'cancelled'
                    assert final['recovery_iterations'] == 0 and final['recovery_seconds'] == 0
                row['audit'] = audit
                row['final'] = {k: v for k, v in final.items() if k not in ('x', 'x_solver', 'y', 'z', 's')}
                save()
                print(json.dumps({'case': name, 'blocks': final['execution_blocks'], 'iterations': final['iterations'], 'solve_seconds': final['solve_seconds'], 'qualified': audit['qualified'], 'primal': audit['primal'], 'dual': audit['dual'], 'gap': audit['gap']}), flush=True)
    report['complete'] = True
    report.pop('active_case', None)
    report['solve_calls'] = len(report['cases'])
    save()
except BaseException as error:
    report['failure'] = repr(error)
    save()
    raise
