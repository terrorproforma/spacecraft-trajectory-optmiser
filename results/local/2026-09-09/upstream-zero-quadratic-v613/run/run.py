"""Four bounded exact-zero-Hessian reference calls with unchanged KKT gates."""
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

root = Path(__file__).resolve().parent / 'upstream-zero-q-v613'
out = root / 'run'
out.mkdir(exist_ok=False)
shutil.copyfile(__file__, out / 'run.py')
manifest = json.loads((root / 'manifest.json').read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
binary = root / 'upstream_snapshot_replay'
assert manifest['complete'] and sha(binary) == manifest['executable_sha256'] == 'd9987b87b90cfc0dd83c69db251e16d5bf29771ae884153e78d68b569e83237d'
for name, expected in manifest['source_files_sha256'].items():
    assert sha(root / 'source' / name) == expected
for name, expected in manifest['fixtures_sha256'].items():
    assert sha(root / 'fixtures' / name) == expected
spec = importlib.util.spec_from_file_location('audit_zero_q_v613', root / 'source/scripts/gpu/audit_persistent_snapshot.py')
auditor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(auditor)
env = {k: v for k, v in os.environ.items() if not k.startswith(('SPACEPDHCG_', 'QOCO_', 'PDHCG_', 'CUDA_VISIBLE_DEVICES', 'LD_LIBRARY_PATH'))}
env.update(CUDA_VISIBLE_DEVICES='0', LD_LIBRARY_PATH='/usr/local/cuda-12.8/lib64')
report = {'complete': False, 'pid': os.getpid(), 'maximum_native_calls': 4, 'native_calls': 0,
          'manifest_sha256': sha(root / 'manifest.json'), 'runner_sha256': sha(out / 'run.py'),
          'executable_sha256': sha(binary), 'cases': [],
          'scope': 'exact linear objective dispatch ablation; compare with archived v608 same-input same-caps reference; no mission or SOTA claim'}
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
        assert not report['compute_processes_before'].strip(), 'GPU busy; no calls launched'
        for seeded in (True, False):
            for capture in ('conditioning', 'difficult'):
                name = capture + ('-seeded' if seeded else '-cold')
                snapshot = root / 'fixtures' / (capture + '.txt')
                command = [str(binary), str(snapshot), '--iterations', '1' if seeded else '100000',
                           '--deadline-seconds', '10' if seeded else '60', '--omit-zero-quadratic']
                if seeded:
                    command += ['--initial-point', str(root / 'fixtures' / (capture + '-initial.txt'))]
                row = {'name': name, 'seeded': seeded, 'command': command}
                report['cases'].append(row)
                report['native_calls'] += 1
                report['active_case'] = name
                save()
                start = time.perf_counter()
                try:
                    with (out / (name + '.log')).open('x') as log:
                        result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=30 if seeded else 80)
                    row['returncode'] = result.returncode
                except subprocess.TimeoutExpired:
                    row.update(returncode=None, external_timeout=True)
                row['wall_seconds'] = time.perf_counter() - start
                row['log_sha256'] = sha(out / (name + '.log'))
                save()
                assert row['returncode'] == 0, row
                text = (out / (name + '.log')).read_text()
                meta, final = records(text, 'UPSTREAM_REPLAY_META'), records(text, 'UPSTREAM_REPLAY_RESULT')
                assert len(meta) == len(final) == 1 and len(records(text, 'UPSTREAM_REPLAY_DONE')) == 1
                meta, final = meta[0], final[0]
                assert meta['input_sha256'] == sha(snapshot) and meta['executable_sha256'] == sha(binary)
                assert meta['omit_zero_quadratic'] and meta['quadratic_exactly_zero'] and meta['quadratic_descriptor'] == 'nullptr_exact_linear_objective'
                assert final['inner_iterations'] == 0, 'expected exact NON_Q dispatch without inner solves'
                audit = auditor.audit(auditor.load_snapshot(snapshot), final | {'termination': final['termination_code']}, backend='persistent', coordinates='original')
                assert audit['qualified'] == final['qualified'] and audit['passes_common_kkt_gate'] == final['passes_common_kkt_gate']
                if seeded:
                    assert final['iterations'] == 0 and audit['qualified']
                row['audit'] = audit
                row['final'] = {k: v for k, v in final.items() if k not in ('x', 'y', 'z', 's', 'x_solver', 'normal_dual_solver')}
                save()
                print(json.dumps({'case': name, 'qualified': audit['qualified'], 'iterations': final['iterations'], 'inner_iterations': final['inner_iterations'], 'solve_wall_seconds': final['solve_wall_seconds'], 'primal': audit['primal'], 'dual': audit['dual'], 'gap': audit['gap']}), flush=True)
    report['complete'] = True
    report.pop('active_case', None)
    save()
except BaseException as error:
    report['failure'] = repr(error)
    save()
    raise
