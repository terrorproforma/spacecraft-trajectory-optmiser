"""Four bounded exact-zero-Hessian reference calls with unchanged KKT gates."""
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

root = Path(__file__).resolve().parent / 'upstream-zero-q-v613'
out = root / 'run'
resume = sys.argv[1:] == ['--resume']
assert resume or not sys.argv[1:]
out.mkdir(exist_ok=resume)
shutil.copyfile(__file__, out / ('resume.py' if resume else 'run.py'))
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
if resume:
    report = json.loads((out / 'report.json').read_text())
    assert not report['complete'] and report['native_calls'] == len(report['cases']) == 3
    assert all(row['returncode'] == 0 for row in report['cases'])
    shutil.copyfile(out / 'report.json', out / 'first-observation-failure.json')
    report['observation_failure'] = report.pop('failure')
    report['resume_runner_sha256'] = sha(out / 'resume.py')
    report['resume_pid'] = os.getpid()
    report['native_inner_counter_semantics'] = 'pdhg_core_op.cu increments inner_solver->total_count once unconditionally per outer PDHG update, including NON_Q; this is not a BB iteration'
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
                previous = next((r for r in report['cases'] if r['name'] == name), None)
                row = previous if previous is not None else {'name': name, 'seeded': seeded, 'command': command}
                if previous is None:
                    report['cases'].append(row)
                    report['native_calls'] += 1
                report['active_case'] = name
                save()
                if previous is None:
                    start = time.perf_counter()
                    try:
                        with (out / (name + '.log')).open('x') as log:
                            result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=30 if seeded else 80)
                        row['returncode'] = result.returncode
                    except subprocess.TimeoutExpired:
                        row.update(returncode=None, external_timeout=True)
                    row['wall_seconds'] = time.perf_counter() - start
                else:
                    assert row['command'] == command and sha(out / (name + '.log')) == row['log_sha256']
                    row['existing_completed_call_reused_without_rerun'] = True
                row['log_sha256'] = sha(out / (name + '.log'))
                save()
                assert row['returncode'] == 0, row
                text = (out / (name + '.log')).read_text()
                meta, final = records(text, 'UPSTREAM_REPLAY_META'), records(text, 'UPSTREAM_REPLAY_RESULT')
                assert len(meta) == len(final) == 1 and len(records(text, 'UPSTREAM_REPLAY_DONE')) == 1
                meta, final = meta[0], final[0]
                assert meta['input_sha256'] == sha(snapshot) and meta['executable_sha256'] == sha(binary)
                assert meta['omit_zero_quadratic'] and meta['quadratic_exactly_zero'] and meta['quadratic_descriptor'] == 'nullptr_exact_linear_objective'
                # The upstream counter includes one unconditional increment per outer update.
                assert final['inner_iterations'] == final['iterations'], 'unexpected extra inner iterations on NON_Q path'
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
