"""Eight real-capture comparisons; four seeded bootstraps are counted separately."""
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np

live = Path(__file__).resolve().parents[2]
frozen = Path('/home/angus/spacepdhcg-common-kkt-v609e')
root = live / 'build/performance/common-kkt-real-v609e'
root.mkdir(exist_ok=False)
out = root / 'run'
out.mkdir()
binary = frozen / 'build/cuda-tests/persistent_snapshot_replay'
core = Path('/home/angus/spacepdhcg-common-kkt-v609d/build/cuda/libspacepdhcg_cuda.so')
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
assert sha(binary) == '68683409e81b51acc30d9039349c7479b16b8320a16ea08967adcc2a5fd240ed'
assert sha(core) == 'a38d579579be2f47f9d8a8290a9440fd43f5beca2864773d9a5d6465a2ee61dd'
manifest = json.loads((frozen / 'manifest.json').read_text())
assert manifest['complete'] and manifest['library_sha256'] == sha(core)
shutil.copyfile(frozen / 'manifest.json', root / 'manifest.json')
shutil.copyfile(__file__, root / 'run.py')
shutil.copyfile(live / 'scripts/gpu/audit_persistent_snapshot.py', root / 'independent_auditor.py')
assert sha(root / 'independent_auditor.py') == '0d944f768ad9c089499cc6e95cd1313677e153f650eced2c7d84b9815c00c7ff'
inputs = root / 'inputs'
inputs.mkdir()
for name in ('conditioning', 'difficult'):
    for suffix in ('.txt', '-initial.txt'):
        shutil.copyfile(live / 'build/performance/known-point-replay-v606/inputs' / (name + suffix), inputs / (name + suffix))
assert sha(inputs / 'conditioning.txt') == '1eb1b5a3979c5e86522337dfa94f264e354993163ed44129eff361371e3b49cf'
assert sha(inputs / 'difficult.txt') == '14fdc84e9b654df4ac16db71f0bce4fccfa62577f9620ae622ad2196e2f17080'
spec = importlib.util.spec_from_file_location('independent_auditor', root / 'independent_auditor.py')
auditor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(auditor)
env = {key: value for key, value in os.environ.items()
       if not key.startswith(('SPACEPDHCG_', 'QOCO_', 'PDHCG_', 'CUDA_VISIBLE_DEVICES', 'LD_LIBRARY_PATH'))}
env['CUDA_VISIBLE_DEVICES'] = '0'
env['LD_LIBRARY_PATH'] = str(core.parent) + ':/usr/local/cuda-12.8/lib64'
report = {'complete': False, 'pid': os.getpid(), 'executable_sha256': sha(binary), 'core_sha256': sha(core),
          'source_manifest_sha256': sha(root / 'manifest.json'), 'runner_sha256': sha(Path(__file__)),
          'scope': 'known-point stopping plus paired single-block cold diagnostics, no throughput or mission score claim',
          'maximum_executions': 8, 'maximum_solve_API_calls': 12, 'maximum_bootstrap_iterations': 4,
          'inputs_sha256': {p.name: sha(p) for p in inputs.iterdir()}, 'cases': [], 'executions_started': 0}

def save():
    (out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')

def records(text, prefix):
    return [json.loads(line[len(prefix) + 1:]) for line in text.splitlines() if line.startswith(prefix + ' ')]

save()
try:
    with Path('/home/angus/.spacepdhcg-gpu.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        process = subprocess.check_output(['/usr/lib/wsl/lib/nvidia-smi', '--query-compute-apps=pid,process_name', '--format=csv,noheader'], text=True)
        report['compute_processes_before'] = process
        report['gpu_identity'] = subprocess.check_output(['/usr/lib/wsl/lib/nvidia-smi', '--query-gpu=name,uuid,driver_version', '--format=csv,noheader'], text=True)
        save()
        assert not process.strip(), 'other compute work present; do not launch'
        cases = [(name, True, True, blocks) for name in ('conditioning', 'difficult') for blocks in (0, 2)]
        cases += [(name, False, common, 0) for name in ('conditioning', 'difficult') for common in (False, True)]
        for capture, seeded, common, blocks in cases:
            name = f"{capture}-{'seeded' if seeded else 'cold'}-{'common' if common else 'natural'}-blocks{blocks}"
            snapshot = inputs / (capture + '.txt')
            command = [str(binary), str(snapshot), '--tolerance', '1e-9', '--iterations', '1' if seeded else '100000',
                       '--deadline-seconds', '20' if seeded else '60', '--execution-blocks', str(blocks), '--mode', 'cold']
            if common:
                command.append('--common-kkt-stop')
            if seeded:
                command += ['--initial-point', str(inputs / (capture + '-initial.txt'))]
            row = {'name': name, 'capture': capture, 'seeded': seeded, 'common': common, 'execution_blocks': blocks, 'command': command}
            report['executions_started'] += 1
            report['active_case'] = name
            report['cases'].append(row)
            save()
            start = time.perf_counter()
            try:
                with (out / (name + '.log')).open('x') as log:
                    result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=65 if seeded else 80)
                row['returncode'] = result.returncode
            except subprocess.TimeoutExpired:
                row.update(returncode=None, external_timeout=True)
            row['wall_seconds'] = time.perf_counter() - start
            save()
            assert row['returncode'] == 0, row
            text = (out / (name + '.log')).read_text()
            meta, finals = records(text, 'PERSISTENT_REPLAY_META'), records(text, 'PERSISTENT_REPLAY')
            pre, bootstrap = records(text, 'PERSISTENT_REPLAY_PRESTEP'), records(text, 'PERSISTENT_REPLAY_BOOTSTRAP')
            initial = records(text, 'PERSISTENT_REPLAY_INITIAL_POINT')
            assert len(meta) == len(finals) == 1
            final = finals[0]
            assert meta[0]['input_sha256'] == sha(snapshot) and meta[0]['library_sha256'] == sha(core)
            assert meta[0]['requested_execution_blocks'] == final['execution_blocks'] == blocks
            assert len(bootstrap) == len(pre) == len(initial) == int(seeded)
            row['solve_API_calls'] = 1 + len(bootstrap)
            row['bootstrap'] = bootstrap
            row['prestep'] = pre
            audited = auditor.audit(auditor.load_snapshot(snapshot), final, backend='persistent', coordinates='original')
            assert audited['qualified'] == final['qualified_original']
            row['independent_audit'] = audited
            if common:
                gpu = final['gpu_common_kkt']
                assert gpu['enabled'] and gpu['valid'] and gpu['finite']
                assert gpu['passes'] == audited['passes_common_kkt_gate']
                assert final['recovery_iterations'] == 0 and final['recovery_seconds'] == 0
            if seeded:
                assert audited['qualified'] and final['iterations'] == 0
                assert final['gpu_common_kkt']['evaluations'] == 1
                assert pre[0]['native_seed_verified_unchanged'] and bootstrap[0]['iterations'] <= 1
                for key in ('x', 'y', 'z'):
                    assert np.array_equal(np.asarray(initial[0][key], dtype=np.float64).view(np.uint64),
                                          np.asarray(final[key], dtype=np.float64).view(np.uint64)), key
                row['seeded_primal_dual_final_bits_unchanged'] = True
            row['final'] = {key: value for key, value in final.items() if key not in ('x', 'x_solver', 'y', 'z', 's')}
            save()
            print(json.dumps({'case': name, 'qualified': audited['qualified'], 'iterations': final['iterations'],
                              'gap': audited['gap'], 'native_seconds': final['solve_seconds']}), flush=True)
    report['complete'] = True
    report.pop('active_case', None)
    report['solve_API_calls'] = sum(row['solve_API_calls'] for row in report['cases'])
    report['bootstrap_iterations'] = sum(b['iterations'] for row in report['cases'] for b in row['bootstrap'])
    assert report['solve_API_calls'] <= 12 and report['bootstrap_iterations'] <= 4
    save()
except BaseException as error:
    report['failure'] = repr(error)
    save()
    raise
print(json.dumps({'complete': True, 'solve_API_calls': report['solve_API_calls'], 'bootstrap_iterations': report['bootstrap_iterations']}))
