"""Two supplied starts and four matched-grid cold comparisons; no automatic retries."""
import argparse
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

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--frozen-root', type=Path, required=True)
parser.add_argument('--manifest-sha256', required=True)
parser.add_argument('--blocks', type=int, choices=[128], required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--tiny-report', type=Path, required=True)
args = parser.parse_args()
assert args.blocks > 0
live = Path(__file__).resolve().parents[2]
root = args.output.resolve()
root.mkdir(exist_ok=False)
out = root / 'run'
out.mkdir()
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
frozen = args.frozen_root.resolve()
assert sha(frozen / 'manifest.json') == args.manifest_sha256
manifest = json.loads((frozen / 'manifest.json').read_text())
assert manifest['complete']
tiny_path=args.tiny_report.resolve()
tiny=json.loads(tiny_path.read_text())
assert tiny['complete'] and tiny['status']=='passed' and tiny['manifest_sha256']==args.manifest_sha256
assert tiny['actual_solve_api_calls']==9 and tiny['actual_optimization_iterations']==10
assert tiny['core_sha256']==manifest['library_sha256'] and tiny['test_sha256']==manifest['persistent_l1_test_sha256']
binary = frozen / 'build/cuda-tests/persistent_snapshot_replay'
core = frozen / 'build/cuda/libspacepdhcg_cuda.so'
assert sha(binary) == manifest['persistent_snapshot_replay_sha256']
assert sha(core) == manifest['library_sha256']
for name, expected in manifest['source_sha256'].items():
    assert sha(frozen / 'repo' / name) == expected, name
shutil.copyfile(frozen / 'manifest.json', root / 'manifest.json')
shutil.copyfile(__file__, root / 'run.py')
shutil.copyfile(tiny_path,root/'tiny-report.json')
shutil.copyfile(live / 'scripts/gpu/audit_persistent_snapshot.py', root / 'independent_auditor.py')
assert sha(root / 'independent_auditor.py') == '0d944f768ad9c089499cc6e95cd1313677e153f650eced2c7d84b9815c00c7ff'
spec = importlib.util.spec_from_file_location('audit_l1_v615', root / 'independent_auditor.py')
auditor = importlib.util.module_from_spec(spec)
exec(compile((root / 'independent_auditor.py').read_bytes(), str(root / 'independent_auditor.py'), 'exec'), auditor.__dict__)
inputs = root / 'inputs'
inputs.mkdir()
for capture in ('conditioning', 'difficult'):
    for suffix in ('.txt', '-initial.txt'):
        shutil.copyfile(live / 'build/performance/known-point-replay-v606/inputs' / (capture + suffix), inputs / (capture + suffix))
assert sha(inputs / 'conditioning.txt') == '1eb1b5a3979c5e86522337dfa94f264e354993163ed44129eff361371e3b49cf'
assert sha(inputs / 'difficult.txt') == '14fdc84e9b654df4ac16db71f0bce4fccfa62577f9620ae622ad2196e2f17080'
assert sha(inputs / 'conditioning-initial.txt') == '9caf303469dcd254d56b74f8bf3818c8044f51a3a20bc3c07835b76dd23c5632'
assert sha(inputs / 'difficult-initial.txt') == 'b92be6c5a0051ab99c000c459ef466c69fc8b2fd669c38b3580aea000d926da6'
env = {k: v for k, v in os.environ.items() if not k.startswith(('SPACEPDHCG_', 'QOCO_', 'PDHCG_', 'CUDA_VISIBLE_DEVICES', 'LD_LIBRARY_PATH'))}
env.update(CUDA_VISIBLE_DEVICES='0', LD_LIBRARY_PATH=str(core.parent) + ':/usr/local/cuda-12.8/lib64')
cases = [(capture, 'l1', True) for capture in ('conditioning', 'difficult')]
cases += [(capture, mode, False) for capture in ('conditioning', 'difficult') for mode in ('off', 'l1')]
report = {'complete': False, 'pid': os.getpid(), 'maximum_executions': 6, 'maximum_solve_API_calls': 8,
          'maximum_bootstrap_iterations': 2, 'cold_iteration_cap': 100000, 'cold_deadline_seconds': 30,
          'execution_blocks': args.blocks, 'capacity_policy': 'native setter validates solve and scaling occupancy before any solve; abort on unsupported 128, never fall back', 'binary_sha256': sha(binary), 'core_sha256': sha(core),
          'tiny_report_sha256': sha(tiny_path), 'manifest_sha256': args.manifest_sha256, 'runner_sha256': sha(root / 'run.py'),
          'inputs_sha256': {p.name: sha(p) for p in inputs.iterdir()}, 'executions_started': 0, 'cases': [],
          'scope': 'fixed original common gates; same-build dual-first generic LP versus exact L1 prox with reduced preconditioning; full-layout memory retained; no SOTA or mission-backend claim'}
def save():
    (out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
def invalid_constant(value):
    raise ValueError('nonfinite JSON constant: '+value)
def records(text, prefix):
    # parse_int=float preserves a written -0 as a signed FP64 zero.
    return [json.loads(line[len(prefix) + 1:],parse_int=float,parse_constant=invalid_constant) for line in text.splitlines() if line.startswith(prefix + ' ')]
save()
try:
    with Path('/home/angus/.spacepdhcg-gpu.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        report['compute_processes_before'] = subprocess.check_output(['/usr/lib/wsl/lib/nvidia-smi', '--query-compute-apps=pid,process_name', '--format=csv,noheader'], text=True)
        report['gpu_identity'] = subprocess.check_output(['/usr/lib/wsl/lib/nvidia-smi', '--query-gpu=name,uuid,driver_version', '--format=csv,noheader'], text=True)
        save()
        assert not report['compute_processes_before'].strip(), 'GPU busy; no launch'
        assert 'GPU-4df2f6b5-e866-14a0-eeac-332cb2b757d4' in report['gpu_identity']
        for capture, mode, seeded in cases:
            name = f"{capture}-{'seeded' if seeded else 'cold'}-{mode}"
            snapshot = inputs / (capture + '.txt')
            command = [str(binary), str(snapshot), '--tolerance', '1e-9', '--iterations', '1' if seeded else '100000',
                       '--deadline-seconds', '20' if seeded else '30', '--mode', 'cold', '--execution-blocks', str(args.blocks),
                       '--common-kkt-stop']
            if mode=='l1':command+=['--l1-prox']
            if seeded:
                command += ['--initial-point', str(inputs / (capture + '-initial.txt'))]
            row = {'name': name, 'capture': capture, 'mode': mode, 'seeded': seeded, 'command': command}
            report['cases'].append(row)
            report['executions_started'] += 1
            report['active_case'] = name
            save()
            start = time.perf_counter()
            try:
                with (out / (name + '.log')).open('x') as log:
                    result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=65 if seeded else 50)
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
            bootstrap = records(text, 'PERSISTENT_REPLAY_BOOTSTRAP')
            initial = records(text, 'PERSISTENT_REPLAY_INITIAL_POINT')
            pre = records(text, 'PERSISTENT_REPLAY_PRESTEP')
            assert len(bootstrap) == len(initial) == len(pre) == int(seeded)
            assert meta['input_sha256'] == sha(snapshot) and meta['library_sha256'] == sha(core)
            assert meta['requested_execution_blocks'] == final['execution_blocks'] == args.blocks
            assert meta['halpern_mode']=='off' and meta['l1_prox']==(mode=='l1')
            assert meta['source_commit']==manifest['frozen_commit']
            assert meta['source_sha256']==manifest['compiled_snapshot_source_sha256']
            audit = auditor.audit(auditor.load_snapshot(snapshot), final, backend='persistent', coordinates='original')
            assert audit['qualified'] == final['qualified_original']
            gpu = final['gpu_common_kkt']
            assert gpu['enabled']
            if gpu['valid']:
                assert gpu['passes'] == audit['passes_common_kkt_gate']
            else:
                assert not gpu['passes'] and final['termination'] in (3, 4)
            assert final['recovery_iterations'] == 0 and final['recovery_seconds'] == 0
            if seeded:
                assert audit['qualified'] and final['iterations'] == 0 and final['l1']['updates'] == 0 and final['l1']['completions']==0
                assert final['termination']==1 and initial[0]['supplied_qualified'] and initial[0]['mapped_reference_qualified']
                assert meta['initial_point_sha256']==sha(inputs/(capture+'-initial.txt'))
                assert initial[0]['point_sha256']==meta['initial_point_sha256']
                assert pre[0]['native_seed_verified_unchanged'] and bootstrap[0]['iterations'] <= 1
                for key in ('x', 'y', 'z'):
                    assert np.array_equal(np.asarray(initial[0][key], dtype=np.float64).view(np.uint64),
                                          np.asarray(final[key], dtype=np.float64).view(np.uint64)), key
                row['seed_primal_dual_bits_preserved'] = True
            if mode=='l1':
                l=final['l1'];assert l['enabled'] and l['valid'] and l['updates']==final['iterations']
                assert l['pairs']==(1470 if capture=='conditioning' else 1631)
                assert l['active_variables']==l['retained_variables']-l['pairs']
                assert l['active_rows']==l['retained_rows']-2*l['pairs']
            row.update(solve_API_calls=1 + len(bootstrap), bootstrap=bootstrap, pre=pre, audit=audit,
                       final={k: v for k, v in final.items() if k not in ('x', 'x_solver', 'y', 'z', 's')})
            save()
            print(json.dumps({'case': name, 'qualified': audit['qualified'], 'iterations': final['iterations'], 'solve_seconds': final['solve_seconds'],
                              'primal': audit['primal'], 'dual': audit['dual'], 'gap': audit['gap'], 'l1': final.get('l1')}), flush=True)
    report['complete'] = True
    report.pop('active_case', None)
    report['solve_API_calls'] = sum(r['solve_API_calls'] for r in report['cases'])
    report['bootstrap_iterations'] = sum(b['iterations'] for r in report['cases'] for b in r['bootstrap'])
    assert report['solve_API_calls'] <= 8 and report['bootstrap_iterations'] <= 2
    save()
except BaseException as error:
    report['failure'] = repr(error)
    save()
    raise
