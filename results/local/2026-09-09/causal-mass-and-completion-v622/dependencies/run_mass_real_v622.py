"""Six finite executions: two original seeds, then unit-L1 versus mass+unit-L1 cold."""
from pathlib import Path
import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import time

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def strict_json(text, preserve_float_bits=False):
    def invalid(value):
        raise ValueError('nonfinite JSON constant: ' + value)
    kwargs = {'parse_constant': invalid}
    if preserve_float_bits:
        kwargs['parse_int'] = float  # Preserve a written -0 in vector exports.
    return json.loads(text, **kwargs)


def records(text, prefix):
    return [strict_json(line[len(prefix) + 1:], True) for line in text.splitlines()
            if line.startswith(prefix + ' ')]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frozen-root', type=Path, required=True)
    parser.add_argument('--manifest-sha256', required=True)
    parser.add_argument('--blocks', type=int, choices=[128], required=True)
    parser.add_argument('--gpu-uuid', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--tiny-report', type=Path, required=True)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    live = Path(__file__).resolve().parents[2]
    frozen = args.frozen_root.resolve()
    manifest_path = frozen / 'manifest.json'
    assert sha(manifest_path) == args.manifest_sha256
    manifest = strict_json(manifest_path.read_text())
    assert manifest['complete'] and manifest['gpu_calls'] == 0
    assert manifest['frozen_commit'] is None and manifest['source_identity_kind'] == 'sha256_tree_not_git_commit'
    assert manifest['compiled_source_commit'] == 'uncommitted'
    tree = ''.join(k + ':' + v + '\n' for k, v in manifest['source_sha256'].items())
    assert hashlib.sha256(tree.encode()).hexdigest() == manifest['source_tree_sha256']
    assert sha(frozen / 'source.tar.gz') == manifest['source_archive_sha256']
    for name, expected in manifest['source_sha256'].items():
        assert sha(frozen / 'repo' / name) == expected, name
    binary = frozen / 'build/cuda-tests/persistent_snapshot_replay'
    core = frozen / 'build/cuda/libspacepdhcg_cuda.so'
    assert sha(binary) == manifest['replay']['sha256']
    assert sha(core) == manifest['library']['sha256']
    tiny_path = args.tiny_report.resolve()
    tiny = strict_json(tiny_path.read_text())
    assert tiny['complete'] and tiny['status'] == 'passed' and tiny['manifest_sha256'] == args.manifest_sha256
    assert tiny['actual_solve_api_calls'] == 7 and tiny['actual_optimization_iterations'] == 5
    assert tiny['core_sha256'] == manifest['library']['sha256'] and tiny['test_sha256'] == manifest['test']['sha256']
    assert tiny['source_tree_sha256'] == manifest['source_tree_sha256']
    assert tiny['runner_sha256'] == '691fa7fb4e3e82de59d555ad8e2f51bc2d24600194e91cccaed60dab470b8a84'
    assert tiny['gpu_uuid_required'] == args.gpu_uuid
    assert len(tiny['cases']) == 1 and sha(tiny_path.parent / 'tiny.log') == tiny['cases'][0]['log_sha256']
    source_meta = records((frozen / 'validate-conditioning-mass.log').read_text(), 'PERSISTENT_REPLAY_META')
    assert len(source_meta) == 1 and source_meta[0]['source_tree_sha256'] == manifest['source_tree_sha256']
    compiled_source_id = source_meta[0]['source_sha256']

    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    out = root / 'run'
    out.mkdir()
    inputs = root / 'inputs'
    inputs.mkdir()
    for source, dest in [(manifest_path, 'manifest.json'), (Path(__file__), 'run.py'),
                         (tiny_path, 'tiny-report.json'),
                         (live / 'scripts/gpu/audit_persistent_snapshot.py', 'independent_auditor.py')]:
        shutil.copyfile(source, root / dest)
    assert sha(root / 'independent_auditor.py') == '0d944f768ad9c089499cc6e95cd1313677e153f650eced2c7d84b9815c00c7ff'
    spec = importlib.util.spec_from_file_location('audit_mass_v622', root / 'independent_auditor.py')
    auditor = importlib.util.module_from_spec(spec)
    exec(compile((root / 'independent_auditor.py').read_bytes(), str(root / 'independent_auditor.py'), 'exec'), auditor.__dict__)
    expected_inputs = {
        'conditioning.txt': '1eb1b5a3979c5e86522337dfa94f264e354993163ed44129eff361371e3b49cf',
        'difficult.txt': '14fdc84e9b654df4ac16db71f0bce4fccfa62577f9620ae622ad2196e2f17080',
        'conditioning-initial.txt': '9caf303469dcd254d56b74f8bf3818c8044f51a3a20bc3c07835b76dd23c5632',
        'difficult-initial.txt': 'b92be6c5a0051ab99c000c459ef466c69fc8b2fd669c38b3580aea000d926da6',
    }
    for name, expected in expected_inputs.items():
        shutil.copyfile(live / 'build/performance/known-point-replay-v606/inputs' / name, inputs / name)
        assert sha(inputs / name) == expected, name
    snapshots = {name: auditor.load_snapshot(inputs / (name + '.txt')) for name in ('conditioning', 'difficult')}
    plan = [(capture, 'mass', True) for capture in ('conditioning', 'difficult')]
    plan += [(capture, mode, False) for capture in ('conditioning', 'difficult') for mode in ('unit', 'mass')]
    report = {
        'complete': False, 'pid': os.getpid(), 'maximum_executions': 6, 'maximum_solve_API_calls': 8,
        'maximum_bootstrap_iterations': 2, 'cold_iteration_cap': 100000, 'cold_deadline_seconds': 30,
        'execution_blocks': args.blocks,
        'capacity_policy': 'Native setter validates original and new solve/init occupancy; abort if 128 unsupported, no fallback or grid sweep.',
        'binary_sha256': sha(binary), 'core_sha256': sha(core), 'tiny_report_sha256': sha(tiny_path),
        'manifest_sha256': args.manifest_sha256, 'runner_sha256': sha(root / 'run.py'),
        'source_tree_sha256': manifest['source_tree_sha256'], 'source_archive_sha256': manifest['source_archive_sha256'],
        'source_commit': None, 'source_commit_scope': 'uncommitted frozen C++ source tree, not a Git commit',
        'base_commit': manifest['base_commit'], 'compiled_snapshot_source_sha256': compiled_source_id,
        'auditor_sha256': sha(root / 'independent_auditor.py'), 'inputs_sha256': expected_inputs,
        'gpu_uuid_required': args.gpu_uuid, 'executions_started': 0, 'cases': [],
        'lock_policy': 'LOCK_EX|LOCK_NB, inherited by native child',
        'scope': 'Same-build unit L1 versus causal mass elimination plus a direct original-coordinate diagonal metric. The reduction and metric change together. Full original x/y/z/s and actual mass steps are retained. All six final candidates receive the unchanged original common KKT audit regardless of termination. No tolerance weakening, extra weight arm or convergence claim.',
        'bootstrap_scope': 'Two seeded executions each contain a separately counted at-most-one-update bootstrap. Bootstrap original vectors are not exported; only final seeded/cold vectors receive the independent audit.',
    }

    def save():
        (out / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')

    save()
    if args.prepare_only:
        report['status'] = 'prepared_only_zero_GPU_calls'
        report['plan'] = plan
        save()
        print(json.dumps({'prepared': True, 'gpu_calls': 0, 'report': str(out / 'report.json')}), flush=True)
        return 0
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(('SPACEPDHCG_', 'QOCO_', 'PDHCG_', 'CUDA_VISIBLE_DEVICES', 'LD_LIBRARY_PATH'))}
    env.update(CUDA_VISIBLE_DEVICES=args.gpu_uuid, LD_LIBRARY_PATH=str(core.parent) + ':/usr/local/cuda-12.8/lib64',
               PYTHONDONTWRITEBYTECODE='1')
    try:
        with Path('/home/angus/.spacepdhcg-gpu.lock').open('a+') as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                report['status'] = 'lock_busy_zero_GPU_calls'
                save()
                return 75
            report['compute_processes_before'] = subprocess.check_output(
                ['/usr/lib/wsl/lib/nvidia-smi', '--query-compute-apps=pid,process_name', '--format=csv,noheader'], text=True, timeout=15)
            report['gpu_identity'] = subprocess.check_output(
                ['/usr/lib/wsl/lib/nvidia-smi', '--query-gpu=name,uuid,driver_version', '--format=csv,noheader'], text=True, timeout=15)
            save()
            assert not report['compute_processes_before'].strip(), 'GPU busy; no launch'
            gpu_lines = [line for line in report['gpu_identity'].splitlines() if line.strip()]
            assert len(gpu_lines) == 1 and gpu_lines[0].split(',')[1].strip() == args.gpu_uuid
            for capture, mode, seeded in plan:
                name = f"{capture}-{'seeded' if seeded else 'cold'}-{mode}"
                snapshot = inputs / (capture + '.txt')
                command = [str(binary), str(snapshot), '--tolerance', '1e-9', '--iterations', '1' if seeded else '100000',
                           '--deadline-seconds', '20' if seeded else '30', '--mode', 'cold', '--execution-blocks', str(args.blocks),
                           '--common-kkt-stop', '--l1-prox']
                if mode == 'mass':
                    command += ['--mass-eliminate']
                if seeded:
                    command += ['--initial-point', str(inputs / (capture + '-initial.txt'))]
                row = {'name': name, 'capture': capture, 'mode': mode, 'seeded': seeded, 'command': command}
                report['cases'].append(row)
                report['executions_started'] += 1
                report['active_case'] = name
                save()
                start = time.perf_counter()
                log_path = out / (name + '.log')
                try:
                    with log_path.open('x') as log:
                        result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                                timeout=65 if seeded else 50, pass_fds=(lock.fileno(),))
                    row['returncode'] = result.returncode
                except subprocess.TimeoutExpired:
                    row.update(returncode=None, external_timeout=True)
                finally:
                    row['wall_seconds'] = time.perf_counter() - start
                    row['log_sha256'] = sha(log_path)
                    save()
                text = log_path.read_text()
                # Parse every emitted JSON record, including failed final attempts.
                for line in text.splitlines():
                    if line.startswith('PERSISTENT_'):
                        strict_json(line.split(' ', 1)[1], True)
                assert row['returncode'] == 0, row
                meta, final = records(text, 'PERSISTENT_REPLAY_META'), records(text, 'PERSISTENT_REPLAY')
                assert len(meta) == len(final) == 1
                meta, final = meta[0], final[0]
                bootstrap = records(text, 'PERSISTENT_REPLAY_BOOTSTRAP')
                initial = records(text, 'PERSISTENT_REPLAY_INITIAL_POINT')
                pre = records(text, 'PERSISTENT_REPLAY_PRESTEP')
                assert len(bootstrap) == len(initial) == len(pre) == int(seeded)
                assert meta['input_sha256'] == sha(snapshot) and meta['library_sha256'] == sha(core)
                assert meta['requested_execution_blocks'] == final['execution_blocks'] == args.blocks
                assert meta['halpern_mode'] == 'off' and meta['l1_prox'] and meta['l1_weight_policy'] == 'unit_default'
                assert meta['mass_eliminate'] == (mode == 'mass')
                assert meta['source_commit'] == 'uncommitted' and meta['source_commit_scope'] == 'uncommitted_frozen_source_tree'
                assert meta['source_dirty'] is True and meta['source_tree_sha256'] == manifest['source_tree_sha256']
                assert meta['base_commit'] == manifest['base_commit'] and meta['source_sha256'] == compiled_source_id
                audit = auditor.audit(snapshots[capture], final, backend='persistent', coordinates='original')
                row['audit'] = audit
                audit_path = out / (name + '-audit.json')
                audit_path.write_text(json.dumps({'log_sha256': row['log_sha256'], 'input_sha256': sha(snapshot),
                                                 'auditor_sha256': report['auditor_sha256'], 'audit': audit}, indent=2, allow_nan=False) + '\n')
                row['audit_sha256'] = sha(audit_path)
                save()
                assert audit['qualified'] == final['qualified_original']
                gpu = final['gpu_common_kkt']
                assert gpu['enabled']
                if gpu['valid']:
                    assert gpu['passes'] == audit['passes_common_kkt_gate']
                else:
                    assert not gpu['passes'] and final['termination'] in (3, 4)
                assert final['recovery_iterations'] == 0 and final['recovery_seconds'] == 0
                l1 = final['l1']
                assert l1['enabled'] and l1['updates'] == final['iterations'] and l1['weight_mode'] == 0
                pairs, nodes = (1470, 211) if capture == 'conditioning' else (1631, 234)
                assert l1['pairs'] == pairs
                removed_nodes = nodes if mode == 'mass' else 0
                assert l1['active_variables'] == l1['retained_variables'] - pairs - removed_nodes
                assert l1['active_rows'] == l1['retained_rows'] - 2 * pairs - removed_nodes
                if mode == 'mass':
                    mass = final['mass']
                    assert mass['enabled'] and mass['nodes'] == nodes
                    assert mass['updates'] == mass['completions'] == final['iterations']
                    assert len(meta['mass_map']) == nodes
                    steps = np.asarray(mass['steps_original_layout'], dtype=np.float64)
                    assert len(steps) == mass['retained_variables'] + mass['retained_rows']
                    if mass['valid'] and mass['finite']:
                        assert np.isfinite(steps).all() and (steps > 0).all()
                        assert 0 < mass['norm_squared_upper'] < 1
                        assert mass['theta'] == 0.95 and l1['eta'] == l1['bound_scale'] == l1['objective_scale'] == 1
                    else:
                        assert final['termination'] in (3, 4)
                if seeded:
                    assert audit['qualified'] and final['termination'] == 1 and final['iterations'] == 0
                    assert l1['updates'] == l1['completions'] == 0
                    assert initial[0]['supplied_qualified'] and initial[0]['mapped_reference_qualified']
                    assert meta['initial_point_sha256'] == sha(inputs / (capture + '-initial.txt'))
                    assert initial[0]['point_sha256'] == meta['initial_point_sha256']
                    assert pre[0]['native_seed_verified_unchanged'] and pre[0]['seeded_iterations'] == 0
                    assert bootstrap[0]['iterations'] <= 1
                    for key in ('x', 'y', 'z'):
                        assert np.array_equal(np.asarray(initial[0][key], dtype=np.float64).view(np.uint64),
                                              np.asarray(final[key], dtype=np.float64).view(np.uint64)), key
                    row['seed_primal_dual_bits_preserved'] = True
                clean_final = {k: v for k, v in final.items() if k not in ('x', 'x_solver', 'y', 'z', 's')}
                if 'mass' in clean_final:
                    clean_final['mass'] = {k: v for k, v in final['mass'].items() if k != 'steps_original_layout'}
                row.update(solve_API_calls=1 + len(bootstrap), bootstrap=bootstrap, pre=pre, final=clean_final)
                save()
                print(json.dumps({'case': name, 'qualified': audit['qualified'], 'iterations': final['iterations'],
                                  'solve_seconds': final['solve_seconds'], 'primal': audit['primal'],
                                  'dual': audit['dual'], 'gap': audit['gap']}), flush=True)
            report['lock_scope_ending'] = True
        report['complete'] = True
        report['status'] = 'passed'
        report.pop('active_case', None)
        report['solve_API_calls'] = sum(row['solve_API_calls'] for row in report['cases'])
        report['bootstrap_iterations'] = sum(b['iterations'] for row in report['cases'] for b in row['bootstrap'])
        assert report['executions_started'] == 6 and report['solve_API_calls'] == 8 and report['bootstrap_iterations'] <= 2
        save()
    except BaseException as error:
        report['status'] = 'failed'
        report['failure'] = repr(error)
        save()
        raise
    print(json.dumps({'complete': True, 'report': str(out / 'report.json'), 'sha256': sha(out / 'report.json')}), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
