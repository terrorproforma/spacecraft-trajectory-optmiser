"""One reviewed native correctness process. Prepare only; root controls GPU launch."""
from pathlib import Path
import argparse
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import time


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def strict_json(value):
    def bad(token):
        raise ValueError('nonfinite JSON token ' + token)
    return json.loads(value, parse_constant=bad)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--build-root', type=Path, required=True)
    parser.add_argument('--manifest-sha256', required=True)
    parser.add_argument('--gpu-uuid', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.build_root.resolve()
    manifest_path = root / 'manifest.json'
    assert digest(manifest_path) == args.manifest_sha256, 'manifest identity'
    manifest = strict_json(manifest_path.read_text())
    assert manifest['complete'] and manifest['gpu_calls'] == 0
    assert manifest['source_identity_kind'] == 'sha256_tree_not_git_commit'
    assert manifest['frozen_commit'] is None
    for name, sha in manifest['source_sha256'].items():
        assert digest(root / 'repo' / name) == sha, name
    tree = ''.join(name + ':' + sha + '\n' for name, sha in manifest['source_sha256'].items())
    assert hashlib.sha256(tree.encode()).hexdigest() == manifest['source_tree_sha256']
    assert digest(root / 'source.tar.gz') == manifest['source_archive_sha256']
    binary = root / 'build/cuda-tests/persistent_mass_test'
    core = root / 'build/cuda/libspacepdhcg_cuda.so'
    assert digest(binary) == manifest['test']['sha256']
    assert digest(core) == manifest['library']['sha256']
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, output / Path(__file__).name)
    shutil.copy2(manifest_path, output / 'build-manifest.json')
    report = {
        'complete': False, 'gpu_executions_started': 0, 'maximum_executions': 1,
        'maximum_solve_api_calls': 7, 'maximum_requested_iterations': 9,
        'expected_optimization_iterations': 5,
        'call_scope': 'One native process; seven solve APIs. Setup/proof/residual kernels are additional GPU kernels, not additional solves.',
        'manifest_sha256': args.manifest_sha256,
        'source_tree_sha256': manifest['source_tree_sha256'],
        'base_commit': manifest['base_commit'], 'source_commit': None,
        'source_commit_scope': 'uncommitted frozen C++ source tree, not a Git commit',
        'source_archive_sha256': manifest['source_archive_sha256'],
        'core_sha256': digest(core), 'test_sha256': digest(binary),
        'runner_sha256': digest(__file__), 'gpu_uuid_required': args.gpu_uuid,
        'lock_path': '/home/angus/.spacepdhcg-gpu.lock',
        'lock_policy': 'LOCK_EX|LOCK_NB', 'process_timeout_seconds': 180,
        'cases': [],
    }

    def save():
        (output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')

    save()
    start = time.perf_counter()
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(('SPACEPDHCG_', 'PDHCG_', 'QOCO_', 'LD_LIBRARY_PATH'))}
    env['CUDA_VISIBLE_DEVICES'] = args.gpu_uuid
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    with open(report['lock_path'], 'a+') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            report['status'] = 'lock_busy_zero_GPU_calls'
            save()
            return 75
        report['lock_acquired'] = True
        report['pid'] = os.getpid()
        save()
        print(json.dumps({'status': 'running', 'pid': os.getpid(), 'output': str(output)}), flush=True)
        try:
            smi = '/usr/lib/wsl/lib/nvidia-smi'
            for label, query in [('gpu', '--query-gpu=name,uuid,driver_version'),
                                 ('compute-processes-before', '--query-compute-apps=pid,process_name,used_memory')]:
                result = subprocess.run([smi, query, '--format=csv,noheader'],
                                        capture_output=True, text=True, timeout=15)
                (output / (label + '.txt')).write_text(result.stdout + result.stderr)
                assert result.returncode == 0, label
                if label == 'gpu':
                    lines = [line for line in result.stdout.splitlines() if line.strip()]
                    assert len(lines) == 1 and lines[0].split(',')[1].strip() == args.gpu_uuid, 'GPU identity'
                else:
                    assert not result.stdout.strip(), 'another compute process is active'
            command = [str(binary)]
            entry = {'command': command, 'started': True, 'records': []}
            report['cases'].append(entry)
            report['gpu_executions_started'] = 1
            report['environment'] = {'CUDA_VISIBLE_DEVICES': env['CUDA_VISIBLE_DEVICES'], 'solver_overrides_removed': True}
            save()
            begin = time.perf_counter()
            path = output / 'tiny.log'
            try:
                with path.open('x') as log:
                    result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                            timeout=180, pass_fds=(lock.fileno(),))
            finally:
                entry['wall_seconds'] = time.perf_counter() - begin
                if path.exists():
                    entry['log_sha256'] = digest(path)
                save()
            entry['returncode'] = result.returncode
            for line in path.read_text().splitlines():
                if line.startswith('MASS_'):
                    prefix, value = line.split(' ', 1)
                    entry['records'].append({'prefix': prefix, 'record': strict_json(value)})
            save()
            assert result.returncode == 0, path.read_text()[-6000:]
            records = entry['records']
            assert [x['prefix'] for x in records] == ['MASS_CPU'] + ['MASS_TEST'] * 7 + ['MASS_VALIDATION', 'MASS_TEST_SUMMARY']
            assert records[0]['record'] == {'passed': True, 'nodes': 4, 'pairs': 3, 'GPU_calls': 0}
            cases = [x['record'] for x in records if x['prefix'] == 'MASS_TEST']
            expected = [('three_interval_oracle', 2, 3), ('disabled_to_L1', 2, 1),
                        ('fresh_L1_control', 2, 1), ('original_seed_zero_step', 1, 0),
                        ('cancel_before_initial', 3, 0), ('nonfinite_seed', 4, 0), ('metric_overflow', 4, 0)]
            assert [(x['case'], x['termination'], x['iterations']) for x in cases] == expected
            assert cases[0]['mass_valid'] and cases[0]['finite'] and cases[0]['norm_squared_upper'] < 1
            assert not cases[1]['mass_enabled'] and not cases[2]['mass_enabled']
            assert cases[3]['common_valid'] and cases[3]['common_passes'] and cases[3]['completions'] == 0
            assert not cases[4]['common_valid'] and not cases[4]['mass_valid']
            assert not cases[5]['finite'] and not cases[6]['finite']
            assert records[-2]['record'] == {'mass_cost': True, 'extra_equality': True, 'extra_SOC': True, 'negative_gamma': True, 'overlap': True}
            assert records[-1]['record'] == {'passed': True, 'solve_API_calls': 7, 'iteration_caps': 9, 'actual_updates': 5}
            assert sum(x['iterations'] for x in cases) == 5
            entry['strict_json_parsed'] = True
            report['actual_solve_api_calls'] = 7
            report['actual_optimization_iterations'] = 5
            report['status'] = 'passed'
            report['complete'] = True
        except BaseException as exc:
            report['status'] = 'failed'
            report['error'] = repr(exc)
            raise
        finally:
            report['wall_seconds'] = time.perf_counter() - start
            report['lock_scope_ending'] = True
            save()
    print(json.dumps({'status': report['status'], 'report': str(output / 'report.json'),
                      'sha256': digest(output / 'report.json')}), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
