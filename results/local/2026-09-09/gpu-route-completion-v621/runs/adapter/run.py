"""Single-use foreground supervisor for four frozen production adapter GPU tests."""
from __future__ import annotations

import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import traceback

KIT = Path(__file__).resolve().parent
OUTPUT = KIT / 'gpu-output'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    def reject(value):
        raise ValueError('Nonfinite JSON token: ' + value)
    return json.loads(Path(path).read_text(), parse_constant=reject)


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def main():
    if not __debug__:
        raise RuntimeError('Assertions must remain enabled')
    write(KIT / 'launch-marker.json', {'pid': os.getpid(), 'time': time.time(), 'runner_sha256': sha(__file__)})
    OUTPUT.mkdir(exist_ok=False)
    report = {'complete': False, 'passed': False, 'supervisor_pid': os.getpid(),
              'started': time.time(), 'failures': [], 'processes': [],
              'budget': {'child_processes': 1, 'evaluation_calls': 8, 'candidates': 1048,
                         'actual_lambert_requests': 0, 'timeout_seconds': 180, 'automatic_retries': 0}}
    lock = None
    child = None

    def interrupted(signum, frame):
        raise InterruptedError(f'Supervisor received signal {signum}')

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)

    def reap():
        if child is None or child.poll() is not None:
            return
        os.killpg(child.pid, signal.SIGTERM)
        try:
            child.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait(timeout=30)

    try:
        pins = read(KIT / 'profile.json')
        ready = read(KIT / 'ready.json')
        for name, digest in ready['files'].items():
            assert sha(KIT / name) == digest, name
        assert ready['cpu_collect_only_passed']
        assert sha(pins['host_report_path']) == pins['host_report_sha256']
        host = read(pins['host_report_path'])
        assert host['complete'] and host['gpu_calls'] == 0
        for name, digest in host['source_sha256'].items():
            assert sha(Path(pins['host_source_root']) / name) == digest, name
        tree = ''.join(k + ':' + v + '\n' for k, v in sorted(host['source_sha256'].items()))
        assert hashlib.sha256(tree.encode()).hexdigest() == pins['host_source_tree_sha256']
        assert sha(pins['native_manifest_path']) == pins['native_manifest_sha256']
        native = read(pins['native_manifest_path'])
        assert native['complete'] and native['library'] == pins['library']
        assert sha(pins['library']['path']) == pins['library']['sha256']
        for name, digest in native['source_sha256'].items():
            assert sha(Path(pins['native_source_root']) / name) == digest, name
        prior = read(pins['native_controls_report_path'])
        assert sha(pins['native_controls_report_path']) == pins['native_controls_report_sha256']
        assert prior['complete'] and prior['passed'] and prior['actual_case_evaluations'] == 304
        assert prior['native_kernel_launches'] == 2 and prior['fixture_kernel_launches'] == 1
        assert sha(pins['python']) == pins['python_sha256']
        report['pins'] = pins
        report['ready_sha256'] = sha(KIT / 'ready.json')
        lock = open(pins['gpu_lock'], 'a+')
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        report['lock_acquired'] = time.time()
        inventory = subprocess.check_output([pins['nvidia_smi'],
            '--query-gpu=name,uuid,driver_version', '--format=csv,noheader'], text=True, timeout=15)
        rows = list(csv.reader(inventory.splitlines()))
        assert len(rows) == 1 and rows[0][1].strip() == pins['gpu_uuid']
        assert 'RTX 5090' in rows[0][0]
        processes = subprocess.check_output([pins['nvidia_smi'],
            '--query-compute-apps=pid,process_name,used_gpu_memory', '--format=csv,noheader'],
            text=True, timeout=15)
        assert not processes.strip() or processes.strip() == 'No running processes found'
        report['gpu_inventory'] = inventory
        report['compute_processes_before'] = processes
        environment = {k: v for k, v in os.environ.items()
                       if not k.startswith(('SPACEPDHCG_', 'QOCO_', 'PDHCG_', 'PYTEST_', 'LD_LIBRARY_PATH'))
                       and k not in ('PYTHONPATH', 'PYTHONOPTIMIZE')}
        environment.update({
            'CUDA_VISIBLE_DEVICES': pins['gpu_uuid'], 'PYTHONDONTWRITEBYTECODE': '1',
            'PYTHONOPTIMIZE': '0', 'PYTEST_DISABLE_PLUGIN_AUTOLOAD': '1',
            'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1',
            'SPACEPDHCG_GTOC12_CUDA_LIBRARY': pins['library']['path'],
            'SPACEPDHCG_GTOC12_GPU_TESTS': '1',
            'SPACEPDHCG_COMPLETION_ADAPTER_SUPERVISOR': str(os.getpid()),
            'SPACEPDHCG_COMPLETION_ADAPTER_LOCK_FD': str(lock.fileno()),
            'LD_LIBRARY_PATH': pins['ld_library_path'],
        })
        command = [pins['python'], '-B', str(KIT / 'pytest_child.py'), '--output', str(OUTPUT / 'child')]
        item = {'command': command, 'started': time.time(), 'timeout_seconds': 180}
        report['processes'].append(item)
        write(OUTPUT / 'preflight.json', report)
        with (OUTPUT / 'pytest.log').open('x') as stream:
            child = subprocess.Popen(command, cwd=pins['host_source_root'], env=environment,
                stdout=stream, stderr=subprocess.STDOUT, start_new_session=True, pass_fds=(lock.fileno(),))
            item['pid'] = child.pid
            write(OUTPUT / 'child-launch.json', item)
            print(json.dumps({'supervisor_pid': os.getpid(), 'child_pid': child.pid, 'stage': 'four-adapter-cases'}), flush=True)
            try:
                item['exit_code'] = child.wait(timeout=180)
            except subprocess.TimeoutExpired:
                item['timed_out'] = True
                reap()
                item['exit_code'] = child.returncode
                raise
            finally:
                item['ended'] = time.time()
        assert item['exit_code'] == 0
        result = read(OUTPUT / 'child/report.json')
        assert result['complete'] and result['passed'] and not result['collect_only']
        assert result['actual_evaluation_calls'] == result['actual_completed_evaluation_calls'] == 8
        assert result['actual_candidates'] == 1048 and result['lambert_requests'] == 0
        assert not result['forbidden_gpu_requests']
        report.update(passed=True, actual_evaluation_calls=8, actual_candidates=1048,
                      lambert_requests=0, child_report_sha256=sha(OUTPUT / 'child/report.json'))
    except BaseException as error:
        report['failures'].append({'type': type(error).__name__, 'message': str(error),
                                  'traceback': traceback.format_exc()})
        raise
    finally:
        try:
            reap()
        except BaseException as error:
            report['failures'].append({'cleanup_error': repr(error), 'owned_pid': child.pid if child else None})
        report.update(complete=True, ended=time.time())
        for p in OUTPUT.rglob('*'):
            if p.is_file():
                report.setdefault('output_sha256', {})[p.relative_to(OUTPUT).as_posix()] = sha(p)
        write(OUTPUT / 'report.json', report)
        if lock is not None:
            # The inherited descriptor keeps the GPU locked if an owned child
            # cannot be reaped. Never unlock the shared description explicitly.
            lock.close()
        print(json.dumps({'complete': True, 'passed': report['passed'], 'output': str(OUTPUT)}), flush=True)


if __name__ == '__main__':
    main()
