"""Finite integration checks of the combined core; no performance experiment."""

from pathlib import Path
import fcntl
import hashlib
import json
import os
import signal
import subprocess
import sys
import tarfile
import time
import traceback
import xml.etree.ElementTree as ET

LIVE = Path('/mnt/c/Users/Angus/Desktop/projects/spacecraft-trajectory-optmiser')
ROOT = Path('/home/angus/spacepdhcg-combined-v622a')
HOST = Path('/home/angus/spacepdhcg-completion-model-v622g')
KIT = LIVE / 'build/performance/completion-pack-benchmark-v622'
OUT = LIVE / 'build/performance/combined-smoke-v622a'
MANIFEST_SHA = '9bff6c54af1edca6323502908d440305b4edc8447550572a5d93b50a421e3730'
HOST_SHA = '56a65032e639a6eb9af08052e9f3df87f5f1c84cfe9cf2ece6f3f63eafb2a12a'
HOST_FULL_SHA = 'f38ec9f963a4ea75982bd71a5bbd59909ae318e12f30bf796738dcbff45d803f'
HOST_ARCHIVE_SHA = '7281c43ff8a4d30237f84d31ae7435ee36ff90eed711f1e718f5af0732ea532e'
UUID = 'GPU-4df2f6b5-e866-14a0-eeac-332cb2b757d4'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def child_compact():
    assert os.environ.get('SPACEPDHCG_COMBINED_SMOKE_SUPERVISED')
    sys.meta_path = [f for f in sys.meta_path
                     if f.__class__.__module__ != '_editable_skbc_spacepdhcg']
    sys.path.insert(0, str(HOST / 'source/src'))
    import spacepdhcg.gtoc12.gpu_completion as completion
    assert Path(completion.__file__).resolve() == HOST / 'source/src/spacepdhcg/gtoc12/gpu_completion.py'
    import pytest
    return pytest.main([str(HOST / 'source/tests/test_gtoc12_gpu_completion_model.py'),
                        '-q', '-p', 'no:cacheprovider', '--junitxml=' + str(OUT / 'compact.xml')])


def main():
    assert __debug__, 'Assertions must be enabled'
    assert sha(ROOT / 'manifest.json') == MANIFEST_SHA
    assert sha(HOST / 'report.json') == HOST_SHA
    manifest, host = read(ROOT / 'manifest.json'), read(HOST / 'report.json')
    assert manifest['complete'] and manifest['gpu_calls'] == 0
    assert host['complete'] and host['gpu_calls'] == 0
    assert sha(ROOT / 'source.tar.gz') == manifest['source_archive_sha256']
    assert sha(HOST / 'source.tar.gz') == host['source_archive_sha256']
    for name, digest in manifest['source_sha256'].items():
        assert sha(ROOT / 'repo' / name) == digest, name
    for name, digest in host['owned_sources'].items():
        assert sha(HOST / 'source' / name) == digest, name
    # Compact-g's archive contains only overlays. Bind the full host snapshot
    # from the completed benchmark, including ordinary import dependencies.
    assert sha(KIT / 'source-sha256.json') == HOST_FULL_SHA
    assert sha(KIT / 'source-full.tar.gz') == HOST_ARCHIVE_SHA
    host_sources = read(KIT / 'source-sha256.json')
    assert len(host_sources) == 341
    for name, digest in host_sources.items():
        assert sha(HOST / 'source' / name) == digest, name
    seen = set()
    with tarfile.open(KIT / 'source-full.tar.gz') as archive:
        for member in archive:
            if member.isfile():
                path = Path(member.name)
                assert not path.is_absolute() and '..' not in path.parts
                assert member.name in host_sources and member.name not in seen
                seen.add(member.name)
                assert (HOST / 'source' / path).read_bytes() == archive.extractfile(member).read(), member.name
    assert seen == set(host_sources)
    for key in ('library', 'test', 'completion_test'):
        assert sha(manifest[key]['path']) == manifest[key]['sha256'], key
    OUT.mkdir(exist_ok=False)
    (OUT / 'readbacks').mkdir()
    (OUT / 'run.py').write_bytes(Path(__file__).read_bytes())
    (OUT / 'build-manifest.json').write_bytes((ROOT / 'manifest.json').read_bytes())
    (OUT / 'host-report.json').write_bytes((HOST / 'report.json').read_bytes())
    (OUT / 'host-source-sha256.json').write_bytes((KIT / 'source-sha256.json').read_bytes())
    report = {
        'complete': False, 'passed': False, 'supervisor_pid': os.getpid(),
        'runner_sha256': sha(__file__), 'manifest_sha256': MANIFEST_SHA,
        'host_report_sha256': HOST_SHA, 'core_sha256': manifest['library']['sha256'],
        'host_full_source_sha256': HOST_FULL_SHA, 'host_full_archive_sha256': HOST_ARCHIVE_SHA,
        'gpu_uuid': UUID, 'maximum_processes': 3, 'per_process_timeout_seconds': 90,
        'mass_solve_API_budget': 7, 'mass_iteration_caps': 9, 'mass_expected_updates': 5,
        'legacy_completion_candidate_budget': 262, 'legacy_nonempty_evaluations': 2,
        'compact_valid_call_budget': 18, 'compact_candidate_budget': 2358,
        'compact_malformed_call_budget': 6, 'fresh_lambert_solves': 0,
        'fresh_refinements': 0, 'fleet_promotions': 0,
        'scope': 'Full combined-core correctness only; no cold convergence rerun or performance claim.',
        'cases': [], 'failures': [],
    }

    def save():
        (OUT / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')

    child = None

    def reap():
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait(timeout=10)

    def interrupted(signum, frame):
        raise InterruptedError(f'Signal {signum}')

    signal.signal(signal.SIGINT, interrupted)
    signal.signal(signal.SIGTERM, interrupted)
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(('SPACEPDHCG_', 'QOCO_', 'PDHCG_', 'LD_LIBRARY_PATH'))}
    env.update(CUDA_VISIBLE_DEVICES=UUID, PYTHONDONTWRITEBYTECODE='1', PYTHONOPTIMIZE='0',
               OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
               SPACEPDHCG_COMBINED_SMOKE_SUPERVISED=str(os.getpid()),
               SPACEPDHCG_GTOC12_GPU_TESTS='1',
               SPACEPDHCG_COMPLETION_TEST_LIBRARY=manifest['library']['path'],
               SPACEPDHCG_COMPLETION_TEST_READBACKS=str(OUT / 'readbacks'))
    save()
    with open('/home/angus/.spacepdhcg-gpu.lock', 'a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            report['lock_acquired'] = True
            smi = '/usr/lib/wsl/lib/nvidia-smi'
            def query(flag):
                return subprocess.check_output([smi, flag, '--format=csv,noheader'], text=True, timeout=15).strip()
            report['gpu_inventory'] = query('--query-gpu=uuid,name,driver_version')
            report['compute_before'] = query('--query-compute-apps=pid,process_name')
            assert UUID in report['gpu_inventory'] and not report['compute_before']
            commands = [
                ('mass', [manifest['test']['path']]),
                ('legacy', [manifest['completion_test']['path']]),
                ('compact', [sys.executable, '-B', str(OUT / 'run.py'), '--child-compact']),
            ]
            for label, command in commands:
                entry = {'name': label, 'command': command}
                report['cases'].append(entry)
                started = time.perf_counter()
                try:
                    with (OUT / (label + '.log')).open('x') as log:
                        child = subprocess.Popen(command, cwd=HOST / 'source', env=env, stdout=log,
                                                 stderr=subprocess.STDOUT, start_new_session=True,
                                                 pass_fds=(lock.fileno(),))
                        entry['pid'] = child.pid
                        save()
                        print(json.dumps({'case': label, 'pid': child.pid}), flush=True)
                        entry['exit_code'] = child.wait(timeout=90)
                finally:
                    reap()
                    entry['wall_seconds'] = time.perf_counter() - started
                    entry['log_sha256'] = sha(OUT / (label + '.log'))
                    save()
                assert entry['exit_code'] == 0, label
                lines = (OUT / (label + '.log')).read_text().splitlines()
                if label == 'mass':
                    records = [(line.split(' ', 1)[0], json.loads(line.split(' ', 1)[1]))
                               for line in lines if line.startswith('MASS_')]
                    entry['records'] = records
                    assert records[-1] == ('MASS_TEST_SUMMARY', {
                        'passed': True, 'solve_API_calls': 7, 'iteration_caps': 9, 'actual_updates': 5})
                    assert len([r for key, r in records if key == 'MASS_TEST']) == 7
                elif label == 'legacy':
                    records = [json.loads(line.split(' ', 1)[1])
                               for line in lines if line.startswith('COMPLETION_TEST ')]
                    entry['records'] = records
                    assert records[-1] == {'phase': 'native', 'kernel_launches': 2,
                                           'candidates': 262, 'passed': True}
                else:
                    suites = list(ET.parse(OUT / 'compact.xml').getroot().iter('testsuite'))
                    entry['junit'] = {key: sum(int(s.attrib.get(key, 0)) for s in suites)
                                      for key in ('tests', 'failures', 'errors', 'skipped')}
                    assert entry['junit'] == {'tests': 7, 'failures': 0, 'errors': 0, 'skipped': 0}
                    assert {p.name for p in (OUT / 'readbacks').iterdir()} == {
                        f'{model}-{size}.npz' for model in ('fit', 'flat', 'ratio') for size in (259, 3)}
                save()
            report['compute_after'] = query('--query-compute-apps=pid,process_name')
            report['passed'] = True
        except BaseException as error:
            report['failures'].append({'error': repr(error), 'traceback': traceback.format_exc()})
            raise
        finally:
            reap()
            report['readbacks'] = {p.name: {'sha256': sha(p), 'bytes': p.stat().st_size}
                                   for p in sorted((OUT / 'readbacks').iterdir())}
            report['complete'] = True
            save()
    print(json.dumps({'passed': report['passed'], 'report_sha256': sha(OUT / 'report.json')}))


if __name__ == '__main__':
    if sys.argv[1:] == ['--child-compact']:
        raise SystemExit(child_compact())
    assert len(sys.argv) == 1
    main()
