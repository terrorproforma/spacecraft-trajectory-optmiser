"""Build and validate the frozen candidate, serializing every GPU operation."""
from datetime import datetime, timezone
from pathlib import Path
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
import xml.etree.ElementTree as ET

ROOT = Path('/home/angus/build-spacepdhcg-joint-v596')
SOURCE = ROOT / 'source'
BUILD = ROOT / 'build'
EVIDENCE = Path('/mnt/c/Users/Angus/Desktop/projects/spacecraft-trajectory-optmiser/build/performance/joint-local-v596/evidence-complete-fixture')
PYTHON = '/home/angus/worktrees/spacepdhcg-literature-venv/bin/python'
CMAKE = '/home/angus/spacecraft-trajectory-optmiser/.venv/bin/cmake'
NVCC = '/usr/local/cuda-12.8/bin/nvcc'
UPSTREAM = '/home/angus/spacecraft-trajectory-optmiser/_upstream/pdhcg'
DATA = '/home/angus/worktrees/spacepdhcg-release/benchmarks/gtoc12/data'
CORE = BUILD / 'cuda/libspacepdhcg_cuda.so'
EVIDENCE.mkdir(exist_ok=False)
REPORT = {'schema': 'joint-local-v596', 'started_utc': datetime.now(timezone.utc).isoformat(),
          'pid': os.getpid(), 'complete': False, 'success': False, 'commands': [],
          'claim': 'Search-surrogate parity and timing only; no new trajectory or fleet certification.'}
ENV = {k: v for k, v in os.environ.items() if not k.startswith('SPACEPDHCG_')}
ENV.update(SPACEPDHCG_GTOC12_GPU_TESTS='1', SPACEPDHCG_GTOC12_CUDA_LIBRARY=str(CORE),
           SPACEPDHCG_GTOC12_DATA=DATA, SPACEPDHCG_GTOC12_JOINT_ARCHIVE=str(SOURCE / 'incumbent-sources'),
           SPACEPDHCG_TEST_GTOC12_JOINT_DEVICE_SELECTION='1',
           OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
           PYTHONHASHSEED='0', PYTHONPATH=str(SOURCE / 'src'),
           PATH='/usr/local/cuda/bin:' + os.environ.get('PATH', ''))
ENV.pop('PYTEST_ADDOPTS', None)
REPORT['environment'] = {k: v for k, v in ENV.items() if k.startswith('SPACEPDHCG_') or k in
                         ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'PYTHONHASHSEED', 'PYTHONPATH')}
REPORT['native_smoke_scope'] = 'Original full-evaluation API; does not call the new best_host selection API.'
REPORT['python_sanitizer_scope'] = 'Fresh frozen 42-test parity suite, including best_host winner selection, under all three sanitizers.'
REPORT['prior_local_results_apply'] = False
REPORT['prior_local_note'] = 'This snapshot adds native device winner selection after the earlier 42-test local run and benchmark; it requires its own fresh build and checks.'


def stamp():
    return datetime.now(timezone.utc).isoformat()


def status(phase, **extras):
    value = {'pid': os.getpid(), 'phase': phase, 'updated_utc': stamp(), **extras}
    target = EVIDENCE.parent / 'resume-status.json'
    temp = EVIDENCE.parent / 'resume-status.tmp.json'
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(target)
    (EVIDENCE / 'report.json').write_text(json.dumps(REPORT, indent=2) + '\n')
    print(json.dumps(value), flush=True)


def run(name, command, *, environment=ENV):
    begin = time.perf_counter()
    record = {'name': name, 'command': command, 'started_utc': stamp()}
    REPORT['commands'].append(record)
    with (EVIDENCE / (name + '.log')).open('w') as log:
        process = subprocess.Popen(command, cwd=SOURCE, env=environment,
                                   stdout=log, stderr=subprocess.STDOUT, text=True)
        record['pid'] = process.pid
        status(name, child_pid=process.pid)
        record['returncode'] = process.wait()
    record['seconds'] = time.perf_counter() - begin
    record['finished_utc'] = stamp()
    status(name + '-complete', child_pid=process.pid, returncode=record['returncode'])
    if record['returncode']:
        raise RuntimeError(f'{name} returned {record["returncode"]}')


def parity_command(junit):
    code = ("import sys; from pathlib import Path; "
            "sys.meta_path=[f for f in sys.meta_path if f.__class__.__module__!='_editable_skbc_spacepdhcg']; "
            "sys.path.insert(0,str(Path('src').resolve())); import pytest; "
            "raise SystemExit(pytest.main(['-q','tests/test_gtoc12_gpu_joint.py',"
            "'tests/test_gtoc12_jointopt.py','--junitxml'," + repr(str(junit)) + "]))")
    return [PYTHON, '-c', code]


def check_parity(name, junit):
    suites = ET.parse(junit).getroot().findall('testsuite')
    totals = {key: sum(int(s.get(key, '0')) for s in suites) for key in ('tests', 'failures', 'errors', 'skipped')}
    REPORT.setdefault('pytest_totals', {})[name] = totals
    assert totals == {'tests': 42, 'failures': 0, 'errors': 0, 'skipped': 0}, (name, totals)


try:
    original = json.loads((EVIDENCE.parent / 'evidence/report.json').read_text())
    manifest_path = EVIDENCE.parent / 'resume-source-manifest.json'
    manifest = json.loads(manifest_path.read_text())
    for name, record in manifest['files'].items():
        assert hashlib.sha256((SOURCE / name).read_bytes()).hexdigest() == record['sha256'], name
    REPORT['source_manifest_sha256'] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    REPORT['base_commit'] = manifest['base_commit']
    REPORT['isolated_snapshot_commit'] = original['isolated_snapshot_commit']
    REPORT['snapshot_addendum'] = 'Only the omitted historical test fixture was added after the isolated snapshot commit; all native/Python/test source bytes are unchanged.'
    REPORT['core'] = original['core']
    REPORT['native_checks_record'] = str(EVIDENCE.parent / 'evidence/report.json')
    REPORT['native_checks_record_sha256'] = hashlib.sha256((EVIDENCE.parent / 'evidence/report.json').read_bytes()).hexdigest()
    assert hashlib.sha256(CORE.read_bytes()).hexdigest() == REPORT['core']['sha256']
    with Path('/home/angus/.spacepdhcg-gpu.lock').open('a') as lock:
        status('waiting-for-gpu-lock')
        fcntl.flock(lock, fcntl.LOCK_EX)
        REPORT['gpu_lock_acquired_utc'] = stamp()
        while True:
            gpu_processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,process_name', '--format=csv,noheader'], text=True).strip()
            if not gpu_processes:
                break
            status('waiting-for-idle-gpu', processes=gpu_processes)
            time.sleep(15)
        run('gpu-before', ['nvidia-smi', '-q'])
        for suite, expected in [('python-parity', 42), ('python-selection', 8)]:
            for checker in (None, 'memcheck', 'synccheck', 'racecheck'):
                name = suite if checker is None else suite + '-' + checker
                junit = EVIDENCE / (name + '.xml')
                command = parity_command(junit)
                if suite == 'python-selection':
                    command[-1] = command[-1].replace(
                        "'tests/test_gtoc12_gpu_joint.py','tests/test_gtoc12_jointopt.py'",
                        "'tests/test_gtoc12_gpu_joint_selection.py'")
                if checker:
                    command = ['/usr/local/cuda-12.8/bin/compute-sanitizer', '--tool', checker,
                               '--target-processes', 'all', '--error-exitcode', '99', *command]
                run(name, command)
                suites = ET.parse(junit).getroot().findall('testsuite')
                totals = {key: sum(int(s.get(key, '0')) for s in suites)
                          for key in ('tests', 'failures', 'errors', 'skipped')}
                REPORT.setdefault('pytest_totals', {})[name] = totals
                assert totals == {'tests': expected, 'failures': 0, 'errors': 0, 'skipped': 0}, (name, totals)
        benchmark = SOURCE / 'build/performance/joint-benchmark-v593/benchmark_joint.py'
        argv = [str(benchmark), str(EVIDENCE / 'benchmark.json'), '--archive-dir', str(SOURCE / 'incumbent-sources'),
                '--data-dir', DATA, '--core', str(CORE), '--source-root', str(SOURCE),
                '--mesh-days', '3', '--repeats', '3', '--warmups', '1']
        code = ("import sys,runpy; from pathlib import Path; "
                "sys.meta_path=[f for f in sys.meta_path if f.__class__.__module__!='_editable_skbc_spacepdhcg']; "
                "sys.path.insert(0,str(Path('src').resolve())); sys.argv=" + repr(argv) + "; "
                "runpy.run_path(sys.argv[0],run_name='__main__')")
        run('benchmark', [PYTHON, '-c', code])
        bench = json.loads((EVIDENCE / 'benchmark.json').read_text())
        assert bench['complete'] and bench['correctness_passed'], bench.get('error')
        REPORT['benchmark_correctness_passed'] = bench['correctness_passed']
        run('gpu-after', ['nvidia-smi', '-q'])
        REPORT['gpu_lock_released_utc'] = stamp()
    REPORT['success'] = True
except BaseException:
    REPORT['exception'] = traceback.format_exc()
    (EVIDENCE / 'exception.log').write_text(REPORT['exception'])
finally:
    REPORT['finished_utc'] = stamp()
    REPORT['complete'] = True
    status('complete' if REPORT['success'] else 'failed', success=REPORT['success'])
sys.exit(0 if REPORT['success'] else 1)
