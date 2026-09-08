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
EVIDENCE = Path('/mnt/c/Users/Angus/Desktop/projects/spacecraft-trajectory-optmiser/build/performance/joint-local-v596/evidence')
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
    target = EVIDENCE.parent / 'status.json'
    temp = EVIDENCE.parent / 'status.tmp.json'
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
    manifest = json.loads((ROOT / 'source-manifest.json').read_text())
    for name, record in manifest['files'].items():
        assert hashlib.sha256((SOURCE / name).read_bytes()).hexdigest() == record['sha256'], name
    REPORT['source_manifest_sha256'] = hashlib.sha256((ROOT / 'source-manifest.json').read_bytes()).hexdigest()
    REPORT['base_commit'] = manifest['base_commit']
    # This commit exists only in the isolated validation payload; no project checkout changes.
    git_env = dict(ENV, GIT_CONFIG_NOSYSTEM='1', GIT_CONFIG_GLOBAL='/dev/null',
                   GIT_AUTHOR_NAME='Validation snapshot', GIT_AUTHOR_EMAIL='validation@localhost',
                   GIT_COMMITTER_NAME='Validation snapshot', GIT_COMMITTER_EMAIL='validation@localhost')
    run('snapshot-git-init', ['git', 'init', str(SOURCE)], environment=git_env)
    run('snapshot-git-add', ['git', '-C', str(SOURCE), 'add', '-f', '.'], environment=git_env)
    run('snapshot-git-commit', ['git', '-c', 'core.hooksPath=/dev/null', '-C', str(SOURCE), 'commit',
                              '-m', 'Isolated H100 joint validation snapshot based on ' + REPORT['base_commit']], environment=git_env)
    REPORT['isolated_snapshot_commit'] = subprocess.check_output(['git', '-C', str(SOURCE), 'rev-parse', 'HEAD'], text=True).strip()
    run('cmake-configure', [CMAKE, '-S', str(SOURCE / 'cpp'), '-B', str(BUILD),
                           '-DCMAKE_BUILD_TYPE=Release', '-DSPACEPDHCG_BUILD_CUDA=ON',
                           '-DSPACEPDHCG_BUILD_NATIVE_TESTS=OFF', '-DCMAKE_CUDA_ARCHITECTURES=120',
                           '-DCMAKE_CUDA_COMPILER=' + NVCC,
                           '-DSPACEPDHCG_PDHCG_SOURCE_ROOT=' + UPSTREAM])
    run('cmake-build', [CMAKE, '--build', str(BUILD), '--target', 'spacepdhcg_cuda', '-j', '8'])
    REPORT['core'] = {'path': str(CORE), 'sha256': hashlib.sha256(CORE.read_bytes()).hexdigest(), 'bytes': CORE.stat().st_size}
    smoke = ROOT / 'joint-smoke'
    run('smoke-build', [NVCC, '-std=c++17', '-UNDEBUG', '-arch=sm_120', '--fmad=false',
                        '-I' + str(SOURCE / 'cpp/cuda/include'),
                        str(SOURCE / 'cpp/cuda/tests/gtoc12_joint_smoke.cu'), str(CORE),
                        '-Xlinker', '-rpath=' + str(CORE.parent), '-o', str(smoke)])
    REPORT['smoke_sha256'] = hashlib.sha256(smoke.read_bytes()).hexdigest()
    selection_smoke = ROOT / 'joint-selection-test'
    run('selection-build', [NVCC, '-std=c++17', '-UNDEBUG', '-arch=sm_120', '--fmad=false',
                            '-I' + str(SOURCE / 'cpp/cuda/include'),
                            str(SOURCE / 'cpp/cuda/tests/gtoc12_joint_selection_test.cu'), str(CORE),
                            '-Xlinker', '-rpath=' + str(CORE.parent), '-o', str(selection_smoke)])
    REPORT['selection_test_sha256'] = hashlib.sha256(selection_smoke.read_bytes()).hexdigest()
    with Path('/home/angus/.spacepdhcg-gpu.lock').open('a') as lock:
        status('waiting-for-gpu-lock')
        fcntl.flock(lock, fcntl.LOCK_EX)
        REPORT['gpu_lock_acquired_utc'] = stamp()
        # Respect non-cooperating jobs as well: do not run if another GPU process exists.
        while True:
            gpu_processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,process_name', '--format=csv,noheader'], text=True).strip()
            if not gpu_processes:
                break
            status('waiting-for-idle-gpu', processes=gpu_processes)
            time.sleep(15)
        run('gpu-before', ['nvidia-smi', '-q'])
        run('native-smoke', [str(smoke)])
        for checker in ('memcheck', 'synccheck', 'racecheck'):
            run(checker, ['/usr/local/cuda-12.8/bin/compute-sanitizer', '--tool', checker,
                          '--error-exitcode', '99', str(smoke)])
        run('native-selection', [str(selection_smoke)])
        for checker in ('memcheck', 'synccheck', 'racecheck'):
            run('selection-' + checker, ['/usr/local/cuda-12.8/bin/compute-sanitizer', '--tool', checker,
                                        '--error-exitcode', '99', str(selection_smoke)])
        junit = EVIDENCE / 'pytest.xml'
        run('python-parity', parity_command(junit))
        check_parity('python-parity', junit)
        for checker in ('memcheck', 'synccheck', 'racecheck'):
            name = 'python-' + checker
            junit = EVIDENCE / (name + '.xml')
            run(name, ['/usr/local/cuda-12.8/bin/compute-sanitizer', '--tool', checker,
                       '--target-processes', 'all', '--error-exitcode', '99', *parity_command(junit)])
            check_parity(name, junit)
        selection_junit = EVIDENCE / 'python-selection.xml'
        selection_command = parity_command(selection_junit)
        selection_command[-1] = selection_command[-1].replace(
            "'tests/test_gtoc12_gpu_joint.py','tests/test_gtoc12_jointopt.py'",
            "'tests/test_gtoc12_gpu_joint_selection.py'")
        run('python-selection', selection_command)
        selection_suites = ET.parse(selection_junit).getroot().findall('testsuite')
        selection_totals = {key: sum(int(s.get(key, '0')) for s in selection_suites)
                            for key in ('tests', 'failures', 'errors', 'skipped')}
        REPORT['pytest_totals']['python-selection'] = selection_totals
        assert selection_totals == {'tests': 8, 'failures': 0, 'errors': 0, 'skipped': 0}, selection_totals
        for checker in ('memcheck', 'synccheck', 'racecheck'):
            name = 'python-selection-' + checker
            junit = EVIDENCE / (name + '.xml')
            command = parity_command(junit)
            command[-1] = command[-1].replace(
                "'tests/test_gtoc12_gpu_joint.py','tests/test_gtoc12_jointopt.py'",
                "'tests/test_gtoc12_gpu_joint_selection.py'")
            run(name, ['/usr/local/cuda-12.8/bin/compute-sanitizer', '--tool', checker,
                       '--target-processes', 'all', '--error-exitcode', '99', *command])
            suites = ET.parse(junit).getroot().findall('testsuite')
            totals = {key: sum(int(s.get(key, '0')) for s in suites)
                      for key in ('tests', 'failures', 'errors', 'skipped')}
            REPORT['pytest_totals'][name] = totals
            assert totals == {'tests': 8, 'failures': 0, 'errors': 0, 'skipped': 0}, totals
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
