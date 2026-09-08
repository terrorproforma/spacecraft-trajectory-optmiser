"""One finite, serialized local GPU parity test from the reviewed e snapshot."""
from pathlib import Path
import datetime
import fcntl
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

ROOT = Path('/home/angus/spacepdhcg-completion-model-v622g')
LIVE = Path('/mnt/c/Users/Angus/Desktop/projects/spacecraft-trajectory-optmiser')
OUT = LIVE/'build/performance/completion-model-gpu-v622d'
UUID = 'GPU-4df2f6b5-e866-14a0-eeac-332cb2b757d4'
REPORT_SHA = '56a65032e639a6eb9af08052e9f3df87f5f1c84cfe9cf2ece6f3f63eafb2a12a'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()

if '--child' in sys.argv:
    sys.meta_path = [f for f in sys.meta_path if f.__class__.__module__ != '_editable_skbc_spacepdhcg']
    sys.path.insert(0, str(ROOT/'source/src'))
    import spacepdhcg.gtoc12.gpu_completion as completion
    assert Path(completion.__file__).resolve() == ROOT/'source/src/spacepdhcg/gtoc12/gpu_completion.py'
    import pytest
    raise SystemExit(pytest.main([str(ROOT/'source/tests/test_gtoc12_gpu_completion_model.py'),
        '-q', '-p', 'no:cacheprovider', '--junitxml='+str(OUT/'pytest.xml')]))

assert sha(ROOT/'report.json') == REPORT_SHA
freeze = json.loads((ROOT/'report.json').read_text())
assert freeze['complete'] and freeze['gpu_calls'] == 0
assert sha(ROOT/'source.tar.gz') == freeze['source_archive_sha256']
for name, digest in freeze['owned_sources'].items(): assert sha(ROOT/'source'/name) == digest
assert sha(Path(freeze['library']['path'])) == freeze['library']['sha256']
OUT.mkdir(exist_ok=False)
(OUT/'readbacks').mkdir()
(OUT/'run.py').write_bytes(Path(__file__).read_bytes())
report = {'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(), 'complete':False,
    'source_report_sha256':REPORT_SHA, 'library_sha256':freeze['library']['sha256'],
    'gpu_uuid':UUID, 'maximum_valid_evaluations':18, 'maximum_candidates':2358,
    'maximum_invalid_calls':6, 'worker_timeout_seconds':90,
    'fresh_lambert_solves':0, 'fresh_refinements':0, 'fleet_promotions':0}
save = lambda: (OUT/'report.json').write_text(json.dumps(report, indent=2)+'\n')
save()
with open('/home/angus/.spacepdhcg-gpu.lock', 'a') as lock:
    # Join the existing GPU queue once; never race or interrupt its current job.
    # This supervisor remains foreground-owned and yields through exec while it
    # waits. A finite alarm ends the wait without starting any GPU work.
    report['lock_wait_limit_seconds'] = 300
    report['status'] = 'waiting_for_gpu_lock'
    save()
    def wait_timeout(signum, frame):
        raise TimeoutError('GPU lock wait exceeded 300 seconds; zero GPU calls')
    previous_handler = signal.signal(signal.SIGALRM, wait_timeout)
    signal.setitimer(signal.ITIMER_REAL, 300)
    waiting = time.perf_counter()
    try:
        fcntl.flock(lock, fcntl.LOCK_EX)
    except TimeoutError:
        report['status'] = 'gpu_lock_wait_timeout_zero_GPU_calls'
        save()
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
    report['lock_wait_seconds'] = time.perf_counter() - waiting
    report['status'] = 'running'
    save()
    def query(flag):
        p = subprocess.run(['nvidia-smi', flag, '--format=csv,noheader'],
                           capture_output=True, text=True, timeout=10)
        assert p.returncode == 0, p.stderr
        return p.stdout.strip()
    report['preflight_gpu'] = query('--query-gpu=uuid,name,utilization.gpu,memory.used')
    report['preflight_compute'] = query('--query-compute-apps=pid,process_name')
    assert UUID in report['preflight_gpu'] and not report['preflight_compute'], report
    env = {k:v for k,v in os.environ.items() if not k.startswith(('SPACEPDHCG_', 'QOCO_', 'PDHCG_', 'LD_LIBRARY_PATH'))}
    env.update(CUDA_VISIBLE_DEVICES=UUID, PYTHONDONTWRITEBYTECODE='1',
        SPACEPDHCG_GTOC12_GPU_TESTS='1', SPACEPDHCG_COMPLETION_TEST_LIBRARY=freeze['library']['path'],
        SPACEPDHCG_COMPLETION_TEST_READBACKS=str(OUT/'readbacks'))
    start = time.perf_counter()
    with (OUT/'pytest.log').open('x') as output:
        p = subprocess.Popen([sys.executable, '-B', str(OUT/'run.py'), '--child'], cwd=ROOT/'source',
                             env=env, stdout=output, stderr=subprocess.STDOUT, start_new_session=True,
                             pass_fds=(lock.fileno(),))
        report['worker_pid'] = p.pid
        save()
        try: code = p.wait(timeout=90)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGTERM)
            try: code = p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL); code = p.wait(timeout=5)
            report['timed_out'] = True
    report.update(exit_code=code, worker_seconds=time.perf_counter()-start)
    report['postflight_compute'] = query('--query-compute-apps=pid,process_name')
report['readbacks'] = {p.name:{'sha256':sha(p), 'bytes':p.stat().st_size} for p in sorted((OUT/'readbacks').iterdir())}
expected = {f'{model}-{size}.npz' for model in ('fit','flat','ratio') for size in (259,3)}
report['expected_readbacks_present'] = set(report['readbacks']) == expected
suites = list(ET.parse(OUT/'pytest.xml').getroot().iter('testsuite'))
report['junit'] = {key:sum(int(s.attrib.get(key,0)) for s in suites)
                   for key in ('tests','failures','errors','skipped')}
report['complete'] = (code == 0 and report['expected_readbacks_present'] and
    report['junit'] == {'tests':7,'failures':0,'errors':0,'skipped':0})
if not report['complete'] and code == 0: code = 1
report['status'] = 'passed' if report['complete'] else 'failed'
report['log_sha256'] = sha(OUT/'pytest.log')
save()
print((OUT/'pytest.log').read_text()[-12000:])
print(json.dumps(report, indent=2))
raise SystemExit(code)



