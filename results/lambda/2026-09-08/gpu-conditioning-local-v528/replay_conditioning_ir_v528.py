from pathlib import Path
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import time
from analyse_qps_v169 import problem, audit

root = Path('build/performance/conditioning-qp-ir-v528')
root.mkdir(exist_ok=False)
prior = Path('build/performance/conditioning-qp-reg-v527')
binary = (prior / 'qoco_snapshot_replay').resolve()
lib = Path('/home/angus/build-qoco-soc-step-v358/final/libqoco.so')
env = {k: v for k, v in os.environ.items() if not k.startswith(('SPACEPDHCG_TEST_', 'QOCO_REPLAY_'))}
env.update(SPACEPDHCG_TEST_QOCO_IPM_GRAPH='1', LD_LIBRARY_PATH=str(lib.parent) + ':/home/angus/cudss-isolated-0.8.0.10/nvidia/cu12/lib:/usr/local/cuda-12.8/lib64')
report = dict(pid=os.getpid(), complete=False, cases=[], qoco_sha256=hashlib.sha256(lib.read_bytes()).hexdigest(),
              binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest())
def save():
    temporary = root / 'report.tmp'
    temporary.write_text(json.dumps(report, indent=2))
    temporary.replace(root / 'report.json')
save()
for name in ['analyse_qps_v169.py', 'replay_conditioning_ir_v528.py']:
    shutil.copy2(Path('build/performance') / name, root / name)
try:
    with open('/home/angus/.spacepdhcg-gpu.lock', 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        for tolerance, maximum in [('1e-12', 20), ('1e-14', 20), ('1e-16', 20), ('1e-14', 80), ('1e-16', 80)]:
            name = f'ir{tolerance}-max{maximum}'
            lines = (prior / 'ruiz2.txt').read_text().splitlines()
            integers = lines[2].split()
            integers[2] = str(maximum)
            numbers = lines[3].split()
            numbers[0] = tolerance
            lines[2], lines[3] = ' '.join(integers), ' '.join(numbers)
            qp = root / (name + '.txt')
            qp.write_text('\n'.join(lines) + '\n')
            data = problem(qp)
            start = time.perf_counter()
            with (root / (name + '.log')).open('x') as log:
                child = subprocess.Popen([str(binary), str(qp), '3'], env=env, stdout=log, stderr=subprocess.STDOUT)
                report.update(stage=name, child_pid=child.pid)
                save()
                code = child.wait(timeout=180)
            assert code == 0, (name, code)
            records = [json.loads(s[10:]) for s in (root / (name + '.log')).read_text().splitlines() if s.startswith('QP_REPLAY ')]
            assert len(records) == 3
            row = dict(ir_tolerance=tolerance, max_ir_iterations=maximum, seconds=time.perf_counter() - start,
                       iterations=[r['iterations'] for r in records], audits=[audit(data, r) for r in records])
            report['cases'].append(row)
            save()
            print(name, sum(a['qualified'] for a in row['audits']), '/ 3', row['iterations'], flush=True)
    report['complete'] = True
except Exception as error:
    report['error'] = repr(error)
save()
