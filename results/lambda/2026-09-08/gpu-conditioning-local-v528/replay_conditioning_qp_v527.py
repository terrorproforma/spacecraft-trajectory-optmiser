from pathlib import Path
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import time

from analyse_qps_v169 import problem, audit

root = Path('build/performance/conditioning-qp-reg-v527')
root.mkdir(exist_ok=False)
source = Path('/home/angus/build-qoco-soc-step-v358/source')
lib = Path('/home/angus/build-qoco-soc-step-v358/final/libqoco.so')
binary = (root / 'qoco_snapshot_replay').resolve()
env = {k: v for k, v in os.environ.items() if not k.startswith(('SPACEPDHCG_TEST_', 'QOCO_REPLAY_'))}
env.update(SPACEPDHCG_TEST_QOCO_IPM_GRAPH='1',
           LD_LIBRARY_PATH=str(lib.parent) + ':/home/angus/cudss-isolated-0.8.0.10/nvidia/cu12/lib:/usr/local/cuda-12.8/lib64')
report = dict(pid=os.getpid(), complete=False, cases=[],
              qoco_sha256=hashlib.sha256(lib.read_bytes()).hexdigest())
def save():
    temporary = root / 'report.tmp'
    temporary.write_text(json.dumps(report, indent=2))
    temporary.replace(root / 'report.json')
save()
try:
    paths = [next(Path(f'build/performance/conditioning-qp-v526/origin0-ruiz{ruiz}/snapshots').glob('qp-*.txt')) for ruiz in [0, 2]]
    lines = [p.read_text().splitlines() for p in paths]
    assert lines[0][:2] == lines[1][:2] and lines[0][3:] == lines[1][3:]
    report['same_numerical_qp'] = True
    for q in paths:
        shutil.copy2(q, root / ('ruiz' + q.parent.parent.name[-1] + '.txt'))
    for name in ['analyse_qps_v169.py', 'replay_conditioning_qp_v527.py']:
        shutil.copy2(Path('build/performance') / name, root / name)
    shutil.copy2('cpp/cuda/tests/qoco_snapshot_replay.cu', root / 'qoco_snapshot_replay.cu')
    cmd = ['/usr/local/cuda-12.8/bin/nvcc', '--default-stream', 'per-thread', '-arch=sm_120', '-std=c++17']
    for include in ['include', 'algebra/cuda', 'lib/qdldl/include', 'lib/amd']:
        cmd += ['-I', str(source / include)]
    cmd += ['cpp/cuda/tests/qoco_snapshot_replay.cu', '-L', str(lib.parent), '-lqoco', '-o', str(binary)]
    built = subprocess.run(cmd, env=env, text=True, capture_output=True)
    (root / 'build.log').write_text(built.stdout + built.stderr)
    built.check_returncode()
    report['binary_sha256'] = hashlib.sha256(binary.read_bytes()).hexdigest()
    with open('/home/angus/.spacepdhcg-gpu.lock', 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        for ruiz in [0, 2]:
            qp = root / f'ruiz{ruiz}.txt'
            data = problem(qp)
            for regularization in ['1e-8', '1e-10', '1e-12', '1e-6']:
                name = f'ruiz{ruiz}-reg{regularization}'
                env['QOCO_REPLAY_REGULARIZATION'] = regularization
                start = time.perf_counter()
                with (root / (name + '.log')).open('x') as log:
                    child = subprocess.Popen([str(binary), str(qp), '3'], env=env, stdout=log, stderr=subprocess.STDOUT)
                    report.update(stage=name, child_pid=child.pid)
                    save()
                    code = child.wait(timeout=120)
                assert code == 0, (name, code)
                records = [json.loads(s[10:]) for s in (root / (name + '.log')).read_text().splitlines() if s.startswith('QP_REPLAY ')]
                assert len(records) == 3
                row = dict(ruiz=ruiz, regularization=regularization, seconds=time.perf_counter() - start,
                           iterations=[r['iterations'] for r in records], audits=[audit(data, r) for r in records])
                report['cases'].append(row)
                save()
                print(name, sum(a['qualified'] for a in row['audits']), '/ 3', row['iterations'], flush=True)
    report['complete'] = True
except Exception as error:
    report['error'] = repr(error)
save()
