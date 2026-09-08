from pathlib import Path
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import time
from analyse_qps_v169 import problem, audit

root = Path('build/performance/conditioning-paths-v529')
root.mkdir(exist_ok=False)
source = Path('/home/angus/build-qoco-soc-step-v358/source')
lib = Path('/home/angus/build-qoco-soc-step-v358/final/libqoco.so')
binary = (root / 'qoco_snapshot_replay').resolve()
env = {k: v for k, v in os.environ.items() if not k.startswith(('SPACEPDHCG_TEST_', 'QOCO_REPLAY_'))}
env['LD_LIBRARY_PATH'] = str(lib.parent) + ':/home/angus/cudss-isolated-0.8.0.10/nvidia/cu12/lib:/usr/local/cuda-12.8/lib64'
report = dict(pid=os.getpid(), complete=False, cases=[],
              qoco_sha256=hashlib.sha256(lib.read_bytes()).hexdigest(),
              scope='Identical QP and tolerances; direct setup versus numeric update, graph versus host-dispatched GPU arithmetic. Fresh process for each of three repeats.')

def save():
    tmp = root / 'report.tmp'
    tmp.write_text(json.dumps(report, indent=2))
    tmp.replace(root / 'report.json')

save()
try:
    code = Path('cpp/cuda/tests/qoco_snapshot_replay.cu').read_text()
    code = code.replace('auto setup_settings=settings;setup_settings.ruiz_iters=0;',
                        'const bool direct=std::getenv("QOCO_REPLAY_DIRECT_SETUP") != nullptr;\n'
                        'if(direct && shift) throw std::runtime_error("direct shift unsupported");\n'
                        'auto setup_settings=settings;if(!direct) setup_settings.ruiz_iters=0;')
    code = code.replace('CHECK(qoco_gpu_update_numeric(update,values,nullptr));',
                        'if(!direct) CHECK(qoco_gpu_update_numeric(update,values,nullptr));')
    (root / 'qoco_snapshot_replay.cu').write_text(code)
    for name in ['analyse_qps_v169.py', 'replay_conditioning_paths_v529.py']:
        shutil.copy2(Path('build/performance') / name, root / name)
    cmd = ['/usr/local/cuda-12.8/bin/nvcc', '--default-stream', 'per-thread', '-arch=sm_120', '-std=c++17']
    for include in ['include', 'algebra/cuda', 'lib/qdldl/include', 'lib/amd']:
        cmd += ['-I', str(source / include)]
    cmd += [str(root / 'qoco_snapshot_replay.cu'), '-L', str(lib.parent), '-lqoco', '-o', str(binary)]
    report['build_command'] = cmd
    built = subprocess.run(cmd, env=env, text=True, capture_output=True)
    (root / 'build.log').write_text(built.stdout + built.stderr)
    built.check_returncode()
    report['binary_sha256'] = hashlib.sha256(binary.read_bytes()).hexdigest()
    with open('/home/angus/.spacepdhcg-gpu.lock', 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        for ruiz in [0, 2]:
            qp = root / f'ruiz{ruiz}.txt'
            shutil.copy2(Path('build/performance/conditioning-qp-reg-v527') / qp.name, qp)
            data = problem(qp)
            for direct in [False, True]:
                for graph in [False, True]:
                    env['SPACEPDHCG_TEST_QOCO_IPM_GRAPH'] = str(int(graph))
                    env.pop('QOCO_REPLAY_DIRECT_SETUP', None)
                    if direct:
                        env['QOCO_REPLAY_DIRECT_SETUP'] = '1'
                    name = f'ruiz{ruiz}-direct{int(direct)}-graph{int(graph)}'
                    row = dict(ruiz=ruiz, direct=direct, graph=graph, audits=[], iterations=[], seconds=[])
                    report['cases'].append(row)
                    for repeat in range(3):
                        log_path = root / f'{name}-{repeat}.log'
                        start = time.perf_counter()
                        with log_path.open('x') as log:
                            child = subprocess.Popen([str(binary), str(qp)], env=env, stdout=log, stderr=subprocess.STDOUT)
                            report.update(stage=name, repeat=repeat, child_pid=child.pid)
                            save()
                            result = child.wait(timeout=120)
                        assert result == 0, (name, result)
                        records = [json.loads(s[10:]) for s in log_path.read_text().splitlines() if s.startswith('QP_REPLAY ')]
                        assert len(records) == 1
                        row['audits'].append(audit(data, records[0]))
                        row['iterations'].append(records[0]['iterations'])
                        row['seconds'].append(time.perf_counter() - start)
                        save()
                    print(name, sum(a['qualified'] for a in row['audits']), '/3', row['iterations'], flush=True)
    report['complete'] = True
except Exception as error:
    report['error'] = repr(error)
save()
