from pathlib import Path
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import time

base = Path('/home/angus/spacepdhcg-return-qp-v676')
out = Path('/home/angus/spacepdhcg-return-conditioning-v681')
out.mkdir(exist_ok=False)
sys.path.insert(0, str(base / 'replay-v678'))
from audit_helper import problem, audit
original = base / 'replay-v678/input-qp.txt'
data = problem(original)
lines = original.read_text().splitlines()
binary = base / 'replay-v678/qoco_snapshot_replay'
env = {k: v for k, v in os.environ.items() if not k.startswith(('SPACEPDHCG_TEST_', 'QOCO_REPLAY_'))}
env.update(SPACEPDHCG_TEST_QOCO_IPM_GRAPH='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', LD_LIBRARY_PATH='/home/angus/build-qoco-scaled-pool-v540/final:/home/angus/cudss-isolated-0.8.0.10/nvidia/cu12/lib:/usr/local/cuda-12.8/lib64')
cases = [('baseline', {}, {}, {}), ('reg-1e-11', {}, {1:'1e-11', 2:'1e-11', 3:'1e-11'}, {}), ('reg-1e-13', {}, {1:'1e-13', 2:'1e-13', 3:'1e-13'}, {}), ('reg-1e-7', {}, {1:'1e-7', 2:'1e-7', 3:'1e-7'}, {}), ('ir-100-tight', {2:'100'}, {0:'1e-14'}, {}), ('ruiz5', {1:'5'}, {}, {}), ('ruiz10', {1:'10'}, {}, {}), ('ruiz5-preserve', {1:'5'}, {}, {'SPACEPDHCG_TEST_QOCO_RUIZ_PRESERVE_OBJECTIVE':'1'})]
report = dict(complete=False, original_sha256=hashlib.sha256(original.read_bytes()).hexdigest(), binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(), physical_equations_changed=False, rows=[])
def save():
    (out / 'report.json').write_text(json.dumps(report, indent=2))
save()
try:
    with Path('/home/angus/.spacepdhcg-gpu.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        for name, ints, floats, flags in cases:
            modified = list(lines)
            for index, replacements in ((2, ints), (3, floats)):
                values = modified[index].split()
                for key, value in replacements.items(): values[key] = value
                modified[index] = ' '.join(values)
            assert modified[4:] == lines[4:]
            snapshot = out / (name + '.txt')
            snapshot.write_text('\n'.join(modified) + '\n')
            started = time.perf_counter()
            with (out / (name + '.log')).open('x') as log:
                result = subprocess.run([str(binary), str(snapshot), '4'], env=dict(env, **flags), stdout=log, stderr=subprocess.STDOUT, timeout=120)
            records = [json.loads(line[10:]) for line in (out / (name + '.log')).read_text().splitlines() if line.startswith('QP_REPLAY ')]
            checks = [dict(status=r['status'], iterations=r['iterations'], ir_iterations=r['ir_iterations'], **audit(data, r)) for r in records]
            row = dict(name=name, settings_lines=modified[2:4], flags=flags, returncode=result.returncode, seconds=time.perf_counter()-started, audits=checks, qualified=sum(r['qualified'] for r in checks), repeats=len(checks))
            report['rows'].append(row)
            save()
            print(json.dumps({k:v for k,v in row.items() if k not in ('audits','settings_lines')}), flush=True)
    report['complete'] = True
finally:
    save()
