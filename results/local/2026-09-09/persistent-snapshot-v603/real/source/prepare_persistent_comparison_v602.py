"""Freeze two real captured problems and build the existing QOCO replay reference."""
from pathlib import Path
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import time

workspace = Path(__file__).resolve().parents[2]
root = workspace / "build/performance/persistent-replay-20260909"
root.mkdir(exist_ok=False)
(root / "inputs").mkdir()
(root / "source").mkdir()
qoco_root = Path('/home/angus/build-qoco-scaled-pool-v540')
qoco = qoco_root / 'final/libqoco.so'
assert hashlib.sha256(qoco.read_bytes()).hexdigest() == '0cc27a1d8bde1edd74f7ef00e04ec64c2d6c0f8fe526a1d94b04d509a74b3315'

def record(path):
    return dict(path=str(path), bytes=path.stat().st_size,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest())

for name in ('scripts/gpu/audit_persistent_snapshot.py', 'tests/test_persistent_snapshot_audit.py',
             'cpp/cuda/tests/qoco_snapshot_replay.cu'):
    shutil.copyfile(workspace / name, root / "source" / Path(name).name)
shutil.copyfile(__file__, root / 'source' / Path(__file__).name)
spec = importlib.util.spec_from_file_location('independent_audit', root/'source/audit_persistent_snapshot.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)
manifest = dict(complete=False, gpu_calls=0, qoco=record(qoco), inputs=[],
                source={p.name: record(p) for p in (root/'source').iterdir() if p.is_file()})
inputs = [
    ('conditioning', 'build/performance/conditioning-qp-v526/origin0-ruiz0/snapshots/qp-534-000000.txt',
     '1eb1b5a3979c5e86522337dfa94f264e354993163ed44129eff361371e3b49cf'),
    ('difficult', 'build/performance/qp-ir-v309/qp.txt',
     '14fdc84e9b654df4ac16db71f0bce4fccfa62577f9620ae622ad2196e2f17080'),
]
for label, relative, expected in inputs:
    source = workspace / relative
    assert record(source)['sha256'] == expected
    target = root / 'inputs' / (label+'.txt')
    shutil.copyfile(source, target)
    problem = audit.load_snapshot(target)
    manifest['inputs'].append(dict(label=label, file=record(target),
        dimensions={k: problem[k] for k in ('n','p','m','l','shift')},
        soc_count=len(problem['soc']),
        quadratic_structural_nonzeros=problem['quadratic_structural_nonzeros'],
        quadratic_numerical_nonzeros=problem['quadratic_numerical_nonzeros']))
include = qoco_root / 'source/include'
manifest['qoco_headers'] = {p.name: record(p) for p in include.glob('*.h')}
binary = root/'qoco_snapshot_replay'
cuda_lib = '/usr/local/cuda-12.8/lib64'
cudss_lib = '/home/angus/cudss-isolated-0.8.0.10/nvidia/cu12/lib'
environment = dict(os.environ, LD_LIBRARY_PATH=f'{qoco.parent}:{cudss_lib}:{cuda_lib}')
command = ['g++','-std=c++17','-O2','-x','c++',str(root/'source/qoco_snapshot_replay.cu'),
           '-I'+str(include),'-I'+str(qoco_root/'source/algebra'),'-I/usr/local/cuda-12.8/include',
           '-L'+str(qoco.parent),'-L'+cuda_lib,'-Wl,-rpath,'+str(qoco.parent)+':'+cudss_lib+':'+cuda_lib,
           '-lqoco','-lcudart','-o',str(binary)]
began = time.perf_counter()
with (root/'qoco-build.log').open('x') as log:
    result = subprocess.run(command, env=environment, stdout=log, stderr=subprocess.STDOUT, timeout=90)
manifest['qoco_build'] = dict(command=command, returncode=result.returncode, seconds=time.perf_counter()-began)
manifest['complete'] = result.returncode == 0
if manifest['complete']:
    manifest['qoco_replay'] = record(binary)
(root/'preparation.json').write_text(json.dumps(manifest, indent=2)+'\n')
print(json.dumps({k: manifest[k] for k in ('complete','gpu_calls','inputs','qoco_build')}, indent=2))
raise SystemExit(result.returncode)
