"""Compile the frozen diagnostic with the CUDA toolchain; no GPU solves."""
import hashlib
import json
import os
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parent / 'upstream-snapshot-v608'
manifest = json.loads((root / 'manifest.json').read_text())
out = root / 'nvcc-cmake-check'
out.mkdir(exist_ok=False)
old = manifest['command']
definitions = [arg for arg in old if arg.startswith('-D')]
includes = [arg for arg in old if arg.startswith('-I')]
command = ['/usr/local/cuda-12.8/bin/nvcc', '-std=c++20', '-O3', '-arch=sm_120', '--cudart', 'shared',
           '-Xcompiler=-Wall,-Wextra,-Werror', str(root / 'source/cpp/cuda/tests/upstream_snapshot_replay.cu'),
           manifest['reference_archive'], *includes, *definitions, '-L/usr/local/cuda-12.8/lib64',
           '-Xlinker=-rpath', '-Xlinker=/usr/local/cuda-12.8/lib64', '-lcublas', '-lcusolver', '-lcusparse',
           '-lz', '-lpthread', '-ldl', '-o', str(out / 'upstream_snapshot_replay')]
report = {'complete': False, 'gpu_calls': 0, 'command': command,
          'flags_note': 'g++ source already passed Wpedantic; nvcc-generated GCC line directives reject Wpedantic and are not used by the CMake target; first flag failure retained in nvcc-check'}
(out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
with (out / 'build.log').open('x') as log:
    built = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=90)
report['build_returncode'] = built.returncode
(out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
assert built.returncode == 0
with (out / 'validate.log').open('x') as log:
    checked = subprocess.run([str(out / 'upstream_snapshot_replay'), str(root / 'fixtures/mixed.txt'), '--validate-only'],
                             env=os.environ | {'CUDA_VISIBLE_DEVICES': '-1'}, stdout=log, stderr=subprocess.STDOUT, timeout=15)
report['validate_returncode'] = checked.returncode
assert checked.returncode == 0
report['executable_sha256'] = hashlib.sha256((out / 'upstream_snapshot_replay').read_bytes()).hexdigest()
report['complete'] = True
(out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report))
