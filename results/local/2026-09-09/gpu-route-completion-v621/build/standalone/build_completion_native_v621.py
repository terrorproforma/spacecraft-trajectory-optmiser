"""Freeze/compile the new completion translation unit; run CPU-only checks, never GPU work."""
from pathlib import Path
import hashlib
import json
import os
import shutil
import subprocess
import tarfile
import time

live = Path('/mnt/c/Users/Angus/Desktop/projects/spacecraft-trajectory-optmiser')
root = Path('/home/angus/spacepdhcg-completion-v621')
evidence = live / 'build/performance/completion-native-core-v621'
root.mkdir(exist_ok=False)
evidence.mkdir(exist_ok=False)
repo = root / 'repo'
owned = [
    'cpp/cuda/include/spacepdhcg/cuda/gtoc12_completion_c_api.h',
    'cpp/cuda/src/gtoc12_completion.cu',
    'cpp/cuda/tests/gtoc12_completion_test.cu',
    'cpp/cuda/CMakeLists.txt',
]
for name in owned:
    target = repo / name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(live / name, target)
shutil.copy2(__file__, root / Path(__file__).name)
digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
manifest = {
    'complete': False, 'gpu_calls': 0, 'git_operations': 0,
    'source_identity_kind': 'sha256_tree_not_git_commit', 'frozen_commit': None,
    'source_sha256': {name: digest(repo / name) for name in owned},
    'scope': 'Standalone completion shared library and linked test; integration CMake recorded, not full-core rebuild.',
    'stages': [],
}
manifest['source_tree_sha256'] = hashlib.sha256(''.join(
    k + ':' + v + '\n' for k, v in sorted(manifest['source_sha256'].items())
).encode()).hexdigest()
with tarfile.open(root / 'source.tar.gz', 'w:gz') as archive:
    for name in owned:
        archive.add(repo / name, arcname=name, recursive=False)
manifest['source_archive_sha256'] = digest(root / 'source.tar.gz')
env = {k: v for k, v in os.environ.items()
       if not k.startswith(('SPACEPDHCG_', 'QOCO_', 'PDHCG_', 'LD_LIBRARY_PATH'))}
env['CUDA_VISIBLE_DEVICES'] = ''

def save():
    (root / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    for p in root.iterdir():
        if p.is_file() and p.suffix in ('.json', '.log', '.py', '.gz'):
            shutil.copy2(p, evidence / p.name)

def call(name, command):
    start = time.perf_counter()
    with (root / (name + '.log')).open('x') as log:
        result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                timeout=180, cwd=repo)
    manifest['stages'].append({'name': name, 'command': command,
                              'returncode': result.returncode,
                              'seconds': time.perf_counter() - start})
    save()
    if result.returncode:
        raise RuntimeError(name + ': ' + (root / (name + '.log')).read_text()[-6000:])

save()
nvcc = '/usr/local/cuda-12.8/bin/nvcc'
build = root / 'build'
build.mkdir()
common = [nvcc, '-std=c++20', '-O3', '--fmad=false',
          '-gencode=arch=compute_90,code=sm_90', '-gencode=arch=compute_120,code=sm_120',
          '-I' + str(repo / 'cpp/cuda/include')]
call('compiler-version', [nvcc, '--version'])
library = build / 'libspacepdhcg_completion.so'
test = build / 'gtoc12_completion_test'
call('library-build', common + ['-shared', '-Xcompiler=-fPIC',
     str(repo / 'cpp/cuda/src/gtoc12_completion.cu'), '-o', str(library)])
call('test-build', common + [str(repo / 'cpp/cuda/tests/gtoc12_completion_test.cu'),
     '-L' + str(build), '-lspacepdhcg_completion', '-Xlinker=-rpath',
     '-Xlinker=$ORIGIN', '-o', str(test)])
call('cpu-fixture-check', [str(test), '--cpu-only'])
records = [json.loads(line.split(' ', 1)[1]) for line in
           (root / 'cpu-fixture-check.log').read_text().splitlines()
           if line.startswith('COMPLETION_TEST ')]
assert records == [{'phase': 'cpu_only', 'fixtures': 262, 'GPU_calls': 0}]
call('resource-usage', ['/usr/local/cuda-12.8/bin/cuobjdump', '--dump-resource-usage', str(library)])
manifest['library'] = {'path': str(library), 'bytes': library.stat().st_size, 'sha256': digest(library)}
manifest['test'] = {'path': str(test), 'bytes': test.stat().st_size, 'sha256': digest(test)}
manifest['cpu_records'] = records
manifest['complete'] = True
save()
print(json.dumps({'complete': True, 'manifest_sha256': digest(root / 'manifest.json'),
                  'frozen_root': str(root), 'evidence': str(evidence)}))
