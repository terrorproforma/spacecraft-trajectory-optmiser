"""Freeze and compile an isolated native upstream replay; never run GPU solves."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

live = Path('/mnt/c/Users/Angus/Desktop/projects/spacecraft-trajectory-optmiser')
root = live / 'build/performance/upstream-snapshot-v608'
root.mkdir(exist_ok=False)
source = root / 'source'
source.mkdir()
reference_build = Path('/home/angus/spacepdhcg-persistent-replay-v603/build/cuda')
reference_root = Path('/home/angus/spacecraft-trajectory-optmiser/_upstream/pdhcg')
reference_archive = reference_build / 'libspacepdhcg_pinned_reference.a'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def git(*args):
    return subprocess.check_output(['git', '-C', str(reference_root), *args], text=True).strip()

assert git('rev-parse', 'HEAD') == '167c8b72b4b96d2f94d405b8763e485514192b81'
assert git('rev-parse', 'HEAD^{tree}') == '62b05e6c1bedd385f6c267af3645ae4aae0421b4'
assert not git('status', '--porcelain=v1')
files = [
    'cpp/cuda/tests/upstream_snapshot_replay.cu', 'cpp/cuda/tests/persistent_snapshot.hpp',
    'cpp/cuda/include/spacepdhcg/cuda/persistent_pdhcg_c_api.h',
    'cpp/include/spacepdhcg/accelerator_c_api.h', 'cpp/cuda/CMakeLists.txt',
    'third_party/patches/pdhcg/0001-free-quadratic-state.patch', 'third_party/pdhcg.lock.json',
    'scripts/gpu/audit_persistent_snapshot.py',
]
for name in files:
    target = source / name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(live / name, target)
hashes = {name: sha(source / name) for name in files}
source_sha = hashlib.sha256(json.dumps(hashes, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
shutil.copyfile(__file__, root / 'build.py')
for name in ['mixed.txt', 'mixed-initial-original.txt', 'mixed-shifted.txt', 'mixed-shifted-initial-translated.txt']:
    (root / 'fixtures').mkdir(exist_ok=True)
    shutil.copyfile(live / 'build/performance/persistent-known-point-v606c/fixtures' / name, root / 'fixtures' / name)
for name in ['conditioning.txt', 'conditioning-initial.txt', 'difficult.txt', 'difficult-initial.txt']:
    shutil.copyfile(live / 'build/performance/known-point-replay-v606/inputs' / name, root / 'fixtures' / name)
assert sha(root / 'fixtures/conditioning.txt') == '1eb1b5a3979c5e86522337dfa94f264e354993163ed44129eff361371e3b49cf'
assert sha(root / 'fixtures/difficult.txt') == '14fdc84e9b654df4ac16db71f0bce4fccfa62577f9620ae622ad2196e2f17080'
binary = root / 'upstream_snapshot_replay'
command = ['g++', '-std=c++20', '-O3', '-Wall', '-Wextra', '-Wpedantic', '-Werror',
    '-x', 'c++', str(source / files[0]), '-x', 'none', str(reference_archive),
    '-I' + str(source / 'cpp/cuda/include'), '-I' + str(source / 'cpp/include'),
    '-I' + str(reference_build / 'pdhcg-patched/include'), '-I/usr/local/cuda-12.8/include',
    '-L/usr/local/cuda-12.8/lib64', '-Wl,-rpath,/usr/local/cuda-12.8/lib64',
    '-lcudart', '-lcublas', '-lcusolver', '-lcusparse', '-lz', '-lpthread', '-ldl',
    '-DSPACEPDHCG_SOURCE_COMMIT="e2b016492e0e288d42a40ea67e43fdfb4a57daab+replay-overlay"',
    '-DSPACEPDHCG_UPSTREAM_REPLAY_SOURCE_SHA256="' + source_sha + '"', '-o', str(binary)]
report = {'complete': False, 'gpu_calls': 0, 'source_sha256': source_sha,
          'source_files_sha256': hashes, 'command': command,
          'upstream_commit': git('rev-parse', 'HEAD'), 'upstream_tree': git('rev-parse', 'HEAD^{tree}'),
          'reference_archive': str(reference_archive), 'reference_archive_sha256': sha(reference_archive),
          'reference_cuda_architecture': 120,
          'fixtures_sha256': {p.name: sha(p) for p in (root / 'fixtures').iterdir()},
          'compiler': subprocess.check_output(['g++', '--version'], text=True)}
(root / 'manifest.json').write_text(json.dumps(report, indent=2) + '\n')
with (root / 'build.log').open('x') as log:
    subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=90)
report.update(complete=True, executable_sha256=sha(binary))
(root / 'manifest.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps({'complete': True, 'executable_sha256': report['executable_sha256'], 'source_sha256': source_sha}))
