"""Fresh normal CMake build from committed 7eb8828f plus the four reviewed completion files.

Read-only Git archive/identity queries are permitted; no Git mutations or GPU tests.
Host Python runtime is deliberately not in this C++ source fingerprint.
"""
from pathlib import Path
import hashlib
import io
import json
import os
import shutil
import subprocess
import tarfile
import time

live = Path('/mnt/c/Users/Angus/Desktop/projects/spacecraft-trajectory-optmiser')
root = Path('/home/angus/spacepdhcg-completion-fullcore-v621')
evidence = live / 'build/performance/completion-fullcore-v621'
standalone = live / 'build/performance/completion-native-core-v621'
root.mkdir(exist_ok=False)
evidence.mkdir(exist_ok=False)
repo = root / 'repo'
repo.mkdir()
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
base = '7eb8828f61bd35ec0abaf98c7651c5e82384ae27'
previous = json.loads((standalone / 'manifest.json').read_text())
assert sha(standalone / 'manifest.json') == 'b85ab26f26db5e71f7a4c7f7a31ef24b6b01e83cffe81731d674db6615f09c08'
owned = list(previous['source_sha256'])
git = ['/usr/bin/git', '-c', 'safe.directory=' + str(live), '-C', str(live)]
env = {k: v for k, v in os.environ.items()
       if not k.startswith(('SPACEPDHCG_', 'QOCO_', 'PDHCG_', 'LD_LIBRARY_PATH'))}
env['CUDA_VISIBLE_DEVICES'] = ''
resolved = subprocess.check_output(git + ['rev-parse', base], env=env, text=True).strip()
assert resolved == base
base_tree = subprocess.check_output(git + ['rev-parse', base + '^{tree}'], env=env, text=True).strip()
base_archive = subprocess.check_output(git + ['archive', '--format=tar', base, 'cpp', 'third_party'], env=env)
with tarfile.open(fileobj=io.BytesIO(base_archive), mode='r:') as archive:
    for member in archive.getmembers():
        path = Path(member.name)
        assert not path.is_absolute() and '..' not in path.parts and (member.isfile() or member.isdir())
        if member.isdir():
            (repo / path).mkdir(parents=True, exist_ok=True)
        else:
            (repo / path).parent.mkdir(parents=True, exist_ok=True)
            (repo / path).write_bytes(archive.extractfile(member).read())
base_sha = {p.relative_to(repo).as_posix(): sha(p) for p in sorted(repo.rglob('*')) if p.is_file()}
for name in owned:
    assert sha(live / name) == previous['source_sha256'][name]
    target = repo / name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(live / name, target)
shutil.copy2(__file__, root / Path(__file__).name)
source_sha = {p.relative_to(repo).as_posix(): sha(p) for p in sorted(repo.rglob('*')) if p.is_file()}
assert all(source_sha[p] == h for p, h in base_sha.items() if p not in owned)
assert set(source_sha) - set(base_sha) == set(owned) - set(base_sha)
tree_sha = hashlib.sha256(''.join(k + ':' + v + '\n' for k, v in source_sha.items()).encode()).hexdigest()
manifest = {
    'complete': False, 'gpu_calls': 0, 'git_mutations': 0,
    'git_readonly_operations': ['rev-parse commit', 'rev-parse tree', 'archive cpp third_party',
                                'CMake pinned third-party identity/cleanliness checks'],
    'base_commit': base, 'base_git_tree': base_tree,
    'base_archive_sha256': hashlib.sha256(base_archive).hexdigest(),
    'source_identity_kind': 'sha256_tree_not_git_commit', 'frozen_commit': None,
    'compiled_source_commit': 'uncommitted', 'source_dirty': True,
    'source_identity_scope': 'Frozen C++/CUDA/CMake and third-party build sources; excludes host Python runtime.',
    'host_python_runtime': 'Not frozen by this build. Root must bind its separate final adapter snapshot.',
    'owned_overlay_paths': owned, 'owned_overlay_sha256': previous['source_sha256'],
    'nonowned_source': 'Exact committed 7eb8828f bytes; no nonowned working-tree file copied.',
    'source_sha256': source_sha, 'source_tree_sha256': tree_sha,
    'base_source_sha256': base_sha, 'reused_compiled_objects': [],
    'build_mode': 'Fresh normal project CMake/Ninja build; no cache object reuse.',
    'cuda_architectures': [120], 'stages': [],
}
with tarfile.open(root / 'source.tar.gz', 'w:gz') as archive:
    for name in source_sha:
        archive.add(repo / name, arcname=name, recursive=False)
manifest['source_archive_sha256'] = sha(root / 'source.tar.gz')

def save():
    (root / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    for p in root.iterdir():
        if p.is_file() and p.suffix in ('.json', '.log', '.py', '.gz'):
            shutil.copy2(p, evidence / p.name)

def call(name, command, timeout=600):
    start = time.perf_counter()
    with (root / (name + '.log')).open('x') as log:
        result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                timeout=timeout, cwd=repo)
    manifest['stages'].append({'name': name, 'command': command,
                              'returncode': result.returncode,
                              'seconds': time.perf_counter() - start})
    save()
    if result.returncode:
        raise RuntimeError(name + ': ' + (root / (name + '.log')).read_text()[-8000:])

save()
cache_path = Path('/home/angus/spacepdhcg-persistent-replay-v603/build/CMakeCache.txt')
cache = dict(line.split('=', 1) for line in cache_path.read_text().splitlines()
             if '=' in line and not line.startswith(('#', '//')))
cmake = cache['CMAKE_COMMAND:INTERNAL']
ctest = cache['CMAKE_CTEST_COMMAND:INTERNAL']
pinned = next(v for k, v in cache.items() if k.startswith('SPACEPDHCG_PDHCG_SOURCE_ROOT:'))
manifest['cmake_tool_source_cache'] = {'path': str(cache_path), 'sha256': sha(cache_path)}
manifest['pinned_upstream_checkout'] = pinned
build = root / 'build'
call('compiler-version', ['/usr/local/cuda-12.8/bin/nvcc', '--version'])
call('cmake-configure', [cmake, '-S', str(repo / 'cpp'), '-B', str(build), '-G', 'Ninja',
    '-DSPACEPDHCG_BUILD_CUDA=ON', '-DCMAKE_BUILD_TYPE=Release',
    '-DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc', '-DCMAKE_CUDA_ARCHITECTURES=120',
    '-DSPACEPDHCG_PDHCG_SOURCE_ROOT=' + pinned,
    '-DSPACEPDHCG_FROZEN_SOURCE_TREE_SHA256=' + tree_sha,
    '-DSPACEPDHCG_FROZEN_SOURCE_BASE_COMMIT=' + base, '-DCMAKE_EXPORT_COMPILE_COMMANDS=ON'])
call('fullcore-build', [cmake, '--build', str(build), '--target',
    'spacepdhcg_cuda', 'gtoc12_completion_test', 'persistent_snapshot_conversion_test', '-j', '4'])
library = build / 'cuda/libspacepdhcg_cuda.so'
test = build / 'cuda-tests/gtoc12_completion_test'
conversion = build / 'cuda/persistent_snapshot_conversion_test'
call('completion-cpu-fixture-check', [str(test), '--cpu-only'])
call('snapshot-cpu-conversion-check', [str(conversion)])
call('ctest-enumerate', [ctest, '--test-dir', str(build), '-N', '-V', '-R', 'gtoc12_completion'])
assert 'gtoc12_completion_test' in (root / 'ctest-enumerate.log').read_text()
call('resource-usage', ['/usr/local/cuda-12.8/bin/cuobjdump', '--dump-resource-usage', str(library)])
call('dynamic-symbols', ['/usr/bin/nm', '-D', '--defined-only', str(library)])
symbols = (root / 'dynamic-symbols.log').read_text()
for symbol in ('spacepdhcg_gtoc12_completion_create', 'spacepdhcg_gtoc12_completion_evaluate_host',
               'spacepdhcg_gtoc12_completion_destroy', 'spacepdhcg_orbitweaver_lambert_workspace_create'):
    assert symbol in symbols, symbol
for name, path in [('library', library), ('test', test), ('conversion_test', conversion)]:
    manifest[name] = {'path': str(path), 'bytes': path.stat().st_size, 'sha256': sha(path)}
manifest['compile_commands_sha256'] = sha(build / 'compile_commands.json')
shutil.copy2(build / 'compile_commands.json', root / 'compile-commands.json')
manifest['object_sha256'] = {p.relative_to(build).as_posix(): sha(p)
                           for p in sorted(build.rglob('*.o'))}
manifest['complete'] = True
save()
print(json.dumps({'complete': True, 'manifest_sha256': sha(root / 'manifest.json'),
                  'frozen_root': str(root), 'library_sha256': manifest['library']['sha256']}))
