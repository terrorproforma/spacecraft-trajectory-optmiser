"""Freeze the narrow JSON writer fix; compile and parse validate-only output without a GPU."""
from pathlib import Path
import hashlib
import json
import os
import shutil
import subprocess
import time

live = Path('/mnt/c/Users/Angus/Desktop/projects/spacecraft-trajectory-optmiser')
base = Path('/home/angus/spacepdhcg-persistent-known-point-v606b')
root = Path('/home/angus/spacepdhcg-persistent-known-point-v606c')
root.mkdir(exist_ok=False)
repo = root / 'repo'
repo.mkdir()
previous = json.loads((base / 'manifest.json').read_text())
digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
for directory in ('cpp', 'third_party'):
    shutil.copytree(base / 'repo' / directory, repo / directory)
for path in previous['owned_paths']:
    shutil.copy2(live / path, repo / path)
(root / 'build/cuda-tests').mkdir(parents=True)
shutil.copy2(__file__, root / 'build_known_point_v606c.py')
manifest = {'complete': False, 'base_frozen_source': str(base / 'repo'),
            'owned_paths': previous['owned_paths'], 'source_sha256': {}, 'stages': [],
            'immutable_core_path': previous['immutable_core_path'],
            'immutable_core_sha256': previous['immutable_core_sha256']}

def save():
    (root / 'manifest.json').write_text(json.dumps(manifest, indent=2))

env = {key: value for key, value in os.environ.items()
       if not key.startswith(('SPACEPDHCG_', 'QOCO_', 'PDHCG_', 'LD_LIBRARY_PATH'))}
env['CUDA_VISIBLE_DEVICES'] = ''

def call(name, command, cwd=None):
    start = time.perf_counter()
    with (root / (name + '.log')).open('x') as log:
        result = subprocess.run(command, cwd=cwd, env=env, stdout=log,
                                stderr=subprocess.STDOUT, timeout=90)
    manifest['stages'].append({'name': name, 'command': command, 'returncode': result.returncode,
                               'seconds': time.perf_counter() - start})
    save()
    assert result.returncode == 0, name

for name, command in [('git-init', ['git', 'init']), ('git-add', ['git', 'add', '-f', '.']),
                      ('git-freeze', ['git', '-c', 'user.name=Replay validation', '-c',
                                      'user.email=replay-validation@localhost', 'commit', '-m',
                                      'Freeze unambiguous JSON string writer for known-point replay'])]:
    call(name, command, repo)
manifest['frozen_commit'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip()
manifest['source_sha256'] = {p.relative_to(repo).as_posix(): digest(p)
                              for p in (repo / 'cpp').rglob('*') if p.is_file()}
parts = ['tests/persistent_snapshot.hpp', 'tests/persistent_snapshot_replay.cu',
         'tests/cuda_test_support.hpp', 'include/spacepdhcg/cuda/persistent_pdhcg_c_api.h']
manifest['compiled_snapshot_source_sha256'] = hashlib.sha256(''.join(
    path + ':' + digest(repo / 'cpp/cuda' / path) + '\n' for path in parts).encode()).hexdigest()
assert digest(Path(manifest['immutable_core_path'])) == manifest['immutable_core_sha256']
for stage in previous['stages']:
    if stage['name'] in ('cpu-build', 'cpu-test', 'cuda-compile', 'link'):
        command = [arg.replace(str(base), str(root)).replace(previous['frozen_commit'], manifest['frozen_commit'])
                   .replace(previous['compiled_snapshot_source_sha256'], manifest['compiled_snapshot_source_sha256'])
                   for arg in stage['command']]
        call(stage['name'], command)
binary = root / 'build/cuda-tests/persistent_snapshot_replay'
manifest['executable_sha256'] = digest(binary)
validation = {'complete': False, 'cuda_visible_devices': '', 'cases': []}
cases = [
    ('unseeded-control', 'mixed.txt', None, False),
    ('original-generic', 'mixed.txt', 'mixed-initial-original.txt', False),
    ('translated-folded', 'mixed.txt', 'mixed-initial-translated.txt', True),
    ('shifted-original-generic', 'mixed-shifted.txt', 'mixed-shifted-initial-original.txt', False),
    ('shifted-translated-folded', 'mixed-shifted.txt', 'mixed-shifted-initial-translated.txt', True),
    ('weak-bound-folded', 'bounds-duplicate.txt', 'bounds-weak-initial.txt', True),
]
resolved = [(name, root / 'fixtures' / snap, root / 'fixtures' / point if point else None, folded)
            for name, snap, point, folded in cases]
inputs = live / 'build/performance/known-point-replay-v606/inputs'
for name in ('conditioning', 'difficult'):
    for folded in (False, True):
        resolved.append((name + ('-folded' if folded else '-generic'), inputs / (name + '.txt'),
                         inputs / (name + '-initial.txt'), folded))
for name, snapshot, point, folded in resolved:
    command = [str(binary), str(snapshot), '--validate-only']
    if point:
        command.extend(['--initial-point', str(point)])
    if folded:
        command.append('--fold-singleton-bounds')
    stage = name + '-validate-no-gpu'
    call(stage, command)
    records = {}
    for line in (root / (stage + '.log')).read_text().splitlines():
        prefix, payload = line.split(' ', 1)
        assert prefix not in records
        records[prefix] = json.loads(payload)
    assert set(records) == ({'PERSISTENT_REPLAY_META', 'PERSISTENT_REPLAY_INITIAL_POINT'} if point else {'PERSISTENT_REPLAY_META'})
    meta = records['PERSISTENT_REPLAY_META']
    assert meta['validate_only'] and meta['input_sha256'] == digest(snapshot)
    assert meta['source_sha256'] == manifest['compiled_snapshot_source_sha256']
    assert meta['initial_point_supplied'] == bool(point)
    assert meta['fold_singleton_bounds'] == folded
    assert 'includes_bootstrap_and_prestep_output' in meta['repeat_seconds_scope']
    assert 'including_bootstrap_and_prestep_output' in meta['setup_seconds_scope']
    if point:
        initial = records['PERSISTENT_REPLAY_INITIAL_POINT']
        assert meta['initial_point_sha256'] == initial['point_sha256'] == digest(point)
        assert meta['initial_point_coordinates'] in ('original', 'translated')
        assert initial['supplied_qualified'] and initial['mapped_reference_qualified']
    validation['cases'].append({'name': name, 'snapshot_sha256': digest(snapshot),
                                'point_sha256': digest(point) if point else None,
                                'record_prefixes': list(records), 'all_json_records_parsed': True})
validation['complete'] = True
(root / 'json-validation.json').write_text(json.dumps(validation, indent=2))
subprocess.run(['git', 'archive', '--format=tar.gz', '-o', str(root / 'source.tar.gz'), 'HEAD'], cwd=repo, check=True)
manifest['complete'] = True
save()
destination = live / 'build/performance/persistent-known-point-v606c'
destination.mkdir(exist_ok=False)
for path in root.iterdir():
    if path.is_file():
        shutil.copy2(path, destination / path.name)
shutil.copytree(root / 'fixtures', destination / 'fixtures')
shutil.copytree(base / 'tiny', destination / 'initial-v606b')
print(json.dumps({key: manifest[key] for key in ('frozen_commit', 'compiled_snapshot_source_sha256', 'executable_sha256', 'immutable_core_sha256')}))
