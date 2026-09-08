"""Compile the exact-zero-Q dispatch ablation with the already-pinned reference."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

live = Path(__file__).resolve().parents[2]
base = live / 'build/performance/upstream-snapshot-v608'
root = live / 'build/performance/upstream-zero-q-v613'
root.mkdir(exist_ok=False)
old = json.loads((base / 'manifest.json').read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
assert old['complete']
assert sha(Path(old['reference_archive'])) == old['reference_archive_sha256']
for name, expected in old['source_files_sha256'].items():
    assert sha(base / 'source' / name) == expected, name
shutil.copytree(base / 'source', root / 'source')
shutil.copytree(base / 'fixtures', root / 'fixtures')
name = 'cpp/cuda/tests/upstream_snapshot_replay.cu'
shutil.copyfile(live / name, root / 'source' / name)
shutil.copyfile(__file__, root / 'build.py')
hashes = {name: sha(root / 'source' / name) for name in old['source_files_sha256']}
source_sha = hashlib.sha256(json.dumps(hashes, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
command = [part.replace(str(base), str(root)) for part in old['command']]
command = [('-DSPACEPDHCG_SOURCE_COMMIT="f8b2ac7aeba9e29ea90f5814060bb8cfe5db22d7+zero-Q-diagnostic"' if part.startswith('-DSPACEPDHCG_SOURCE_COMMIT=') else
            '-DSPACEPDHCG_UPSTREAM_REPLAY_SOURCE_SHA256="' + source_sha + '"' if part.startswith('-DSPACEPDHCG_UPSTREAM_REPLAY_SOURCE_SHA256=') else part) for part in command]
env = dict(os.environ, CUDA_VISIBLE_DEVICES='')
report = old | {'complete': False, 'command': command, 'source_files_sha256': hashes, 'source_sha256': source_sha,
                'base_manifest_sha256': sha(base / 'manifest.json'), 'gpu_calls': 0, 'cpu_checks': [],
                'scope': 'exact zero Hessian omission only; retained objective, constraints and original-coordinate audit'}
def save():
    (root / 'manifest.json').write_text(json.dumps(report, indent=2) + '\n')
save()
with (root / 'build.log').open('x') as log:
    subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=90)
binary = root / 'upstream_snapshot_replay'
for name, expected, flags in [('conditioning', 0, []), ('difficult', 0, []), ('mixed', 2, []),
                             ('mixed-shifted', 2, []), ('conditioning-duplicate', 2, ['--omit-zero-quadratic'])]:
    snapshot_name = 'conditioning' if name.endswith('-duplicate') else name
    args = [str(binary), str(root / 'fixtures' / (snapshot_name + '.txt')), '--validate-only', '--omit-zero-quadratic'] + flags
    with (root / ('validate-' + name + '.log')).open('x') as log:
        run = subprocess.run(args, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=15)
    report['cpu_checks'].append({'name': name, 'command': args, 'returncode': run.returncode, 'expected': expected})
    save()
    assert run.returncode == expected
report.update(complete=True, executable_sha256=sha(binary))
save()
print(json.dumps({'complete': True, 'executable_sha256': report['executable_sha256'], 'source_sha256': source_sha, 'cpu_checks': len(report['cpu_checks'])}))
