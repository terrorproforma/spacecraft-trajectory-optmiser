"""Assemble completed completion evidence without sealing; no GPU/Git/network work."""
from pathlib import Path
import hashlib
import json
import shutil
import tarfile

ROOT = Path(__file__).resolve().parents[2]
SCRATCH = ROOT / 'build/performance/completion-package-v621'
DEST = ROOT / 'results/local/2026-09-09/gpu-route-completion-v621'
DEST.mkdir(parents=True, exist_ok=False)
OBJECTS = SCRATCH / 'source-objects'
OBJECTS.mkdir(exist_ok=False)
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
origins = {}
snapshots = {}


def copy_file(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    assert not destination.exists(), destination
    assert source.suffix.lower() not in {'.pyc', '.so', '.dll', '.exe', '.o', '.a', '.pem', '.key'}
    shutil.copy2(source, destination)
    assert sha(source) == sha(destination)
    origins[destination.relative_to(DEST).as_posix()] = {
        'source': source.relative_to(ROOT).as_posix(), 'bytes': source.stat().st_size, 'sha256': sha(source)}


def tree(source, destination, exclude=()):
    for path in sorted(source.rglob('*')):
        if not path.is_file():
            continue
        relative = path.relative_to(source)
        if any(part in {'__pycache__', '.pytest_cache', '.ruff_cache', 'pytest-tmp'} for part in relative.parts):
            continue
        if any(relative == Path(x) or Path(x) in relative.parents for x in exclude):
            continue
        copy_file(path, destination / relative)


def source_snapshot(name, source, mapping, evidence_path):
    rows = {}
    for relative, digest in mapping.items():
        path = source / relative
        assert path.suffix.lower() not in {'.pyc', '.so', '.dll', '.exe', '.o', '.a', '.pem', '.key'}
        assert path.is_file() and sha(path) == digest, path
        blob = OBJECTS / digest
        if not blob.exists():
            shutil.copy2(path, blob)
        assert sha(blob) == digest
        rows[relative] = {'sha256': digest, 'bytes': path.stat().st_size}
    snapshots[name] = {'source_root': source.relative_to(ROOT).as_posix(),
                       'original_manifest': evidence_path, 'files': rows}


for src, dst in (
    ('completion-native-core-v621', 'build/standalone'),
    ('completion-fullcore-v621', 'build/fullcore'),
    ('completion-native-controls-v621', 'controls'),
    ('completion-adapter-cpu-v621a', 'python/cpu-attempt-a'),
    ('completion-adapter-cpu-v621b', 'python/cpu-attempt-b'),
    ('completion-integration-review-v621', 'reviews/orchestration'),
    ('mass-structure-review-v621', 'reviews/mass-design'),
):
    tree(ROOT / 'build/performance' / src, DEST / dst, exclude=('source',))

# Large adapter call readbacks are compressed losslessly, with their original
# relative paths and hashes. All supervisor/pytest reports remain plain text.
adapter = ROOT / 'build/performance/completion-adapter-gpu-v621'
readbacks = sorted((adapter / 'gpu-output/child').glob('call-*.json'))
assert len(readbacks) == 16
tree(adapter, DEST / 'runs/adapter', exclude=tuple(p.relative_to(adapter).as_posix() for p in readbacks))
members = {}
with tarfile.open(DEST / 'runs/adapter/readbacks.tar.gz', 'w:gz') as archive:
    for path in readbacks:
        name = path.relative_to(adapter).as_posix()
        archive.add(path, arcname=name, recursive=False)
        members[name] = {'bytes': path.stat().st_size, 'sha256': sha(path)}
(DEST / 'runs/adapter/readbacks.json').write_text(json.dumps(members, indent=2) + '\n')

controls = ROOT / 'build/performance/completion-native-controls-v621'
source_snapshot('controls', controls / 'source',
                json.loads((controls / 'source-sha256.json').read_text()), 'controls/source-sha256.json')
for label in ('a', 'b'):
    source = ROOT / f'build/performance/completion-adapter-cpu-v621{label}'
    source_snapshot('cpu-attempt-' + label, source / 'source',
                    json.loads((source / 'report.json').read_text())['source_sha256'],
                    f'python/cpu-attempt-{label}/report.json')

# Preserve only the two older C++ source members actually accessed by the
# coefficient-only mass audit, alongside their unchanged original manifest.
prior = ROOT / 'build/performance/l1-weight-v618b'
copy_file(prior / 'manifest.json', DEST / 'reviews/mass-design/prior-manifest.json')
prior_map = json.loads((prior / 'manifest.json').read_text())['source_sha256']
mass_source = {}
with tarfile.open(prior / 'source.tar.gz', 'r:gz') as archive:
    for name in ('cpp/cuda/src/gtoc12_conic.cu', 'cpp/cuda/src/gtoc12_discretisation.cu'):
        data = archive.extractfile(name).read()
        digest = hashlib.sha256(data).hexdigest()
        assert digest == prior_map[name]
        blob = OBJECTS / digest
        if not blob.exists():
            blob.write_bytes(data)
        mass_source[name] = {'bytes': len(data), 'sha256': digest}
snapshots['mass-source-subset'] = {'source_root': 'Prior frozen v618b archive: only two accessed members',
                                  'original_manifest': 'reviews/mass-design/prior-manifest.json',
                                  'files': mass_source}
for name in ('conditioning', 'difficult'):
    copy_file(ROOT / f'build/performance/known-point-replay-v606/inputs/{name}.txt',
              DEST / f'reviews/mass-design/inputs/{name}.txt')

(DEST / 'python/source-snapshots.json').write_text(json.dumps(snapshots, indent=2) + '\n')
(SCRATCH / 'copy-origins.json').write_text(json.dumps(origins, indent=2) + '\n')
(DEST / '.gitattributes').write_text('* -text\n')
(DEST / 'STATUS.json').write_text(json.dumps({
    'state': 'assembling', 'sealed': False,
    'pending': ['separate bounded benchmark terminal result', 'final independent saved-output reviews',
                'root-owned final documentation', 'source object archive and portable rechecks', 'final index'],
    'native_validation': {'passed': True, 'kernels': 3, 'candidate_evaluations': 304},
    'adapter_validation': {'passed': True, 'evaluation_calls': 8, 'candidate_evaluations': 1048, 'lambert_requests': 0},
}, indent=2) + '\n')
print(json.dumps({'package': str(DEST), 'sealed': False, 'copied_files': len(origins),
                  'source_snapshots': len(snapshots), 'unique_source_blobs': len(list(OBJECTS.iterdir()))}))
