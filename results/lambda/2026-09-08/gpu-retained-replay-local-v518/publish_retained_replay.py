from pathlib import Path
import hashlib
import json
import shutil
import tarfile

p = Path('build/performance')
base = Path('results/lambda/2026-09-08')
target = base / 'gpu-retained-replay-local-v518'
assert not target.exists()
def read(folder, name='report.json'):
    return json.loads((p / folder / name).read_text())

for folder in ['retained-replay-v515', 'retrieved-retained-replay-v514']:
    r = read(folder)
    assert r['complete'] and not r.get('error')
    assert '107 passed' in (p / folder / 'pytest.log').read_text()
    for source, sha in r['source_sha256'].items():
        assert hashlib.sha256(Path(source).read_bytes()).hexdigest() == sha, source
for folder in ['retained-replay-legs-v518', 'retrieved-retained-replay-legs-v517']:
    result = read(folder, 'analysis.json')
    assert not result['lost_baseline']
    assert result['baseline']['converged'] == result['candidate']['converged'] == 205
    assert result['max_certified_mass_delta']['delta_kg'] < 1e-5
for folder in ['retained-replay-campaign-v513', 'retrieved-retained-replay-v514']:
    assert len(read(folder, 'analysis.json')['rows']) == 4

target.mkdir()
archives = {}
for name in ['retained-replay-v511', 'retained-replay-v512',
             'retained-replay-campaign-v513', 'retained-replay-v515',
             'retained-replay-legs-v516', 'retained-replay-legs-v518']:
    root = p / name
    r = read(name)
    if name.endswith(('v511', 'v516')):
        assert not r['complete'] and r.get('error')
    else:
        assert r['complete'] and not r.get('error')
    members = {q.relative_to(root).as_posix(): hashlib.sha256(q.read_bytes()).hexdigest()
               for q in sorted(root.rglob('*')) if q.is_file()}
    archive_name = name + '.tar.gz'
    with tarfile.open(target / archive_name, 'w:gz') as archive:
        for member in members:
            archive.add(root / member, arcname=member, recursive=False)
    archives[archive_name] = members
(target / 'archive-manifests.json').write_text(json.dumps(archives, indent=2))
summary = dict(
    default_enabled=False,
    source_base_commit='dd9917d12a2e7d3c4b5bab4be2d417cf4f64aa60',
    local_campaign=read('retained-replay-campaign-v513', 'analysis.json'),
    lambda_campaign=read('retrieved-retained-replay-v514', 'analysis.json'),
    local_legs=read('retained-replay-legs-v518', 'analysis.json'),
    lambda_legs=read('retrieved-retained-replay-legs-v517', 'analysis.json'),
    limitations=[
        'Opt-in only; small complete-campaign differences have overlapping observations.',
        'Adapter transfer fields include lifetime counters from retained subobjects. They are not per-leg byte totals.',
        'v511 failed an incorrect byte-counter assertion; v516 failed before solving while reading a report being written. Both are retained.',
        'All acceptance tolerances are unchanged; this does not establish full GPU control or a new fleet score.',
    ])
(target / 'summary.json').write_text(json.dumps(summary, indent=2))
for pattern in ['*retained*replay*.py', 'analyze_retained_replay*.py']:
    for q in p.glob(pattern):
        shutil.copy2(q, target / q.name)
folders = [target]
for tag in ['retained-replay-v514', 'retained-replay-legs-v517']:
    root = base / ('gpu-' + tag)
    folders.append(root)
    for name in ['report.json', 'analysis.json']:
        shutil.copy2(p / ('retrieved-' + tag) / name, root / name)
for root in folders:
    (root / '.gitattributes').write_text('* -text whitespace=cr-at-eol,-blank-at-eof\n')
    manifest = {q.relative_to(root).as_posix(): hashlib.sha256(q.read_bytes()).hexdigest()
                for q in sorted(root.rglob('*')) if q.is_file() and q.name != 'files-sha256.json'}
    assert all((root / name).stat().st_size < 90_000_000 for name in manifest)
    (root / 'files-sha256.json').write_text(json.dumps(manifest, indent=2))
    print(root, len(manifest), 'files')
