"""Archive the completed, verified local miner-recovery run without altering it."""
from pathlib import Path
import hashlib
import json
import shutil
import tarfile

repo = Path('/mnt/c/Users/Angus/Desktop/projects/spacecraft-trajectory-optmiser')
source = repo / 'build/performance/orphan-recovery-v595'
run = source / 'output-compatible'
destination = repo / 'results/local/2026-09-09/orphan-recovery-v595'
report = json.loads((run / 'report.json').read_text())
assert report['complete'] and report['status'] == 'improved'
assert report['best']['independent']['ok'] and report['best']['official']['ok']
assert report['best']['score_kg'] > report['baseline']['score_kg']
assert report['best']['total_mass_kg'] > report['baseline']['total_mass_kg']
destination.mkdir(parents=True, exist_ok=False)
for name in ('run.py', 'test_orders.py', 'source-sha256.json', 'compatible-launch.json',
             'compatible.log', 'snapshot-parity.json', 'snapshot-parity.log'):
    shutil.copy2(source / name, destination / name)
shutil.copy2(run / 'report.json', destination / 'report.json')
shutil.copy2(run / 'best/Result.txt', destination / 'Result.txt')
shutil.copy2(run / 'best/viewer/manifest.json', destination / 'viewer-export-manifest.json')
with tarfile.open(destination / 'raw.tar.gz', 'w:gz') as archive:
    for path in sorted(run.rglob('*')):
        if path.is_file():
            archive.add(path, arcname='output/' + path.relative_to(run).as_posix())
    for name in ('report.json',):
        archive.add(source / 'output' / name, arcname='failed-start/' + name)
    archive.add(source / 'run.log', arcname='failed-start/run.log')
with tarfile.open(destination / 'source.tar.gz', 'w:gz') as archive:
    for path in sorted((source / 'source').rglob('*')):
        if path.is_file() and '__pycache__' not in path.parts:
            archive.add(path, arcname='source/' + path.relative_to(source / 'source').as_posix())
digest = hashlib.sha256((destination / 'Result.txt').read_bytes()).hexdigest()
assert digest == report['best']['viewer']['source']['sha256']
manifest = {p.relative_to(destination).as_posix(): {
    'bytes': p.stat().st_size, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()
} for p in sorted(destination.rglob('*')) if p.is_file()}
(destination / 'sha256.json').write_text(json.dumps(manifest, indent=2) + '\n')
(destination / '.gitattributes').write_text('* -text\n')
print(json.dumps({'destination': str(destination), 'files': len(manifest),
                  'bytes': sum(p['bytes'] for p in manifest.values()), 'solution_sha256': digest}))
