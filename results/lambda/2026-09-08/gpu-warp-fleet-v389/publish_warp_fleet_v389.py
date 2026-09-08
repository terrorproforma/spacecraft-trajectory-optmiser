from pathlib import Path
import hashlib
import json
import shutil
import tarfile

source = Path('build/performance/warp-fleet-v389')
report = json.loads((source / 'report.json').read_text())
assert report['complete'] and not report.get('error')
campaigns = {}
for name in ['candidate0', 'baseline0']:
    run = json.loads((source / name / 'output/run_report.json').read_text())
    check = run['best']['independent']
    assert check['ok'] and run['best']['official']['ok']
    assert run['fleet']['ships'] == 4
    campaigns[name] = dict(seconds=run['wall_seconds_total'], screening=run['screening'],
        official=run['best']['official'], independent=check)
assert campaigns['candidate0']['screening'] == campaigns['baseline0']['screening']
assert abs(campaigns['candidate0']['independent']['weighted_score_fixed_bonus_kg'] -
           campaigns['baseline0']['independent']['weighted_score_fixed_bonus_kg']) < 1e-8
root = Path('results/lambda/2026-09-08/gpu-warp-fleet-v389')
root.mkdir(exist_ok=False)
manifest = {p.relative_to(source).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(source.rglob('*')) if p.is_file()}
with tarfile.open(root / 'raw.tar.gz', 'w:gz') as archive:
    for name in manifest:
        archive.add(source / name, arcname=name, recursive=False)
(root / 'archive-manifest.json').write_text(json.dumps(manifest, indent=2))
baseline = campaigns['baseline0']['seconds']
candidate = campaigns['candidate0']['seconds']
summary = dict(hardware='Local NVIDIA RTX 5090', complete=True, campaigns=campaigns,
    speedup=baseline/candidate, time_reduction_percent=100*(1-candidate/baseline),
    same_search_counts=True, all_mission_checkers_pass=True,
    scope='One complete four-ship campaign per mode, candidate then baseline. These are end-to-end times, not isolated GPU kernel throughput. No universal speedup claim; fleet incumbent remains 12,805.194 weighted kg.')
(root / 'summary.json').write_text(json.dumps(summary, indent=2))
for name in ['run_warp_fleet_v389.py', 'prepare_warp_fleet_v389.py', 'publish_warp_fleet_v389.py']:
    shutil.copy2(Path('build/performance') / name, root / name)
(root / '.gitattributes').write_text('* -text\n')
files = {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
         for p in root.rglob('*') if p.is_file()}
(root / 'files-sha256.json').write_text(json.dumps(files, indent=2))
print(json.dumps(dict(speedup=summary['speedup'], archived_files=len(manifest),
                     bytes=(root/'raw.tar.gz').stat().st_size)))
