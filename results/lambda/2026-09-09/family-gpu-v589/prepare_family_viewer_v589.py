from pathlib import Path
import json

root = Path('results/lambda/2026-09-09/family-gpu-v589')
output = root / 'output'
report = json.loads((root / 'report.json').read_text())
fleet = json.loads((output / 'fleet/fleet.json').read_text())
assert fleet['ok'] and fleet['official']['ok'] and fleet['independent']['ok']
fleet['viewer_manifest'] = json.loads((output / 'fleet/viewer/manifest.json').read_text())
Path('build/performance/family-viewer-fleet.json').write_text(json.dumps(fleet, indent=2))
destination = Path('results/lambda/2026-09-06/visualiser/data/gtoc12-family-v589')
destination.mkdir(exist_ok=False)
metadata = dict(run_id='gpu_family32_rich_v589_h100', fleet_run_id='gpu_family32_rich_v589_h100_fleet',
    commit=json.loads((output / 'fleet/viewer/trajectories.json').read_text())['generated_by_commit'],
    source_revision_note='Export ran from a frozen directory without Git metadata. Base a89ed69f plus the weighted cluster incumbent correction; exact source and binary hashes in family-gpu-v589/report.json.',
    weighted_score_fixed_bonus_kg=fleet['weighted_score_fixed_bonus_kg'],
    hardware=dict(gpu='Lambda NVIDIA H100 80 GB'),
    timing=dict(wall_seconds_total=report['seconds'], wall_human=f"{report['seconds']:.2f} s richer family pilot; uncontrolled timing"),
    model=dict(dynamics='Official GTOC12 low-thrust dynamics; official and independent verification passed',
               local_refine='CUDA SCvx, QOCO, assembly and discretisation; GPU Lambert screening; zero Ruiz'),
    optimisation=dict(strategy='One family, three ships, eight in-memory route columns. First ship reproduces the historical 641.068 kg route. Best fleet remains 12,805.194 weighted kg. Python orchestration and CPU fleet master remain.', proven_optimal=False))
(destination / 'compute.json').write_text(json.dumps(metadata, indent=2))
print(destination)
