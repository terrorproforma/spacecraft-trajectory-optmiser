"""Saved display history identity only; no import, ephemeris or propagation."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[3]
kit = root / 'build/performance/current-composition-visualizer-v628'
viewer = root / 'results/lambda/2026-09-06/visualiser'
target = viewer / 'data/gtoc12-current-composition-v628'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def read(p):
    return json.loads(p.read_text())


export, imported = read(kit / 'export-audit.json'), read(kit / 'import-audit.json')
assert export['status'] == imported['status'] == 'passed'
for name, expected in export['source_sha256'].items():
    assert sha(root / name) == expected
for name, expected in export['export_files'].items():
    assert sha(kit / 'export' / name) == expected
for name, expected in imported['input_sha256'].items():
    assert sha(viewer / name) == expected
for name, expected in imported['output_sha256'].items():
    assert sha(target / name) == expected
baseline = root / 'results/lambda/2026-09-09/gpu-regeneration-v799/h100-best/viewer'
replacement = root / 'build/performance/seeded-candidate-visualizer-v627'
old = read(baseline / 'trajectories.json')
new = read(replacement / 'export/trajectories.json')
raw = read(kit / 'export/trajectories.json')
assert read(baseline / 'manifest.json')['files']['trajectories.json']['sha256'] == sha(baseline / 'trajectories.json')
replacement_audit = read(replacement / 'export-audit.json')
assert replacement_audit['source_result_sha256'] == '931defc362c9299d28d089aa46130c05f92bf9ee38ea03d870b4dc2a83915946'
for name, expected in replacement_audit['source_sha256'].items():
    assert sha(root / name) == expected
assert replacement_audit['export_files']['trajectories.json'] == sha(replacement / 'export/trajectories.json')
assert len(raw['trajectories']) == len(old['trajectories']) == len(new['trajectories']) == 23
for i, ship in enumerate(raw['trajectories']):
    previous = (new if i == 7 else old)['trajectories'][i]
    for key in ('replay', 'transcription', 'events'):
        assert ship[key] == previous[key], (i, key)
fleet = read(target / 'fleet.json')
base_imported = read(viewer / 'data/gtoc12-regeneration-v799/fleet.json')
route_imported = read(viewer / 'data/gtoc12-seeded-candidate-v627/fleet.json')
assert len(fleet['ships']) == 23
for i, ship in enumerate(fleet['ships']):
    previous = (route_imported if i == 7 else base_imported)['ships'][i]
    for key in ('replay', 'transcription', 'events'):
        assert ship[key] == previous[key], (i, key)
result = '63446ebf3ff1298911bccade7b789a8fb68a0f7db4171906c8e6783a7f174bab'
binding = read(target / 'checker-binding.json')
original_binding = read(root / 'build/performance/seeded-current-composition-v628/output/checker-binding.json')
assert binding == original_binding
assert fleet['source']['solution_sha256'] == binding['result_sha256'] == export['source_result_sha256'] == result
assert fleet['verification']['ok'] and fleet['source']['official_verifier_ok']
assert fleet['score']['ships'] == 23 and fleet['score']['unique_asteroids'] == 200
assert abs(fleet['score']['total_collected_kg'] - binding['independent']['total_mass_kg']) <= 1e-8
assert fleet['score']['weighted_score_fixed_bonus_kg'] == binding['independent']['weighted_score_fixed_bonus_kg']
assert sum(len(s['events']) for s in fleet['ships']) == export['events'] == imported['summary']['events'] == 446
assert sum(t['replay']['point_count'] for t in raw['trajectories']) == export['replay_points'] == 11676
assert len(fleet['ships'][7]['events']) == 18 and fleet['ships'][7]['miners_deployed'] == fleet['ships'][7]['collects'] == 8
compute = read(target / 'compute.json')
assert compute['composition']['qualified'] and compute['composition']['other_ship_sections_exact'] == 22
assert all(compute['run_counts'][k] == 0 for k in ('native_solves_started', 'search_or_Lambert_calls', 'GPU_calls', 'extra_leg_certificates'))
assert compute['display_replay_provenance']['no_new_propagation']
summary = {'passed': True, 'scope': 'Saved history and file identities only; no display import, ephemeris, trajectory propagation or GPU work.',
           'result_sha256': result, 'fleet_sha256': sha(target / 'fleet.json'),
           'source_histories': {'v799': 22, 'v627': 1}, 'all_replay_transcription_and_events_equal': True,
           'ships': 23, 'asteroids': 200, 'events': 446, 'replay_points': 11676,
           'bound_original_checkers_pass': True, 'export_audit_sha256': sha(kit / 'export-audit.json'),
           'import_audit_sha256': sha(kit / 'import-audit.json'), 'verified_installed_files': imported['output_sha256'],
           'default_was_not_changed_by_this_audit': True}
path = Path(__file__).with_name('viewer-findings.json')
with path.open('x') as f:
    json.dump(summary, f, indent=2, allow_nan=False)
    f.write('\n')
print(json.dumps({'passed': True, 'findings_sha256': sha(path), 'fleet_sha256': summary['fleet_sha256']}))
