from pathlib import Path
import json, hashlib, shutil

root = Path('results/lambda/2026-09-08/gpu-native-campaign-v269')
run = json.loads((root/'v269/output/run_report.json').read_text())
best = run['best']; ship = run['ships'][0]
export = json.loads((root/'v269/output/fleet/viewer/trajectories.json').read_text())
def write(path, value):
    path.write_text(json.dumps(value, indent=2)+'\n')
write(root/'viewer-input.json', dict(fleet=run['fleet'], official=best['official'], independent=best['independent'], viewer_manifest=best['viewer_manifest']))
out = Path('results/lambda/2026-09-06/visualiser/data/gtoc12-v269')
out.mkdir(exist_ok=True)
mass = best['independent']['total_mass_kg']; seconds = run['wall_seconds_total']
run_id = export['trajectories'][0]['source']['run_id']
meta = dict(run_id=run_id, fleet_run_id=run_id, commit=export['generated_by_commit'],
    source_code_commit='58eefa56', source_revision_note='Exporter revision is the temporary Lambda source snapshot; published implementation is 58eefa56, with source hashes in the campaign report.',
    weighted_score_fixed_bonus_kg=mass, raw_kg_per_ship=mass,
    hardware=dict(gpu='Lambda NVIDIA H100 80 GB', upstream_search='44,722,546 CUDA transfer branches; 2,782,091 collection options; zero host retiming table uploads'),
    timing=dict(wall_seconds_total=seconds, wall_human=f'{seconds:.2f} s complete search, refinement and retiming'),
    model=dict(dynamics='Official GTOC12 low-thrust dynamics; official and independent verification pass', local_refine='16 CUDA-refined transfer arcs plus a camp interval; eight mining asteroids'),
    optimisation=dict(strategy='Device-controlled retiming and route extension; 21.766 kg above the previous development mission. The 23-ship incumbent fleet is unchanged.', proven_optimal=False))
write(out/'compute.json', meta)
summary = dict(run_id=run['run_id'], source_code_commit='58eefa56', exporter_snapshot=export['generated_by_commit'],
    wall_seconds_total=seconds, search_seconds=ship['search']['wall_seconds'], screening=run['screening'],
    branch_requests_per_end_to_end_second=run['screening']['completed_branch_requests']/seconds,
    candidate_routes=ship['search']['candidates'], initial_refinements=len(ship['refinements']),
    initial_certified=sum(bool(x.get('official',{}).get('ok')) for x in ship['refinements']),
    retiming_attempts=len(ship['retiming']['attempts']), retiming_seconds=ship['retiming']['wall_seconds'],
    final_certified_candidates=run['master']['columns'], selected_ships=run['fleet']['ships'],
    best_independent=best['independent'], best_official=best['official'],
    development_improvement_kg=mass-526.488706365503,
    incumbent_weighted_score_kg=12805.194102488575,
    scope='One-ship development campaign; three certified columns include an improved version of the original route. Counters are evaluations, not unique complete missions. No fleet score change or globally optimal claim.')
write(root/'summary.json', summary)
viewer = out.parents[1]
for name, old, new in [
    ('app.js', '  "gtoc12-v235":', '  "gtoc12-v269": { directory: "./data/gtoc12-v269", label: "GPU campaign v269 (548 kg certified)" },\n  "gtoc12-v235":'),
    ('index.html', '            <option value="gtoc12-v235"', '            <option value="gtoc12-v269" disabled>GPU campaign v269 — checking…</option>\n            <option value="gtoc12-v235"'),
    ('scripts/check.mjs', '"data/gtoc12-v235"]', '"data/gtoc12-v235", "data/gtoc12-v269"]')]:
    path=viewer/name; text=path.read_text()
    if 'gtoc12-v269' not in text:
        assert old in text
        path.write_text(text.replace(old,new,1))
print(json.dumps(summary,indent=2))
