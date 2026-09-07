from pathlib import Path
import json
root=Path('results/lambda/2026-09-08/gpu-execution-v328')
run=json.loads((root/'v332/output/run_report.json').read_text());best=run['best']
(root/'viewer-input.json').write_text(json.dumps(dict(fleet=run['fleet'],official=best['official'],independent=best['independent'],viewer_manifest=best['viewer_manifest']),indent=2))
viewer=Path('results/lambda/2026-09-06/visualiser');out=viewer/'data/gtoc12-v332';out.mkdir(exist_ok=False)
export=json.loads((root/'v332/output/fleet/viewer/trajectories.json').read_text())
meta=dict(run_id=run['run_id'],fleet_run_id=export['trajectories'][0]['source']['run_id'],commit=export['generated_by_commit'],source_code_commit='d1758141 plus gpu-execution-v328 source manifest',source_revision_note='Development source, not merged into main. Execution source hashes and frozen native binary hashes are in gpu-execution-v328 reports.',weighted_score_fixed_bonus_kg=best['independent']['total_mass_kg'],raw_kg_per_ship=best['independent']['total_mass_kg'],hardware=dict(gpu='Lambda NVIDIA H100 80 GB',upstream_search='45,188,558 CUDA transfer branches; 2,782,091 collection options'),timing=dict(wall_seconds_total=run['wall_seconds_total'],wall_human=f"{run['wall_seconds_total']:.2f} s complete campaign; GPU graph execution"),model=dict(dynamics='Official GTOC12 low-thrust dynamics; both mission checkers pass',local_refine='CUDA SCvx and QOCO graph loops after priming'),optimisation=dict(strategy='Matched graph execution benchmark. Fleet incumbent unchanged at 12,805.194 weighted kg. CPU orchestration remains.',proven_optimal=False))
(out/'compute.json').write_text(json.dumps(meta,indent=2))
for name,old,new in [('app.js','  "gtoc12-v269":','  "gtoc12-v332": { directory: "./data/gtoc12-v332", label: "GPU graphs v332 (548 kg certified)" },\n  "gtoc12-v269":'),('index.html','            <option value="gtoc12-v269"','            <option value="gtoc12-v332" disabled>GPU graphs v332 — checking…</option>\n            <option value="gtoc12-v269"'),('scripts/check.mjs','"data/gtoc12-v269"]','"data/gtoc12-v269", "data/gtoc12-v332"]')]:
 p=viewer/name;s=p.read_text();assert old in s and 'gtoc12-v332' not in s;p.write_text(s.replace(old,new,1))
print(run['wall_seconds_total'],best['independent']['total_mass_kg'])
