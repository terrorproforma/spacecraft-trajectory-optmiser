from pathlib import Path
import json
root=Path('results/lambda/2026-09-08/gpu-nonfinite-ir-v337/remote/confirmation')
run=json.loads((root/'v342/output/run_report.json').read_text());best=run['best']
(root/'viewer-input.json').write_text(json.dumps(dict(fleet=run['fleet'],official=best['official'],independent=best['independent'],viewer_manifest=best['viewer_manifest']),indent=2))
viewer=Path('results/lambda/2026-09-06/visualiser');out=viewer/'data/gtoc12-v342';out.mkdir(exist_ok=False)
export=json.loads((root/'v342/output/fleet/viewer/trajectories.json').read_text())
meta=dict(run_id=run['run_id'],fleet_run_id=export['trajectories'][0]['source']['run_id'],commit=export['generated_by_commit'],source_code_commit='bef73903 plus gpu-nonfinite-ir-v337 source manifest',source_revision_note='Development source, not merged into main. Guard source and native binary hashes are in gpu-nonfinite-ir-v337 reports.',weighted_score_fixed_bonus_kg=best['independent']['total_mass_kg'],raw_kg_per_ship=best['independent']['total_mass_kg'],hardware=dict(gpu='Lambda NVIDIA H100 80 GB',upstream_search='45,188,558 CUDA transfer branches; 2,782,091 collection options'),timing=dict(wall_seconds_total=run['wall_seconds_total'],wall_human=f"{run['wall_seconds_total']:.2f} s complete campaign; GPU graph execution"),model=dict(dynamics='Official GTOC12 low-thrust dynamics; both mission checkers pass',local_refine='CUDA SCvx and QOCO graph loops after priming'),optimisation=dict(strategy='Nonfinite IR guard benchmark. Fleet incumbent unchanged at 12,805.194 weighted kg. CPU orchestration remains.',proven_optimal=False))
(out/'compute.json').write_text(json.dumps(meta,indent=2))
for name,old,new in [('app.js','  "gtoc12-v332":','  "gtoc12-v342": { directory: "./data/gtoc12-v342", label: "GPU IR guard v342 (548 kg certified)" },\n  "gtoc12-v332":'),('index.html','            <option value="gtoc12-v332"','            <option value="gtoc12-v342" disabled>GPU IR guard v342 — checking…</option>\n            <option value="gtoc12-v332"'),('scripts/check.mjs','"data/gtoc12-v332"]','"data/gtoc12-v332", "data/gtoc12-v342"]')]:
 p=viewer/name;s=p.read_text();assert old in s and 'gtoc12-v342' not in s;p.write_text(s.replace(old,new,1))
print(run['wall_seconds_total'],best['independent']['total_mass_kg'])
