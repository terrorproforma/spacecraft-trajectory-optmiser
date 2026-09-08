from pathlib import Path
import json
root=Path('results/lambda/2026-09-08/gpu-soc-step-v359/lambda-final-v359')
run=json.loads((root/'v360/output/run_report.json').read_text());best=run['best']
(root/'viewer-input.json').write_text(json.dumps(dict(fleet=run['fleet'],official=best['official'],independent=best['independent'],viewer_manifest=best['viewer_manifest']),indent=2))
viewer=Path('results/lambda/2026-09-06/visualiser');out=viewer/'data/gtoc12-v360';out.mkdir(exist_ok=False)
export=json.loads((root/'v360/output/fleet/viewer/trajectories.json').read_text())
meta=dict(run_id=run['run_id'],fleet_run_id=export['trajectories'][0]['source']['run_id'],commit=export['generated_by_commit'],source_code_commit='6945f4f7 plus gpu-soc-step-v359 source manifest',source_revision_note='Development source. SOC step header and native binary hashes are in gpu-soc-step-v359 reports.',weighted_score_fixed_bonus_kg=best['independent']['weighted_score_fixed_bonus_kg'],raw_kg_per_ship=best['independent']['total_mass_kg'],hardware=dict(gpu='Lambda NVIDIA H100 80 GB',upstream_search='45,188,558 CUDA transfer branches; 2,782,091 collection options'),timing=dict(wall_seconds_total=run['wall_seconds_total'],wall_human=f"{run['wall_seconds_total']:.2f} s complete campaign; GPU graph execution"),model=dict(dynamics='Official GTOC12 low-thrust dynamics; both mission checkers pass',local_refine='CUDA SCvx and QOCO graph loops after priming'),optimisation=dict(strategy='SOC line-search correction benchmark. Fleet incumbent unchanged at 12,805.194 weighted kg. CPU orchestration remains.',proven_optimal=False))
(out/'compute.json').write_text(json.dumps(meta,indent=2))
for name,old,new in [('app.js','  "gtoc12-v342":','  "gtoc12-v360": { directory: "./data/gtoc12-v360", label: "GPU SOC step v360 (548 kg certified)" },\n  "gtoc12-v342":'),('index.html','            <option value="gtoc12-v342"','            <option value="gtoc12-v360" disabled>GPU SOC step v360 — checking…</option>\n            <option value="gtoc12-v342"'),('scripts/check.mjs','"data/gtoc12-v342"]','"data/gtoc12-v342", "data/gtoc12-v360"]')]:
 p=viewer/name;s=p.read_text();assert old in s and 'gtoc12-v360' not in s;p.write_text(s.replace(old,new,1))
print(run['wall_seconds_total'],best['independent']['total_mass_kg'])
