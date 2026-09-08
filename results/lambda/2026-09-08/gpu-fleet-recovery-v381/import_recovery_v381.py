from pathlib import Path
import json
root=Path('results/lambda/2026-09-08/gpu-fleet-recovery-v381')
run=json.loads((root/'summary.json').read_text());best=run
assert best['official']['ok'] and best['independent']['ok']
viewer=Path('results/lambda/2026-09-06/visualiser');out=viewer/'data/gtoc12-v381';out.mkdir(exist_ok=False)
export=json.loads((root/'fleet/viewer/trajectories.json').read_text())
manifest=json.loads((root/'fleet/viewer/manifest.json').read_text())
Path('build/performance/viewer-input-v381.json').write_text(json.dumps(dict(fleet=run['fleet'],official=best['official'],independent=best['independent'],viewer_manifest=manifest),indent=2))
meta=dict(run_id='gpu_fleet_recovery381',fleet_run_id=export['trajectories'][0]['source']['run_id'],commit=export['generated_by_commit'],source_revision_note='Frozen H100 source and runtime hashes are archived in gpu-fleet-recovery-v381. This run predates the cooperative warp scan.',weighted_score_fixed_bonus_kg=best['independent']['weighted_score_fixed_bonus_kg'],raw_kg_per_ship=best['independent']['total_mass_kg']/best['independent']['ships'],hardware=dict(gpu='Lambda NVIDIA H100 80 GB',upstream_search=f"{run['screening']['completed_branch_requests']:,} CUDA transfer branches"),timing=dict(wall_seconds_total=run['seconds'],wall_human=f"{run['seconds']:.2f} s complete 32-route campaign, selecting 15 ships"),model=dict(dynamics='Official GTOC12 low-thrust dynamics; both final fleet checkers pass',local_refine='CUDA SCvx and QOCO graph loops'),optimisation=dict(strategy='Bounded CUDA recovery produced 32 routes; fleet selection retained 15 under the mission constraints. Incumbent remains 12,805.194 weighted kg.',proven_optimal=False))
(out/'compute.json').write_text(json.dumps(meta,indent=2))
for name,old,new in [('app.js','  "gtoc12-v380":','  "gtoc12-v381": { directory: "./data/gtoc12-v381", label: "H100 recovery v381 (15 ships, 7,802 weighted kg)" },\n  "gtoc12-v380":'),('index.html','            <option value="gtoc12-v380"','            <option value="gtoc12-v381" disabled>H100 recovery v381 — checking…</option>\n            <option value="gtoc12-v380"'),('scripts/check.mjs','"data/gtoc12-v380"]','"data/gtoc12-v380", "data/gtoc12-v381"]')]:
 p=viewer/name;s=p.read_text();assert old in s and 'gtoc12-v381' not in s;p.write_text(s.replace(old,new,1))
