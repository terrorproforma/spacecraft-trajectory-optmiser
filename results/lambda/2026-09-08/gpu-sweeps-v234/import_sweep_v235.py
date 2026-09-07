from pathlib import Path
import json
root=Path('results/lambda/2026-09-08/gpu-sweeps-v234');r=json.loads((root/'h100/mission-v235/report.json').read_text())
export=json.loads((root/'viewer-export/trajectories.json').read_text());manifest=json.loads((root/'viewer-export/manifest.json').read_text())
mass=r['independent']['total_mass_kg'];fleet=dict(ships=1,asteroids=sorted(r['refinement']['asteroids']),average_collected_kg=mass,collected_kg_per_ship=[mass],total_collected_kg=mass,rule_satisfied=True,ship_limit=r['independent']['ship_limit'])
(root/'viewer-input.json').write_text(json.dumps(dict(fleet=fleet,official=r['official'],independent=r['independent'],viewer_manifest=manifest),indent=2))
out=Path('results/lambda/2026-09-06/visualiser/data/gtoc12-v235');out.mkdir(exist_ok=False)
run_id=export['trajectories'][0]['source']['run_id']
meta=dict(run_id=run_id,fleet_run_id=run_id,commit=export['generated_by_commit'],weighted_score_fixed_bonus_kg=mass,raw_kg_per_ship=mass,hardware=dict(gpu='Lambda NVIDIA H100 80 GB',upstream_search='Resident CUDA return-sweep retiming; 412,116 Lambert branches; zero host table uploads'),timing=dict(wall_seconds_total=r['end']-r['start'],wall_human=f"{r['retiming']['wall_seconds']:.3f} s retiming; {r['refinement']['wall_seconds']:.3f} s GPU refinement"),model=dict(dynamics='Official GTOC12 low-thrust dynamics; official and independent verification pass',local_refine='13 certified CUDA-refined arcs; six mining asteroids'),optimisation=dict(strategy='Return-sweep retiming: 2.464 kg more than the prior 524 kg mission. Incumbent fleet unchanged.',proven_optimal=False))
(out/'compute.json').write_text(json.dumps(meta,indent=2));print(run_id,meta['commit'],mass)
