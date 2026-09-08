from pathlib import Path
import json,shutil
p=Path('build/performance');root=p/'retrieved-fast-root-v589/campaign-final'
r=json.loads((root/'report.json').read_text());case=r['campaigns'][0]
output=root/'default/output';result=json.loads((output/'run_report.json').read_text())
assert result['best']['accepted'] and result['best']['official']['ok'] and result['best']['independent']['ok']
dest=Path('results/lambda/2026-09-06/visualiser/data/gtoc12-v588');dest.mkdir(exist_ok=False)
meta=dict(run_id='fastroot588',fleet_run_id='fastroot588_fleet',commit=json.loads((output/'fleet/viewer/trajectories.json').read_text())['generated_by_commit'],source_revision_note='Final default safeguarded CUDA Lambert roots. Frozen source and binary hashes in gpu-fast-lambert-v589.',weighted_score_fixed_bonus_kg=case['score'],raw_kg_per_ship=result['best']['independent']['total_mass_kg'],hardware=dict(gpu='Lambda NVIDIA H100 80 GB',upstream_search='45,188,558 logical CUDA transfer branches; 2,782,091 collection options'),timing=dict(wall_seconds_total=case['process_seconds'],wall_human=f"{case['process_seconds']:.2f} s complete one-ship benchmark process"),model=dict(dynamics='Official GTOC12 low-thrust dynamics; official and independent mission verification passed',local_refine='CUDA SCvx and QOCO graph loops; GPU safeguarded Lambert screening; zero Ruiz'),optimisation=dict(strategy='One-ship performance benchmark. Best fleet remains 12,805.194 weighted kg. CPU orchestration remains.',proven_optimal=False))
(dest/'compute.json').write_text(json.dumps(meta,indent=2))
v=result['best']['independent']
fleet=dict(viewer_manifest=json.loads((output/'fleet/viewer/manifest.json').read_text()),official=result['best']['official'],independent=v,fleet=dict(ships=v['ships'],total_collected_kg=v['total_mass_kg'],collected_kg_per_ship=[v['total_mass_kg']],asteroids=sorted(map(int,v['scored_masses'])),ship_limit=v['ship_limit']))
(p/'fast-root-viewer-fleet.json').write_text(json.dumps(fleet,indent=2))
shutil.copy2('/home/angus/worktrees/spacepdhcg-release/benchmarks/gtoc12/data/GTOC12_Asteroids_Data.txt',p/'fast-root-viewer-catalogue.txt')
