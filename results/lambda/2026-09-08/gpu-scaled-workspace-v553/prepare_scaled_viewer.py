from pathlib import Path
import json,shutil
p=Path('build/performance');root=p/'retrieved-scaled-v553/campaign-v548'
r=json.loads((root/'report.json').read_text());case=next(c for c in r['campaigns'] if c['name']=='candidate0')
output=root/'candidate0/output';result=json.loads((output/'run_report.json').read_text())
assert result['best']['official']['ok'] and result['best']['independent']['ok'] and result['best']['accepted']
dest=Path('results/lambda/2026-09-06/visualiser/data/gtoc12-v548');dest.mkdir(exist_ok=True)
meta=dict(run_id='scaled548',fleet_run_id='scaled548_fleet',commit=r['source_commit'],source_revision_note='Frozen source overlay and binary hashes are archived in gpu-scaled-workspace-v553; experimental scaled reuse, not a new fleet record.',weighted_score_fixed_bonus_kg=case['score'],raw_kg_per_ship=result['best']['independent']['total_mass_kg'],hardware=dict(gpu='Lambda NVIDIA H100 80 GB',upstream_search='45,188,558 logical CUDA transfer branches; 2,782,091 collection options'),timing=dict(wall_seconds_total=case['process_seconds'],wall_human=f"{case['process_seconds']:.2f} s complete one-ship benchmark process"),model=dict(dynamics='Official GTOC12 low-thrust dynamics; both mission checkers pass',local_refine='CUDA SCvx and QOCO graph loops, two objective-preserving Ruiz passes and workspace reuse'),optimisation=dict(strategy='Fixed one-ship performance benchmark; best fleet remains 12,805.194 weighted kg. CPU orchestration remains. Scaled reuse is opt-in; sanitizer failures remain unresolved.',proven_optimal=False))
meta['source_base_commit']=r['source_commit']
meta['commit']=json.loads((output/'fleet/viewer/trajectories.json').read_text())['generated_by_commit']
(dest/'compute.json').write_text(json.dumps(meta,indent=2))
verification=result['best']['independent']
fleet=dict(viewer_manifest=json.loads((output/'fleet/viewer/manifest.json').read_text()),official=result['best']['official'],independent=verification,fleet=dict(ships=verification['ships'],total_collected_kg=verification['total_mass_kg'],collected_kg_per_ship=[verification['total_mass_kg']],asteroids=sorted(map(int,verification['scored_masses'])),ship_limit=verification['ship_limit']))
(p/'scaled-viewer-fleet.json').write_text(json.dumps(fleet,indent=2))
shutil.copy2('/home/angus/worktrees/spacepdhcg-release/benchmarks/gtoc12/data/GTOC12_Asteroids_Data.txt',p/'scaled-viewer-catalogue.txt')
print(json.dumps(dict(score=case['score'],source=str(output/'fleet'),destination=str(dest))))
