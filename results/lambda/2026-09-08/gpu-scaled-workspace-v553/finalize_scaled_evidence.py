from pathlib import Path
import hashlib,json,shutil
p=Path('build/performance');target=Path('results/lambda/2026-09-08/gpu-scaled-workspace-v553')
for name in ['prepare_scaled_viewer.py','scaled-viewer-fleet.json','finalize_scaled_evidence.py','verify_scaled_pool_blobs.py','analyze_scaled_remote.py','prepare_scaled_baseline_race_v554.py']:
 shutil.copy2(p/name,target/name)
summary=json.loads((target/'summary.json').read_text())
summary['sanitizers']['lambda_scoped']['scope']='Requested cuDSS exclusion did not isolate cuDSS conditional-graph kernels: synccheck still reports cuDSS barriers; memory check passes, racecheck crashes. Not a full solver sanitizer pass.'
summary['viewer']=dict(dataset='gtoc12-v548',source='campaign-v548/candidate0/output/fleet',weighted_score_kg=548.2546201231,ships=1,mined_asteroids=8,exact_replay_samples=509,kepler_context_points=3010,url='http://127.0.0.1:4173/?dataset=gtoc12-v548&epoch=69807&preset=oblique&z=1')
(target/'summary.json').write_text(json.dumps(summary,indent=2))
manifest={q.relative_to(target).as_posix():hashlib.sha256(q.read_bytes()).hexdigest() for q in sorted(target.rglob('*')) if q.is_file() and q.name!='files-sha256.json'}
(target/'files-sha256.json').write_text(json.dumps(manifest,indent=2))
