from pathlib import Path
import hashlib,json,shutil
source=Path('build/performance/retrieved-recovery-v381')
root=Path('results/lambda/2026-09-08/gpu-fleet-recovery-v381')
run=json.loads((source/'output/run_report.json').read_text());best=run['best']
assert best['official']['ok'] and best['independent']['ok']
shutil.copytree(source/'output/fleet',root/'fleet')
shutil.copy2(source/'report.json',root/'report.json')
summary=dict(complete=True,hardware='Lambda NVIDIA H100 80 GB',seconds=run['wall_seconds_total'],
    individual_routes=len(run['ships']),fleet=run['fleet'],master=run['master'],
    official=best['official'],independent=best['independent'],screening=run['screening'],
    recovery_attempts=sum(s.get('recovery',{}).get('attempts',0) for s in run['ships']),
    scope='Complete 32-route campaign using CUDA recovery and harvest pricing, before cooperative warp screening. Final selection has 15 ships. Both final mission checkers pass; incumbent remains 12,805.194 weighted kg.')
(root/'summary.json').write_text(json.dumps(summary,indent=2))
for name in ['run_fleet_recovery_v381.py','prepare_fleet_recovery_v381.py','archive_recovery_v381.py','retrieve_recovery_v381.py','publish_recovery_v381.py']:
 shutil.copy2(Path('build/performance')/name,root/name)
(root/'.gitattributes').write_text('* -text\n')
files={p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob('*') if p.is_file() and p.name!='files-sha256.json'}
(root/'files-sha256.json').write_text(json.dumps(files,indent=2))
print(json.dumps(dict(seconds=summary['seconds'],ships=best['independent']['ships'],score=best['independent']['weighted_score_fixed_bonus_kg'],master={k:run['master'].get(k) for k in ['proven_optimal','gap_kg','upper_bound_kg','nodes','exhaustive','lp_bound_kg','mean_collected_kg','ship_limit']})))
