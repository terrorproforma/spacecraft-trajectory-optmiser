from pathlib import Path
import hashlib,json,shutil,statistics
local=Path('/home/angus/spacepdhcg-joint-mesh-v662');remote=Path('build/performance/retrieved-mesh-v670');dest=Path('results/lambda/2026-09-09/gpu-joint-mesh-v670')
read=lambda p:json.loads(p.read_text())
summary=dict(base_commit='2cbdd4255382f63deb598fc3b02095064b495799',scope='GPU mesh generation versus CPU mesh generation, both using resident CUDA geometry and device selection. Component surrogate timings are separate from full recovery processes including CPU verification and export.',gpus={})
for gpu,root in [('RTX5090',local),('H100',remote)]:
 campaign=read(root/'campaign-v667/report.json');assert campaign['success'] and len(campaign['runs'])==4
 runs=campaign['runs'];reference=None
 for run in runs:
  mode=run['device_mesh'];best=run['best'];telemetry=run['screening'];details=read(root/'campaign-v667'/run['name']/'report.json')
  assert best['independent']['ok'] and best['official']['ok']
  assert abs(best['score_kg']-12810.135953048577)<1e-6 and abs(best['total_mass_kg']-14051.854893908598)<1e-6
  assert not details['physics_tolerances_changed'] and details['orders']==18
  assert run['native_solves']==36 and telemetry['completed_joint_evaluations']==23142 and telemetry['completed_joint_batches']==132
  assert telemetry['completed_retime_driver_calls']==36 and telemetry['retime_resident_cells']==53010172
  assert telemetry['joint_result_download_bytes']==88576
  assert telemetry.get('joint_preflight_download_bytes',0)==0
  if mode:
   assert telemetry['completed_joint_mesh_batches']>0
   assert telemetry['joint_mesh_epoch_upload_bytes']==telemetry['joint_mesh_epoch_download_bytes']
   assert telemetry['joint_geometry_stats_download_bytes']==3168
   assert telemetry['joint_geometry_computed_hops']==221023 and telemetry['joint_geometry_rejected_hops']==206881
   assert telemetry['joint_geometry_cached_hops']==0
  proxies={}
  for path in (root/'campaign-v667'/run['name']/'proxies').glob('*.json'):
   r=read(path);proxies[path.name]={k:r[k] for k in ('asteroids','orphaned','deploy_epochs','collect_epochs','foreign_deploy_epochs','earth_return_epoch','collected_mass_kg')}
  assert len(proxies)==4
  if reference is None:reference=proxies
  else:assert proxies==reference,run['name']
  run['native_solve_seconds']=sum(json.loads(line)['seconds'] for line in (root/'campaign-v667'/run['name']/'native-solves.jsonl').read_text().splitlines())
  run['refinement_outcomes']=[dict(case=r['case'],grid_days=r['grid_days'],certified=r['certified'],status=r['status'],seconds=r['seconds']) for r in details['refinements']]
  run['refinement_full_fleet_checks']=sum('verification' in r for r in details['refinements'])
  run['proxy_event_and_payload_parity']=True
 times={str(mode):[r['process_seconds'] for r in runs if r['device_mesh']==mode] for mode in (0,1)}
 medians={mode:statistics.median(v) for mode,v in times.items()}
 check=read(root/'validation-v665/report.json');assert check['success'] and '87 passed' in (root/'validation-v665/pytest.log').read_text()
 for sanitizer in ('memcheck','synccheck','racecheck'):
  text=(root/'validation-v665'/(sanitizer+'.log')).read_text();assert '25 passed' in text and ('0 errors' in text or '0 hazards' in text)
 bench=read(root/'benchmark-v664/benchmark.json');assert bench['complete'] and all(c['exact_winner_parity'] for c in bench['cases'])
 entry=dict(core_sha256=campaign['core_sha256'],qoco_sha256=campaign['qoco_sha256'],tests=87,sanitizer_geometry_tests=25,campaign_runs=runs,process_seconds=times,median_process_seconds=medians,less_time_percent=100*(1-medians['1']/medians['0']),component_cases=[])
 for c in bench['cases']:
  entry['component_cases'].append(dict(ship=c['ship'],cache=c['cache'],candidates=c['candidates'],visits=c['visits'],median_seconds=c['median_seconds'],speedup=c['speedup'],resident_candidates_per_second=c['candidates']/c['median_seconds']['1']))
 summary['gpus'][gpu]=entry
 target=dest/gpu;target.mkdir(exist_ok=False)
 for name in ('report.json','validation-v665/report.json','validation-v665/pytest.log','validation-v665/memcheck.log','validation-v665/synccheck.log','validation-v665/racecheck.log','benchmark-v664/report.json','benchmark-v664/benchmark.json','benchmark-v664/benchmark.log','campaign-v667/report.json'):
  out=target/name;out.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(root/name,out)
 print(gpu,times,entry['less_time_percent'],flush=True)
summary['limitations']=['Only two full recovery processes per mode; no reliable overall speedup is established.','Resident computed Lambert costs are not retained in the Python cache; repeated uncached queries recompute on GPU.','Python mesh progression, metadata and orchestration remain; final physics verification is independent CPU work.','Sanitizer successes apply to resident geometry and mesh tests, not all QOCO/cuDSS code.','Scores reproduce v595; differences below 1e-6 kg are not records.']
summary['fleet']=dict(ships=23,collected_asteroids=195,deployed_asteroids=196,raw_kg=14051.854893908598,weighted_kg=12810.135953048577)
summary['archives']={gpu:read(dest/(gpu+'-raw.tar.record.json')) for gpu in ('local','h100')}
(dest/'summary.json').write_text(json.dumps(summary,indent=2))
workers=dest/'workers';workers.mkdir(exist_ok=False)
names=['build_joint_mesh_v662.py', 'check_joint_mesh_v663.py', 'check_joint_mesh_v665.py', 'update_mesh_tests_v665.py', 'benchmark_joint_mesh_v664.py', 'run_joint_mesh_benchmark_v664.py', 'prepare_mesh_lambda_v666.py', 'upload_mesh_lambda_v666.py', 'prepare_mesh_campaign_v667.py', 'upload_mesh_campaign_v668.py', 'archive_mesh_v670.py', 'collect_mesh_v670.py', 'publish_mesh_v670.py', 'prepare_mesh_archive_lambda_v670.py']
for name in names:shutil.copy2(Path('build/performance')/name,workers/name)
