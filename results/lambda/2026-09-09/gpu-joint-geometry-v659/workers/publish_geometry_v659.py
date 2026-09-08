from pathlib import Path
import hashlib,json,shutil,statistics
local=Path('/home/angus/spacepdhcg-joint-geometry-v651');remote=Path('build/performance/retrieved-geometry-v659');dest=Path('results/lambda/2026-09-09/gpu-joint-geometry-v659')
read=lambda p:json.loads(p.read_text())
summary=dict(base_commit='34cfb4b710be830307065cbb07a754712b60c465',scope='Resident joint geometry versus staged batched CUDA with device selection. Component surrogate timings are separate from full recovery processes including CPU verification and export.',gpus={})
for gpu,root in [('RTX5090',local),('H100',remote)]:
 campaign=read(root/'campaign-v656/report.json');assert campaign['success'] and len(campaign['runs'])==4
 runs=campaign['runs'];reference=None
 for run in runs:
  mode=run['resident_geometry'];best=run['best'];telemetry=run['screening'];details=read(root/'campaign-v656'/run['name']/'report.json')
  assert best['independent']['ok'] and best['official']['ok']
  assert abs(best['score_kg']-12810.135953048577)<1e-6 and abs(best['total_mass_kg']-14051.854893908598)<1e-6
  assert not details['physics_tolerances_changed'] and details['orders']==18
  assert run['native_solves']==36 and telemetry['completed_joint_evaluations']==23142 and telemetry['completed_joint_batches']==132
  assert telemetry['completed_retime_driver_calls']==36 and telemetry['retime_resident_cells']==53010172
  assert telemetry['joint_result_download_bytes']==88576
  assert telemetry.get('joint_preflight_download_bytes',0)==(0 if mode else 1481088)
  if mode:
   assert telemetry['joint_geometry_stats_download_bytes']==3168
   assert telemetry['joint_geometry_computed_hops']==221023 and telemetry['joint_geometry_rejected_hops']==206881
   assert telemetry['joint_geometry_cached_hops']==0
  proxies={}
  for path in (root/'campaign-v656'/run['name']/'proxies').glob('*.json'):
   r=read(path);proxies[path.name]={k:r[k] for k in ('asteroids','orphaned','deploy_epochs','collect_epochs','foreign_deploy_epochs','earth_return_epoch','collected_mass_kg')}
  assert len(proxies)==4
  if reference is None:reference=proxies
  else:assert proxies==reference,run['name']
  run['native_solve_seconds']=sum(json.loads(line)['seconds'] for line in (root/'campaign-v656'/run['name']/'native-solves.jsonl').read_text().splitlines())
  run['proxy_event_and_payload_parity']=True
 times={str(mode):[r['process_seconds'] for r in runs if r['resident_geometry']==mode] for mode in (0,1)}
 medians={mode:statistics.median(v) for mode,v in times.items()}
 check=read(root/'validation-v654/report.json');assert check['success'] and '71 passed' in (root/'validation-v654/pytest.log').read_text()
 for sanitizer in ('memcheck','synccheck','racecheck'):
  text=(root/'validation-v654'/(sanitizer+'.log')).read_text();assert '9 passed' in text and ('0 errors' in text or '0 hazards' in text)
 bench=read(root/'benchmark-v653/benchmark.json');assert bench['complete'] and all(c['exact_winner_parity'] for c in bench['cases'])
 entry=dict(core_sha256=campaign['core_sha256'],qoco_sha256=campaign['qoco_sha256'],tests=71,sanitizer_geometry_tests=9,campaign_runs=runs,process_seconds=times,median_process_seconds=medians,less_time_percent=100*(1-medians['1']/medians['0']),component_cases=[])
 for c in bench['cases']:
  entry['component_cases'].append(dict(ship=c['ship'],cache=c['cache'],candidates=c['candidates'],visits=c['visits'],median_seconds=c['median_seconds'],speedup=c['speedup'],resident_candidates_per_second=c['candidates']/c['median_seconds']['1']))
 summary['gpus'][gpu]=entry
 target=dest/gpu;target.mkdir(exist_ok=False)
 for name in ('report.json','validation-v654/report.json','validation-v654/pytest.log','validation-v654/memcheck.log','validation-v654/synccheck.log','validation-v654/racecheck.log','benchmark-v653/report.json','benchmark-v653/benchmark.json','benchmark-v653/benchmark.log','campaign-v656/report.json'):
  out=target/name;out.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(root/name,out)
 print(gpu,times,entry['less_time_percent'],flush=True)
summary['limitations']=['Only two full recovery processes per mode; no reliable overall speedup is established.','Resident computed Lambert costs are not retained in the Python cache; repeated uncached queries recompute on GPU.','Python move generation, metadata and orchestration remain; final physics verification is independent CPU work.','Sanitizer successes apply to resident geometry tests, not all QOCO/cuDSS code.','Scores reproduce v595; differences below 1e-6 kg are not records.']
summary['fleet']=dict(ships=23,collected_asteroids=195,deployed_asteroids=196,raw_kg=14051.854893908598,weighted_kg=12810.135953048577)
summary['archives']={gpu:read(dest/(gpu+'-raw.tar.record.json')) for gpu in ('local','h100')}
(dest/'summary.json').write_text(json.dumps(summary,indent=2))
workers=dest/'workers';workers.mkdir(exist_ok=False)
names=['build_joint_geometry_v651.py','check_joint_geometry_v652.py','run_joint_geometry_benchmark_v653.py','prepare_geometry_validation_v654.py','prepare_geometry_lambda_v655.py','prepare_geometry_campaign_v656.py','prepare_geometry_campaign_lambda_v657.py','archive_geometry_v659.py','collect_geometry_v659.py','publish_geometry_v659.py','benchmark_joint_selection_v634.py']
for name in names:shutil.copy2(Path('build/performance')/name,workers/name)
