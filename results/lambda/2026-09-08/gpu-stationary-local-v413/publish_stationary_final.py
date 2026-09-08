from pathlib import Path
import json,hashlib,tarfile,shutil,statistics
root=Path('results/lambda/2026-09-08/gpu-stationary-local-v413');root.mkdir(exist_ok=False)
archives={};campaigns=[]
for name in ['stationary-fleet-v409','stationary-v411','stationary-fleet-v413']:
 source=Path('build/performance')/name;r=json.loads((source/'report.json').read_text());assert r['complete'] and not r.get('error')
 if (source/'output/run_report.json').exists():
  run=json.loads((source/'output/run_report.json').read_text());b=run['best'];assert b['accepted'] and b['official']['ok'] and b['independent']['ok']
  campaigns.append(dict(name=name,seconds=run['wall_seconds_total'],score=b['independent']['weighted_score_fixed_bonus_kg'],official=b['official'],independent=b['independent'],screening=run['screening']))
 manifest={p.relative_to(source).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(source.rglob('*')) if p.is_file()}
 with tarfile.open(root/(name+'.tar.gz'),'w:gz') as t:
  for member in manifest:t.add(source/member,arcname=member,recursive=False)
 archives[name+'.tar.gz']=manifest
(root/'archive-manifests.json').write_text(json.dumps(archives,indent=2))
previous=json.loads(Path('results/lambda/2026-09-08/gpu-grid-cache-fleet-v403/summary.json').read_text())
assert campaigns[0]['screening']==previous['screening']
validation=json.loads(Path('build/performance/stationary-v411/report.json').read_text())
assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==sha for p,sha in validation['source_sha256'].items())
(root/'summary.json').write_text(json.dumps(dict(hardware='RTX 5090',campaigns=campaigns,same_four_ship_search_counts_as_v403=True,default_validation=validation,scope='Single wider opt-in mission and single final default-on confirmation; not a paired speed comparison. Physics gates unchanged.'),indent=2))
for name in ['run_stationary_fleet_v409.py','check_stationary_v411.py','run_stationary_default_v413.py','publish_stationary_final.py']:shutil.copy2(Path('build/performance')/name,root/name)
for version in ['fleet-v410','v412']:
 source=Path('build/performance/retrieved-stationary-'+version);target=Path('results/lambda/2026-09-08/gpu-stationary-'+version)
 r=json.loads((source/'report.json').read_text());assert r['complete'] and not r.get('error');cs=r['campaigns']
 for c in cs:
  run=json.loads((source/c['name']/'output/run_report.json').read_text());assert run['best']['accepted'] and run['best']['official']['ok'] and run['best']['independent']['ok']
  c.update(official=run['best']['official'],independent=run['best']['independent'])
 summary=dict(hardware='Lambda H100',campaigns=cs,runtime_sha256=r['runtime_sha256'],scope='ABBA with two runs per mode on frozen v408 runtime.' if version=='fleet-v410' else 'Single final default-on confirmation, separate from the paired timing experiment.')
 if version=='fleet-v410':
  assert all(c['screening']==cs[0]['screening'] for c in cs)
  b=statistics.median(c['seconds'] for c in cs if not c['candidate']);a=statistics.median(c['seconds'] for c in cs if c['candidate'])
  summary.update(baseline_median_seconds=b,candidate_median_seconds=a,speedup=b/a,less_time_percent=100*(1-a/b))
 else:
  assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==sha for p,sha in r['source_sha256'].items())
  summary['source_sha256']=r['source_sha256'];shutil.copytree(source/'source-overlay',target/'source-overlay')
  for name in ['pytest.log','probe.log','memcheck.log','synccheck.log','racecheck.log']:shutil.copy2(source/name,target/name)
 shutil.copy2(source/'report.json',target/'report.json')
 (target/'summary.json').write_text(json.dumps(summary,indent=2))
 scriptversion=version.replace('-','_')
 for prefix in ['run_stationary_','prepare_stationary_','archive_stationary_','retrieve_stationary_']:
  name=prefix+scriptversion+'.py';shutil.copy2(Path('build/performance')/name,target/name)
for folder in [root,Path('results/lambda/2026-09-08/gpu-stationary-fleet-v410'),Path('results/lambda/2026-09-08/gpu-stationary-v412')]:
 (folder/'.gitattributes').write_text('* -text whitespace=cr-at-eol\n')
 manifest={p.relative_to(folder).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.rglob('*') if p.is_file() and p.name!='files-sha256.json'}
 (folder/'files-sha256.json').write_text(json.dumps(manifest,indent=2))
 for name,sha in manifest.items():assert hashlib.sha256((folder/name).read_bytes()).hexdigest()==sha
 print(folder,len(manifest),'files verified')
for name,manifest in archives.items():
 with tarfile.open(root/name) as t:
  assert {m.name for m in t.getmembers()}==set(manifest)
  for member,sha in manifest.items():assert hashlib.sha256(t.extractfile(member).read()).hexdigest()==sha
 print(name,len(manifest),'archive members verified')
