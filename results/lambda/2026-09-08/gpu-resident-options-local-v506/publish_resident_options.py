from pathlib import Path
import hashlib,json,shutil,tarfile
p=Path('build/performance');base=Path('results/lambda/2026-09-08');target=base/'gpu-resident-options-local-v506'
assert not target.exists()
def read(folder,name='report.json'):return json.loads((p/folder/name).read_text())
local=read('resident-options-v506');remote=read('retrieved-resident-options-v507')
for r in [local,remote]:
 assert r['complete'] and not r.get('error')
 for name,sha in r['source_sha256'].items():assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==sha,name
for folder in ['resident-options-v506','retrieved-resident-options-v507']:assert '103 passed' in (p/folder/'pytest.log').read_text()
for folder in ['resident-options-v495','retrieved-resident-options-v497']:
 assert '5930 captured queries across 992 tables' in (p/folder/'replay.log').read_text()
 for tool in ['memcheck','synccheck','racecheck']:
  log=(p/folder/(tool+'.log')).read_text()
  assert 'ERROR SUMMARY: 0 errors' in log or 'RACECHECK SUMMARY: 0 hazards' in log,(folder,tool)
# The final verifier source changed after the wider run; v508 rechecks its
# unchanged result file as well as both incumbent fleets with the final source.
recheck=read('verifier-fleets-v508');assert recheck['complete'] and not recheck.get('error')
assert all(c['summary']['ok'] for c in recheck['cases'])
for name,sha in recheck['source_sha256'].items():assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==sha
for case in recheck['cases']:assert hashlib.sha256(Path(case['path']).read_bytes()).hexdigest()==case['sha256']
for folder in ['lagrange-precision-v502','retrieved-lagrange-precision-v503']:
 for mode in ['prior','candidate']:
  rows=read(folder,mode+'/solves.json');assert len(rows)==192
  assert all(x['status']=='converged' and x['certified'] for x in rows)
target.mkdir()
names=['pipeline-profile-v492','resident-options-capture-v493','resident-options-v494','resident-options-v495','resident-options-campaign-v496','resident-options-v498','resident-options-fleet-v499','lagrange-precision-v502','verifier-knots-v504','verifier-regression-v505','resident-options-v506','verifier-fleets-v508']
archives={}
for name in names:
 source=p/name;r=read(name)
 if name in ['resident-options-v498','verifier-regression-v505']:assert not r['complete'] and r.get('error')
 else:assert r['complete'] and not r.get('error'),name
 manifest={q.relative_to(source).as_posix():hashlib.sha256(q.read_bytes()).hexdigest() for q in sorted(source.rglob('*')) if q.is_file()}
 with tarfile.open(target/(name+'.tar.gz'),'w:gz') as t:
  for member in manifest:t.add(source/member,arcname=member,recursive=False)
 archives[name+'.tar.gz']=manifest
(target/'archive-manifests.json').write_text(json.dumps(archives,indent=2))
shutil.copy2(p/'resident-options-fixture.bin',target/'resident-options-fixture.bin')
summary=dict(default_resident_options=True,hardware='RTX 5090',source_base_commit='37cda103a624f85fae38bb6a62f045e0dadf9ebf',complete_campaign=read('resident-options-campaign-v496','analysis.json'),validation=local,selection_capture=read('resident-options-capture-v493','capture-summary.json'),wider_process_seconds=read('resident-options-fleet-v499')['seconds'],fleet_reverification=recheck,verifier_diagnosis=read('verifier-knots-v504/cases','analysis.json'),limitations=['Two runs per mode; complete timing also varies with numerical refinement. H100 ranges overlap. Four-ship timing does not establish an improvement.','v498/v500 failures included object-vs-list comparison and Lagrange mass consistency. v501/v502/v503 retained old/new solver repetitions and an unsuccessful tighter-tolerance experiment. No experimental solver tolerance is promoted.','v505 regression fails on the old DOP853 replay and passes with polynomial-boundary integration; the old workspace mass assertion and production solver tolerances are unchanged.','v496/v497 selection-download telemetry excludes 5280 bytes from return pruning. Final telemetry counts these; array upload/read counters were already correct.','C++/CUDA retains and selects option tables, but query metadata, route/fleet orchestration and independent reference checks still involve CPU work. No new fleet record or official rank.'])
(target/'summary.json').write_text(json.dumps(summary,indent=2))
prefixes=('prepare_resident','check_resident','run_resident','capture_resident','analyze_resident','launch_resident','status_resident','archive_resident','retrieve_resident','publish_resident','prepare_lagrange','diagnose_lagrange','run_lagrange','analyze_lagrange','launch_lagrange','status_lagrange','archive_lagrange','retrieve_lagrange','prepare_verifier','diagnose_verifier','run_verifier','check_verifier_regression','reverify_fleets_v508','freeze_resident')
for q in p.glob('*.py'):
 if q.name.startswith(prefixes) or q.name in ['profile_v492.py','run_profile_v492.py','prepare_pipeline_profile_v492.py']:shutil.copy2(q,target/q.name)
folders=[target]
for tag in ['resident-options-v497','resident-options-v500','lagrange-repeat-v501','lagrange-precision-v503','resident-options-v507']:
 source=p/('retrieved-'+tag);dest=base/('gpu-'+tag);folders.append(dest)
 for name in ['report.json','analysis.json','diagnostic-source-sha256.json','diagnostic-runtime-sha256.json']:
  if (source/name).exists():shutil.copy2(source/name,dest/name)
 (dest/'summary.json').write_text(json.dumps(dict(hardware='Lambda H100',scope=('Retained failed suite; see verifier diagnosis.' if tag.endswith('500') else 'Solver/verification diagnosis, no changed production tolerances.' if tag.startswith('lagrange-') else 'Measured resident-option validation; see reports and raw archive.'),default_resident_options=tag.endswith('507')),indent=2))
for folder in folders:
 (folder/'.gitattributes').write_text('* -text whitespace=cr-at-eol,-blank-at-eof\n')
 manifest={q.relative_to(folder).as_posix():hashlib.sha256(q.read_bytes()).hexdigest() for q in sorted(folder.rglob('*')) if q.is_file() and q.name!='files-sha256.json'}
 assert all((folder/name).stat().st_size<90_000_000 for name in manifest)
 (folder/'files-sha256.json').write_text(json.dumps(manifest,indent=2))
 print(folder,len(manifest),'files',sum((folder/name).stat().st_size for name in manifest),'bytes',flush=True)
for archive,manifest in archives.items():
 with tarfile.open(target/archive) as t:
  assert {m.name for m in t.getmembers()}==set(manifest)
  for name,sha in manifest.items():assert hashlib.sha256(t.extractfile(name).read()).hexdigest()==sha,name
 print(archive,len(manifest),'members verified',flush=True)
