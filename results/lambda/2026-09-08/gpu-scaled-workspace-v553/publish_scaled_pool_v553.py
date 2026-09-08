from pathlib import Path
import hashlib,json,shutil,subprocess,tarfile
p=Path('build/performance');target=Path('results/lambda/2026-09-08/gpu-scaled-workspace-v553');remote=p/'retrieved-scaled-v553'
def read(path):return json.loads(path.read_text())
for script,folder in [('analyze_scaled_pool_legs.py',p/'scaled-pool-v547'),('analyze_scaled_pool_legs.py',remote/'replay-v549'),('analyze_scaled_pool_campaign.py',p/'scaled-pool-campaign-v543'),('analyze_scaled_pool_campaign.py',remote/'campaign-v548')]:
 subprocess.run(['python3',str(p/script),str(folder)],check=True,stdout=subprocess.DEVNULL)
legs=[read(root/'analysis.json') for root in [p/'scaled-pool-v547',remote/'replay-v549']]
for r in legs:
 assert not r['lost_baseline'] and r['baseline']['converged']==r['candidate']['converged']==205
 assert r['max_certified_mass_delta']['delta_kg']<1e-5
 assert r['baseline']['workspaces']==225 and r['candidate']['workspaces']==72
campaigns=[read(root/'analysis.json') for root in [p/'scaled-pool-campaign-v543',remote/'campaign-v548']]
for r in campaigns:
 assert all(v['same_initial_plans'] and v['same_logical_counts'] and abs(v['score']-548.2546201232)<1e-6 and v['workspace_creations']==17 for v in r['rows'])
for root in [p/'scaled-pool-v542',remote/'validation-v545']:
 assert '117 passed' in (root/'pytest.log').read_text()
 assert read(root/'report.json').get('error')
files={}
sources={name:p/name for name in ['scaled-pool-v539','scaled-pool-v542','scaled-pool-campaign-v543','scaled-pool-sanitizer-v546','scaled-pool-v547','scaled-pool-scoped-sanitizer-v551']}
sources.update({'qoco-build-v540':Path('/home/angus/build-qoco-scaled-pool-v540'),'core-build-v539':Path('/home/angus/build-spacepdhcg-scaled-pool-v539')})
for label,root in sources.items():
 assert root.exists()
 for path in sorted(root.rglob('*')):
  if not path.is_file():continue
  rel=path.relative_to(root)
  if set(rel.parts)&{'.git','__pycache__','build'}:continue
  files[label+'/'+rel.as_posix()]=path
manifest={name:hashlib.sha256(path.read_bytes()).hexdigest() for name,path in files.items()}
archive=target/'local-raw.tar.gz'
with tarfile.open(archive,'w:gz') as t:
 for name,path in files.items():t.add(path,arcname=name,recursive=False)
(target/'local-archive-manifest.json').write_text(json.dumps(manifest,indent=2))
summary=dict(source_base_commit='dd4f4a5adf668247fa64b840c14f846bf8b37be2',default_enabled=False,flag='SPACEPDHCG_TEST_GTOC12_SCALED_QOCO_POOL',requires='SPACEPDHCG_TEST_QOCO_RUIZ_PRESERVE_OBJECTIVE=1 and library capability query',local_legs=legs[0],lambda_legs=legs[1],local_campaign=campaigns[0],lambda_campaign=campaigns[1],preparation=read(p/'scaled-pool-preparation-verification.json'),tests_per_gpu=117,sanitizers={
 'local_full':read(p/'scaled-pool-v542/report.json')['stages'],
 'local_previous_and_candidate':read(p/'scaled-pool-sanitizer-v546/report.json'),
 'local_scoped':read(p/'scaled-pool-scoped-sanitizer-v551/report.json'),
 'lambda_full':read(remote/'validation-v545/report.json')['stages'],
 'lambda_previous_and_candidate':read(remote/'diagnostic-v550/report.json'),
 'lambda_baseline_race':read(remote/'baseline-race-v554/report.json'),
 'lambda_scoped':read(remote/'scoped-v552/report.json')},limitations=['Experimental scaled reuse remains disabled by default.','Full solver sanitizer validation is not clean; detailed failures are retained, including baseline reproductions.','One 225-leg pass per mode does not establish a universal speedup. Complete campaigns compare production zero Ruiz with two preserved-objective Ruiz passes plus reuse.','No new fleet record; best remains 12805.194102488575 weighted kg. CPU route and fleet orchestration still remains.'])
(target/'summary.json').write_text(json.dumps(summary,indent=2))
for pattern in ['*scaled*pool*.py','*scaled*followup*.py','*scaled*scoped*.py','*scaled*analysis*.py','*scaled*remote*v553.py','*scaled*runtime.py']:
 for path in p.glob(pattern):
  if path.name.startswith(('launch_','run_scaled_pool_v')):continue
  shutil.copy2(path,target/path.name)
shutil.copy2(p/'scaled-pool-preparation-verification.json',target/'preparation.json')
(target/'.gitattributes').write_text('* -text whitespace=cr-at-eol,-blank-at-eof\n')
manifest={q.relative_to(target).as_posix():hashlib.sha256(q.read_bytes()).hexdigest() for q in sorted(target.rglob('*')) if q.is_file() and q.name!='files-sha256.json'}
assert all((target/name).stat().st_size<90_000_000 for name in manifest)
(target/'files-sha256.json').write_text(json.dumps(manifest,indent=2))
print(json.dumps(dict(files=len(manifest),local_archive_bytes=archive.stat().st_size,summary=str(target/'summary.json')),indent=2))
