from pathlib import Path
import difflib,hashlib,json,shutil,subprocess,tarfile
p=Path('build/performance')
target=Path('results/lambda/2026-09-08/gpu-device-init-workspace-v566')
remote=p/'retrieved-device-init-v566'
def read(path):return json.loads(path.read_text())
for script,folder in [('analyze_device_initialization_legs.py',p/'device-init-v562'),('analyze_device_initialization_legs.py',remote/'replay-v563'),('analyze_device_initialization_campaign.py',p/'device-init-campaign-v560'),('analyze_device_initialization_campaign.py',remote/'campaign-v561')]:
 subprocess.run(['python3',str(p/script),str(folder)],check=True,stdout=subprocess.DEVNULL)
legs=[read(root/'analysis.json') for root in [p/'device-init-v562',remote/'replay-v563']]
campaigns=[read(root/'analysis.json') for root in [p/'device-init-campaign-v560',remote/'campaign-v561']]
for r in legs:
 assert not r['lost_baseline'] and r['baseline']['converged']==r['candidate']['converged']==205
 assert r['max_certified_mass_delta']['delta_kg']<1e-5
 assert r['baseline']['priming']==522 and r['candidate']['priming']==450
for r in campaigns:
 assert all(v['same_initial_plans'] and v['same_logical_counts'] and abs(v['score']-548.2546201232)<1e-6 and v['workspace_creations']==17 for v in r['rows'])
finals=[read(p/'device-init-v564/report.json'),read(remote/'validation-v565/report.json')]
for root,r in zip([p/'device-init-v564',remote/'validation-v565'],finals):
 assert r['complete'] and not r.get('error')
 assert '65 passed' in (root/'pytest.log').read_text()
 for name,sha in r['source_sha256'].items():
  if name.startswith(('cpp/','src/','tests/','scripts/')):
   assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==sha,name
assert 'ERROR SUMMARY: 0 errors' in (remote/'validation-v565/memcheck.log').read_text()
downloads={}
for name in ['baseline0','candidate0']:
 report=read(p/'device-init-campaign-v560'/name/'calls.json')[0]['solver_reports'][0]
 downloads[name]={k:report[k] for k in ['adapter_d2h_bytes','adapter_d2h_count','device_numeric_updates']}
assert downloads['baseline0']['adapter_d2h_bytes']==1324772
assert downloads['candidate0']['adapter_d2h_bytes']==515324
files={}
sources={name:p/name for name in ['device-init-v555','device-init-v556','device-init-v558','device-init-v562','device-init-v564','device-init-campaign-v560']}
sources.update({'core-v555':Path('/home/angus/build-spacepdhcg-device-init-v555'),'core-v564':Path('/home/angus/build-spacepdhcg-device-init-v564')})
for label,root in sources.items():
 assert root.exists()
 for path in sorted(root.rglob('*')):
  if not path.is_file():continue
  rel=path.relative_to(root)
  if set(rel.parts)&{'.git','__pycache__','build'}:continue
  files[label+'/'+rel.as_posix()]=path
for name in ['cpp/cuda/src/native_qoco_adapter.cpp','tests/test_gtoc12_gpu_device_initialization.py','tests/test_gtoc12_gpu_qoco.py','tests/test_gtoc12_gpu_scvx.py']:
 files['final-source/'+name]=Path(name)
manifest={name:hashlib.sha256(path.read_bytes()).hexdigest() for name,path in files.items()}
with tarfile.open(target/'local-raw.tar.gz','w:gz') as t:
 for name,path in files.items():t.add(path,arcname=name,recursive=False)
(target/'local-archive-manifest.json').write_text(json.dumps(manifest,indent=2))
before=p/'device-init-v555/source/cpp/cuda/src/native_qoco_adapter.cpp'
assert before.exists()
diff=''.join(difflib.unified_diff(before.read_text().splitlines(True),Path('cpp/cuda/src/native_qoco_adapter.cpp').read_text().splitlines(True),fromfile='experiment-v555',tofile='final-default-v564-v565'))
(target/'experiment-to-final.patch').write_text(diff)
summary=dict(source_base_commit='9f3bad95d8b6f83c8f105476bcd9c606be5c2b64',default_enabled_for_device_extension_backend=True,disable_flag='SPACEPDHCG_TEST_QOCO_DEVICE_INITIALIZATION=0',local_legs=legs[0],lambda_legs=legs[1],local_campaign=campaigns[0],lambda_campaign=campaigns[1],first_fresh_campaign_solve_downloads=downloads,final_tests_per_gpu=65,broader_test_inventory=read(p/'device-initialization-test-inventory.json'),final_local_validation=finals[0],final_lambda_validation=finals[1],limitations=['Numerical initialization is device resident; topology, bounds classification, vendor symbolic setup and route/fleet orchestration still use the CPU.','No established overall speedup: single 225-leg pass per mode; two complete campaigns per mode with overlapping timings.','Earlier runs stopped on obsolete telemetry/priming assertions. Raw failures are retained; physics tolerances were unchanged.','Selected H100 memory check passed. Previously documented full-solver synchronization/race sanitizer failures remain unresolved.','No new fleet record. Incumbent remains 12805.194102488575 weighted kg.','Full replay/campaigns use explicit flag 1 on the experiment build. Final builds change only flag default and verbose host diagnostic guard; 65 final tests per GPU include default, explicit enable and disable.'])
(target/'summary.json').write_text(json.dumps(summary,indent=2))
for pattern in ['*device_initialization*.py','*device_init*v566.py','record_device_initialization_inventory.py']:
 for path in p.glob(pattern):
  if path.name.startswith('launch_'):continue
  shutil.copy2(path,target/path.name)
shutil.copy2(p/'device-initialization-test-inventory.json',target/'test-inventory.json')
(target/'.gitattributes').write_text('* -text whitespace=cr-at-eol,-blank-at-eof\n')
manifest={q.relative_to(target).as_posix():hashlib.sha256(q.read_bytes()).hexdigest() for q in sorted(target.rglob('*')) if q.is_file() and q.name!='files-sha256.json'}
assert all((target/name).stat().st_size<90_000_000 for name in manifest)
(target/'files-sha256.json').write_text(json.dumps(manifest,indent=2))
print(json.dumps(dict(files=len(manifest),raw_bytes=(target/'local-raw.tar.gz').stat().st_size)))
