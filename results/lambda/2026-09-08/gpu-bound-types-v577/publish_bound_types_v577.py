from pathlib import Path
import difflib,hashlib,json,shutil,subprocess,tarfile
p=Path('build/performance');remote=p/'retrieved-bound-types-v577';target=Path('results/lambda/2026-09-08/gpu-bound-types-v577')
def read(path):return json.loads(path.read_text())
roots=[p/'bound-types-v570',remote/'validation-v569',p/'bound-types-campaign-v571',remote/'campaign-v572']
for script,root in zip(['analyze_bound_types_legs.py']*2+['analyze_bound_types_campaign.py']*2,roots):
 subprocess.run(['python3',str(p/script),str(root)],check=True,stdout=subprocess.DEVNULL)
legs=[read(root/'analysis.json') for root in roots[:2]];campaigns=[read(root/'analysis.json') for root in roots[2:]]
for r in legs:
 assert not r['lost_baseline'] and r['baseline']['converged']==r['candidate']['converged']==205
 assert r['max_certified_mass_delta']['delta_kg']<1e-5
 assert r['baseline']['priming']==r['candidate']['priming']==450
for r in campaigns:
 assert len(r['rows'])==4
 assert all(v['same_initial_plans'] and v['same_logical_counts'] and abs(v['score']-548.2546201232)<1e-6 and v['workspace_creations']==17 for v in r['rows'])
final_roots=[p/'bound-types-v575',remote/'validation-v576']
finals=[read(root/'report.json') for root in final_roots]
for root,r in zip(final_roots,finals):
 assert r['complete'] and not r.get('error') and '77 passed' in (root/'pytest.log').read_text()
 for name,sha in r['source_sha256'].items():
  if name.startswith(('cpp/','src/','tests/','scripts/')):assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==sha,name
for root in [p/'bound-types-v570',remote/'validation-v569']:assert '154 passed' in (root/'pytest.log').read_text()
for root in [p/'bound-types-native-v573',remote/'native-v574']:
 r=read(root/'report.json');assert r['complete'] and not r.get('error') and len(r['stages'])==6
for root in [p/'bound-types-v568',remote/'validation-v569']:
 for tool in ['memcheck','synccheck','racecheck']:
  log=(root/('bounds-'+tool+'.log')).read_text()
  assert 'ERROR SUMMARY: 0 errors' in log or 'RACECHECK SUMMARY: 0 hazards' in log
ledger=[]
for root in roots[2:]:
 entries={}
 for name in ['baseline0','candidate0']:
  report=read(root/name/'calls.json')[0]['solver_reports'][0]
  entries[name]={k:report[k] for k in ['adapter_d2h_bytes','adapter_d2h_count','device_numeric_updates']}
 assert entries['baseline0']['adapter_d2h_bytes']==515324 and entries['candidate0']['adapter_d2h_bytes']==244499
 assert entries['baseline0']['adapter_d2h_count']==13 and entries['candidate0']['adapter_d2h_count']==10
 ledger.append(entries)
files={}
for label in ['bound-types-v567','bound-types-v568','bound-types-v570','bound-types-campaign-v571','bound-types-native-v573','bound-types-v575']:
 root=p/label;r=read(root/'report.json');assert r['complete'] and not r.get('error'),label
 for path in sorted(root.rglob('*')):
  if path.is_file():files[label+'/'+path.relative_to(root).as_posix()]=path
for label in ['568','575']:
 path=Path('/home/angus/build-spacepdhcg-bound-types-v'+label+'/final/libspacepdhcg_cuda.so');files['core-v'+label+'/libspacepdhcg_cuda.so']=path
manifest={name:hashlib.sha256(path.read_bytes()).hexdigest() for name,path in files.items()}
with tarfile.open(target/'local-raw.tar.gz','w:gz') as t:
 for name,path in files.items():t.add(path,arcname=name,recursive=False)
(target/'local-archive-manifest.json').write_text(json.dumps(manifest,indent=2))
before=p/'bound-types-v568/source/cpp/cuda/src/native_qoco_adapter.cpp'
diff=''.join(difflib.unified_diff(before.read_text().splitlines(True),Path('cpp/cuda/src/native_qoco_adapter.cpp').read_text().splitlines(True),fromfile='experiment-v568',tofile='default-v575-v576'))
(target/'experiment-to-final.patch').write_text(diff)
summary=dict(source_base_commit='08c1b67701d5a0f26b41870114d4bee8106d74b1',default_enabled_with_device_initialization=True,disable_flag='SPACEPDHCG_TEST_QOCO_DEVICE_BOUND_TYPES=0',local_legs=legs[0],lambda_legs=legs[1],local_campaign=campaigns[0],lambda_campaign=campaigns[1],local_first_fresh_solve_downloads=ledger[0],lambda_first_fresh_solve_downloads=ledger[1],final_tests_per_gpu=77,broad_tests_per_gpu=154,final_local_validation=finals[0],final_lambda_validation=finals[1],limitations=['One-byte bound classes are downloaded for host symbolic sparse assembly; raw bound values remain on CUDA. Topology, vendor setup and route/fleet orchestration still use the CPU.','Two campaigns per mode and one full replay per mode cannot establish a universal speedup. Numerical convergence variation remains.','Standalone classification memcheck, synccheck and racecheck pass. Previously documented full-solver cuDSS instrumentation failures remain unresolved.','No new fleet record; incumbent remains 12805.194102488575 weighted kg.','Replays and campaigns use explicit flag values on the experiment build. Final builds change only the flag default and extend the regression test to cover that default; 77 final tests validate both GPUs.'])
(target/'summary.json').write_text(json.dumps(summary,indent=2))
for path in p.glob('*bound_types*.py'):
 if path.name.startswith(('launch_','status_')):continue
 shutil.copy2(path,target/path.name)
(target/'.gitattributes').write_text('* -text whitespace=cr-at-eol,-blank-at-eof\n*.patch whitespace=-blank-at-eol,-blank-at-eof\n')
manifest={q.relative_to(target).as_posix():hashlib.sha256(q.read_bytes()).hexdigest() for q in sorted(target.rglob('*')) if q.is_file() and q.name!='files-sha256.json'}
assert all((target/name).stat().st_size<90_000_000 for name in manifest)
(target/'files-sha256.json').write_text(json.dumps(manifest,indent=2))
print(json.dumps(dict(files=len(manifest),local_bytes=(target/'local-raw.tar.gz').stat().st_size)))
