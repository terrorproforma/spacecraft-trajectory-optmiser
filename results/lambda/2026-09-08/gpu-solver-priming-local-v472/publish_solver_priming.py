from pathlib import Path
import ast,hashlib,json,shutil,tarfile

p=Path('build/performance');base=Path('results/lambda/2026-09-08')
target=base/'gpu-solver-priming-local-v472';target.mkdir(exist_ok=False)
names=['solver-profile-v457','solver-phase-v458','solver-phase-v460','solver-phase-v461','early-graph-v463','early-graph-v464','early-graph-v465','early-graph-v467','early-graph-v468','early-graph-replay-v470','early-graph-fleet-v472']
archives={}
for name in names:
 source=p/name;r=json.loads((source/'report.json').read_text())
 if name in ['solver-phase-v460','early-graph-v463','early-graph-v464']:
  assert not r['complete'] and r.get('error'),name
 else:assert r['complete'] and not r.get('error'),name
 manifest={q.relative_to(source).as_posix():hashlib.sha256(q.read_bytes()).hexdigest() for q in sorted(source.rglob('*')) if q.is_file()}
 with tarfile.open(target/(name+'.tar.gz'),'w:gz') as t:
  for member in manifest:t.add(source/member,arcname=member,recursive=False)
 archives[name+'.tar.gz']=manifest
(target/'archive-manifests.json').write_text(json.dumps(archives,indent=2))

def read(folder,name):return json.loads((p/folder/name).read_text())
def phases(folder):
 r=read(folder,'phase-summary.json');r.pop('rows');return r

validation=read('early-graph-v468','report.json')
for name,sha in validation['source_sha256'].items():
 if name=='tests/test_gtoc12_gpu_scvx.py':
  assert ast.dump(ast.parse(Path(name).read_text()))==ast.dump(ast.parse((p/'early-graph-v468/source'/name).read_text()))
 else:assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==sha,name
fleet=read('early-graph-fleet-v472','output/run_report.json');b=fleet['best']
assert b['accepted'] and b['official']['ok'] and b['independent']['ok']
assert b['independent']['ships']==4 and b['independent']['mined_asteroids']==29
assert abs(b['independent']['weighted_score_fixed_bonus_kg']-2088.668592155373)<1e-6
summary=dict(default_early_graph=False,hardware='RTX 5090',source_base_commit='fb079591a35d59a1a8305957004eb1d15ffba595',phase_profile=phases('solver-phase-v461'),full_campaign=read('early-graph-v467','analysis.json'),fixed_225_legs=read('early-graph-replay-v470','analysis.json'),validation=validation,wider=dict(process_seconds=read('early-graph-fleet-v472','report.json')['seconds'],cli_seconds=fleet['wall_seconds_total'],screening=fleet['screening'],independent=b['independent'],official=b['official']),limitations=['Phase trace measures host wall time at existing synchronization boundaries, not per-kernel time. Nsight API trace has no kernel timeline on this installation.','Early readiness removes one host-dispatched priming solve; it does not remove that numerical solve. H100 regressions prevent default enablement.','v460 and H100 v459 mission checkers passed but their diagnostic exporter failed on a numpy array. Fixed exporter is retained with v461/v462.','v463 runner failed before a numerical child due to a copy2 argument error; corrected v465 ran.','v464 and H100 v466 had 44 passing tests and one incorrect priming-count assertion for state-origin mode; corrected v468/v469 passed all 45. Final test source differs from v468 only in AST-equivalent line wrapping.','v464 State-controller probe passed memcheck, synccheck and racecheck. This is not full vendor IPM graph sanitizer coverage.'])
(target/'summary.json').write_text(json.dumps(summary,indent=2))
# Recipes are retained as executed, including corrected retry generators.
prefixes=('prepare_solver_','run_solver_','solver_phase_details','analyze_solver_','prepare_early_','run_early_','check_early_','analyze_early_','replay_early_','launch_early_','status_early_','archive_early_','retrieve_early_','archive_solver_','retrieve_solver_','publish_solver_')
for q in p.glob('*.py'):
 if q.name.startswith(prefixes):shutil.copy2(q,target/q.name)
folders=[target]
for tag in ['solver-phase-v459','solver-phase-v462','early-graph-v466','early-graph-v469','early-graph-v471','early-graph-focus-v473']:
 source=p/('retrieved-'+tag);dest=base/('gpu-'+tag);folders.append(dest)
 for name in ['report.json','analysis.json','phase-summary.json','performance-analysis.json']:
  if (source/name).exists():shutil.copy2(source/name,dest/name)
 if tag=='solver-phase-v462':s=phases('retrieved-'+tag)
 elif tag in ['early-graph-v469','early-graph-v471','early-graph-focus-v473']:s=read('retrieved-'+tag,'analysis.json')
 else:s=dict(scope='Retained failed diagnostic/test run; see report and raw archive. No failed run is relabelled as passing.')
 (dest/'summary.json').write_text(json.dumps(dict(hardware='Lambda H100',default_early_graph=False,measurements=s),indent=2))
for folder in folders:
 (folder/'.gitattributes').write_text('* -text whitespace=cr-at-eol\n')
 manifest={q.relative_to(folder).as_posix():hashlib.sha256(q.read_bytes()).hexdigest() for q in sorted(folder.rglob('*')) if q.is_file() and q.name!='files-sha256.json'}
 assert all((folder/name).stat().st_size<90_000_000 for name in manifest)
 (folder/'files-sha256.json').write_text(json.dumps(manifest,indent=2))
 print(folder,len(manifest),'files',sum((folder/name).stat().st_size for name in manifest),'bytes')
for archive,manifest in archives.items():
 with tarfile.open(target/archive) as t:
  assert {m.name for m in t.getmembers()}==set(manifest)
  for name,sha in manifest.items():assert hashlib.sha256(t.extractfile(name).read()).hexdigest()==sha,name
 print(archive,len(manifest),'members verified')
