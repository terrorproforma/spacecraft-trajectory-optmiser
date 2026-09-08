from pathlib import Path
import hashlib,json,shutil,tarfile
p=Path('build/performance');base=Path('results/lambda/2026-09-08')
target=base/'gpu-workspace-pool-local-v490'
assert not target.exists()
def read(folder,name='report.json'):return json.loads((p/folder/name).read_text())
local=read('workspace-pool-v490');remote=read('retrieved-workspace-pool-v491')
for report in [local,remote]:
 assert report['complete'] and not report.get('error')
 for name,sha in report['source_sha256'].items():assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==sha,name
for folder in ['workspace-pool-v490','retrieved-workspace-pool-v491']:
 root=p/folder
 assert '54 passed' in (root/'pytest.log').read_text()
 for tool in ['memcheck','synccheck','racecheck']:
  log=(root/(tool+'.log')).read_text()
  assert '32' in log and ('ERROR SUMMARY: 0 errors' in log or 'RACECHECK SUMMARY: 0 hazards' in log),(folder,tool)
fleet=read('workspace-pool-fleet-v488','output/run_report.json');b=fleet['best']
assert b['accepted'] and b['official']['ok'] and b['independent']['ok']
target.mkdir()
names=['workspace-pool-v474','workspace-pool-v475','workspace-pool-v476','workspace-pool-v477','workspace-pool-replay-v479','workspace-pool-sanitizers-v481','workspace-pool-v482','workspace-pool-v483','workspace-pool-replay-v485','workspace-pool-fleet-v488','workspace-pool-v489','workspace-pool-v490']
archives={}
for name in names:
 source=p/name;r=read(name)
 if name in ['workspace-pool-v474','workspace-pool-v475']:assert not r['complete'] and r.get('error'),name
 else:assert r['complete'] and not r.get('error'),name
 manifest={q.relative_to(source).as_posix():hashlib.sha256(q.read_bytes()).hexdigest() for q in sorted(source.rglob('*')) if q.is_file()}
 with tarfile.open(target/(name+'.tar.gz'),'w:gz') as t:
  for member in manifest:t.add(source/member,arcname=member,recursive=False)
 archives[name+'.tar.gz']=manifest
(target/'archive-manifests.json').write_text(json.dumps(archives,indent=2))
summary=dict(default_pool=True,eligible='Zero-Ruiz native QOCO replay with IPM/outer graphs; no snapshot directory or disabled IPM initialization.',capacity=8,hardware='RTX 5090',source_base_commit='64bf2307ecd09d1ecd686692e3387a98e9c941ce',complete_campaign=read('workspace-pool-v483','analysis.json'),fixed_225_legs=read('workspace-pool-replay-v485','analysis.json'),four_entry_campaign=read('workspace-pool-v477','analysis.json'),four_entry_replay=read('workspace-pool-replay-v479','analysis.json'),validation=local,wider=dict(process_seconds=read('workspace-pool-fleet-v488')['seconds'],cli_seconds=fleet['wall_seconds_total'],screening=fleet['screening'],independent=b['independent'],official=b['official']),limitations=['v474/v475 failed because vendor numeric metadata was drained after its producer stream was destroyed. Release now drains before stream destruction.','H100 v478/v484 campaign mode labels were wrong: all runs had pooling disabled. They measure baseline variability only. Explicit pool tests and the separate v480/v486 replay runners are unaffected.','Eight entries bound workspace count, not bytes. Peak retained GPU memory was not established.','Sanitizers cover native rebind/assembly, not all vendor cuDSS/IPM graph execution.','Metadata materialization still transfers vendor data to the CPU. SCvx graph/buffer setup and Python orchestration remain.','The fixed replay preserves 205 certificates; the remaining 20 attempts are unsuccessful. No global infeasibility proof or new fleet record is claimed.'])
(target/'summary.json').write_text(json.dumps(summary,indent=2))
prefixes=('prepare_workspace_pool','check_workspace_pool','run_workspace_pool','replay_workspace_pool','analyze_workspace_pool','launch_workspace_pool','status_workspace_pool','archive_workspace_pool','retrieve_workspace_pool','collect_workspace_pool','publish_workspace_pool')
for q in p.glob('*.py'):
 if q.name.startswith(prefixes) or q.name in ['solver_phase_details.py','analyze_early_graph_replay.py']:shutil.copy2(q,target/q.name)
folders=[target]
for tag in ['workspace-pool-v478','workspace-pool-replay-v480','workspace-pool-v484','workspace-pool-replay-v486','workspace-pool-v487','workspace-pool-v491']:
 source=p/('retrieved-'+tag);dest=base/('gpu-'+tag);folders.append(dest)
 for name in ['report.json','analysis.json']:
  if (source/name).exists():shutil.copy2(source/name,dest/name)
 scope='Final default-enabled build and one-ship mission confirmation.' if tag.endswith('491') else ('Baseline variability only; campaign labels are incorrect.' if tag.endswith(('478','484')) else 'Same-binary pool off/on comparison.')
 (dest/'summary.json').write_text(json.dumps(dict(hardware='Lambda H100',scope=scope,capacity=4 if tag.endswith(('478','480')) else 8,default_pool=tag.endswith('491'),measurements=read('retrieved-'+tag,'analysis.json') if (source/'analysis.json').exists() else read('retrieved-'+tag)),indent=2))
for folder in folders:
 (folder/'.gitattributes').write_text('* -text whitespace=cr-at-eol\n')
 manifest={q.relative_to(folder).as_posix():hashlib.sha256(q.read_bytes()).hexdigest() for q in sorted(folder.rglob('*')) if q.is_file() and q.name!='files-sha256.json'}
 assert all((folder/name).stat().st_size<90_000_000 for name in manifest)
 (folder/'files-sha256.json').write_text(json.dumps(manifest,indent=2))
 print(folder,len(manifest),'files',sum((folder/name).stat().st_size for name in manifest),'bytes',flush=True)
for archive,manifest in archives.items():
 with tarfile.open(target/archive) as t:
  assert {m.name for m in t.getmembers()}==set(manifest)
  for name,sha in manifest.items():assert hashlib.sha256(t.extractfile(name).read()).hexdigest()==sha,name
 print(archive,len(manifest),'members verified',flush=True)
