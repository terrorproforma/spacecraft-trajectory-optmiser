from pathlib import Path
import json,hashlib,tarfile,shutil
root=Path('results/lambda/2026-09-08/gpu-stationary-local-v407');root.mkdir(exist_ok=False)
archives={}
for name in ['stationary-v404','stationary-v405','stationary-v406','stationary-v407']:
 source=Path('build/performance')/name;r=json.loads((source/'report.json').read_text());assert r['complete'] and not r.get('error')
 manifest={p.relative_to(source).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(source.rglob('*')) if p.is_file()}
 with tarfile.open(root/(name+'.tar.gz'),'w:gz') as t:
  for member in manifest:t.add(source/member,arcname=member,recursive=False)
 archives[name+'.tar.gz']=manifest
(root/'archive-manifests.json').write_text(json.dumps(archives,indent=2))
source=Path('build/performance/stationary-v407');c=json.loads((source/'comparison.json').read_text());assert not c['lost_baseline'] and not c['lost_prior'] and not c['candidate']['uncertified']
shutil.copy2(source/'comparison.json',root/'summary.json')
shutil.copy2('build/performance/grid-cache-fleet-v403/scvx-calls.json',root/'replay-input.json')
for name in ['check_stationary_v404.py','check_stationary_v406.py','run_stationary_v405.py','run_stationary_v407.py','replay_stationary_v405.py','compare_stationary_replays.py','publish_stationary_replays.py']:shutil.copy2(Path('build/performance')/name,root/name)
(root/'.gitattributes').write_text('* -text whitespace=cr-at-eol\n')
(root/'files-sha256.json').write_text(json.dumps({p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob('*') if p.is_file()},indent=2))
remote=Path('build/performance/retrieved-stationary-v408');target=Path('results/lambda/2026-09-08/gpu-stationary-v408')
for name in ['comparison.json','report.json','pytest.log','probe.log','memcheck.log','synccheck.log','racecheck.log']:shutil.copy2(remote/name,target/('summary.json' if name=='comparison.json' else name))
shutil.copytree(remote/'source-overlay',target/'source-overlay')
for name in ['prepare_stationary_v408.py','run_stationary_v408.py','archive_stationary_v408.py','retrieve_stationary_v408.py','compare_stationary_replays.py']:shutil.copy2(Path('build/performance')/name,target/name)
(target/'.gitattributes').write_text('* -text whitespace=cr-at-eol\n')
(target/'files-sha256.json').write_text(json.dumps({p.relative_to(target).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in target.rglob('*') if p.is_file()},indent=2))
for folder in [root,target]:
 manifest=json.loads((folder/'files-sha256.json').read_text())
 for name,sha in manifest.items():assert hashlib.sha256((folder/name).read_bytes()).hexdigest()==sha
 print(folder,len(manifest),'files verified')
for name,manifest in archives.items():
 with tarfile.open(root/name) as t:
  assert {m.name for m in t.getmembers()}==set(manifest)
  for member,sha in manifest.items():assert hashlib.sha256(t.extractfile(member).read()).hexdigest()==sha
 print(name,len(manifest),'archive members verified')
