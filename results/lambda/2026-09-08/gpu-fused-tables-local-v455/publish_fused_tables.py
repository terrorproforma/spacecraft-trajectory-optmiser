from pathlib import Path
import json,hashlib,tarfile,shutil

base=Path('results/lambda/2026-09-08');local=Path('build/performance')
root=base/'gpu-fused-tables-local-v455';root.mkdir(exist_ok=False)
names=['pipeline-profile-v446','fused-tables-v447','fused-tables-v448','fused-tables-profile-v450','fused-tables-profile-v451','fused-tables-micro-v452','fused-tables-v454','fused-tables-fleet-v455']
archives={}
for name in names:
 source=local/name;r=json.loads((source/'report.json').read_text())
 if name=='fused-tables-profile-v450':assert not r['complete'] and 'BlockingIOError' in r['error'] and 'child_pid' not in r
 else:assert r['complete'] and not r.get('error')
 manifest={p.relative_to(source).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(source.rglob('*')) if p.is_file()}
 with tarfile.open(root/(name+'.tar.gz'),'w:gz') as t:
  for member in manifest:t.add(source/member,arcname=member,recursive=False)
 archives[name+'.tar.gz']=manifest
(root/'archive-manifests.json').write_text(json.dumps(archives,indent=2))

def campaign(source,name,candidate):
 r=json.loads((source/name/'output/run_report.json').read_text());b=r['best']
 assert b['accepted'] and b['official']['ok'] and b['independent']['ok']
 return dict(name=name,candidate=candidate,seconds=r['wall_seconds_total'],score=b['independent']['weighted_score_fixed_bonus_kg'],screening=r['screening'],official=b['official'],independent=b['independent'])

def comparison(source):
 r=json.loads((source/'report.json').read_text());s=json.loads((source/'analysis.json').read_text())
 s['campaigns']=[campaign(source,c['name'],c['candidate']) for c in r['campaigns']]
 return s

def micro(source):
 r=json.loads((source/'measurement.json').read_text());assert r['complete']
 r.update(json.loads((source/'analysis.json').read_text()));return r

s=comparison(local/'fused-tables-v448')
s.update(hardware='RTX 5090',source_base_commit='64b57319b2907efa6d8a478e4189c36ce8a9d4ae',isolated=micro(local/'fused-tables-micro-v452'),wider=campaign(local/'fused-tables-fleet-v455','',True),final_validation=json.loads((local/'fused-tables-v454/report.json').read_text()),profile_launch_note='v450 stopped at a held nonblocking GPU lock without a numerical child; v451 captured the trace after v448 was confirmed terminal.')
assert s['wider']['independent']['ships']==4 and s['wider']['independent']['mined_asteroids']==29
assert abs(s['wider']['score']-2088.668592154973)<1e-6
for p,sha in s['final_validation']['source_sha256'].items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==sha,p
(root/'summary.json').write_text(json.dumps(s,indent=2))
recipes=['prepare_fused_tables_v447.py','check_fused_tables_v447.py','prepare_fused_tables_runs.py','run_fused_tables_v448.py','prepare_fused_tables_profile.py','profile_fused_tables_v450.py','run_fused_tables_profile_v450.py','profile_fused_tables_v451.py','run_fused_tables_profile_v451.py','prepare_fused_tables_micro.py','fused_tables_micro.py','run_fused_tables_micro_v452.py','prepare_fused_tables_final.py','check_fused_tables_v454.py','run_fused_tables_fleet_v455.py','analyze_fused_tables.py','publish_fused_tables.py']
for name in recipes:shutil.copy2(local/name,root/name)
for version in ['449','453','456']:
 source=local/('retrieved-fused-tables-v'+version);target=base/('gpu-fused-tables-v'+version)
 r=json.loads((source/'report.json').read_text());assert r['complete'] and not r.get('error')
 if version=='449':s=comparison(source)
 elif version=='453':s=dict(isolated=micro(source),scope='Isolated real-grid constructor replay on unchanged v449 native core; no new full mission.')
 else:
  s=dict(campaigns=[campaign(source,'default',True)],scope='Final default-on source and native build; 26 integration tests, three sanitizers, and full mission.')
  for p,sha in r['source_sha256'].items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==sha,p
 s.update(hardware='Lambda H100',source_sha256=r['source_sha256'],runtime_sha256=r['runtime_sha256'],source_base='v444 repository plus recorded opt-in overlay for v449; v453 reuses v449 with captured grid trace; v456 rebuilds the same fused path enabled by default.')
 (target/'summary.json').write_text(json.dumps(s,indent=2))
 for p in source.glob('*.log'):shutil.copy2(p,target/p.name)
 shutil.copy2(source/'report.json',target/'report.json')
 # The complete captured grid trace is retained in raw.tar.gz, avoiding a
 # duplicate expanded multi-megabyte JSON fixture in the published tree.
 shutil.copytree(source/'source-overlay',target/'source-overlay',ignore=shutil.ignore_patterns('collect-grids.json') if version=='453' else None)
 for prefix in ['run_','archive_','retrieve_']:
  name=prefix+'fused_tables_v'+version+'.py';shutil.copy2(local/name,target/name)
for folder in [root]+[base/('gpu-fused-tables-v'+v) for v in ['449','453','456']]:
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
