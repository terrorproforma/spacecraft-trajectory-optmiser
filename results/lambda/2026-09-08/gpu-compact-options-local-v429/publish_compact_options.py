from pathlib import Path
import json,hashlib,tarfile,shutil,statistics
base=Path('results/lambda/2026-09-08');root=base/'gpu-compact-options-local-v429';root.mkdir(exist_ok=False)
names=['options-once-v421','compact-options-v423','compact-options-v424','compact-options-v425','compact-profile-v427','compact-fleet-v428','compact-options-v429']
archives={}
for name in names:
 source=Path('build/performance')/name;r=json.loads((source/'report.json').read_text());assert r['complete'] and not r.get('error')
 manifest={p.relative_to(source).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(source.rglob('*')) if p.is_file()}
 with tarfile.open(root/(name+'.tar.gz'),'w:gz') as t:
  for member in manifest:t.add(source/member,arcname=member,recursive=False)
 archives[name+'.tar.gz']=manifest
(root/'archive-manifests.json').write_text(json.dumps(archives,indent=2))

def campaign(source,name,candidate):
 r=json.loads((source/name/'output/run_report.json').read_text());b=r['best'];assert b['accepted'] and b['official']['ok'] and b['independent']['ok']
 return dict(name=name,candidate=candidate,seconds=r['wall_seconds_total'],score=b['independent']['weighted_score_fixed_bonus_kg'],screening=r['screening'],official=b['official'],independent=b['independent'])

def comparison(source):
 r=json.loads((source/'report.json').read_text())
 cs=[campaign(source,c['name'],c['candidate']) for c in r['campaigns']]
 logical=lambda c:{k:v for k,v in c['screening'].items() if k not in ['completed_compact_options','compact_option_download_bytes']}
 assert all(logical(c)==logical(cs[0]) for c in cs)
 assert max(c['score'] for c in cs)-min(c['score'] for c in cs)<1e-6
 b=statistics.median(c['seconds'] for c in cs if not c['candidate']);a=statistics.median(c['seconds'] for c in cs if c['candidate'])
 return dict(campaigns=cs,baseline_median_seconds=b,candidate_median_seconds=a,speedup=b/a,less_time_percent=100*(1-a/b),scope='ABBA, two complete CLI runs per mode. Compact comparisons use corrected one-allocation host option baseline. Counts and accepted scores match; not a universal speedup claim.')

summary=comparison(Path('build/performance/compact-options-v425'))
summary.update(hardware='RTX 5090',source_base_commit='18e10cb042e6ca208d0c47be24da944aebbd2c50')
summary['vector_allocation_only']=comparison(Path('build/performance/options-once-v421'))
summary['vector_allocation_only']['scope']='ABBA correction of repeated host vector allocation alone, before compact options; overlapping timings do not establish overall speedup.'
summary['wider']=campaign(Path('build/performance/compact-fleet-v428'),'',True)
summary['profile']=json.loads(Path('build/performance/compact-profile-v427/timers.json').read_text())
summary['default']=campaign(Path('build/performance/compact-options-v429'),'default',True)
validation=json.loads(Path('build/performance/compact-options-v429/report.json').read_text());assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==sha for p,sha in validation['source_sha256'].items())
summary['default_validation']=validation
(root/'summary.json').write_text(json.dumps(summary,indent=2))
recipes=['prepare_options_v421.py','options_compare_boot_v421.py','options-baseline-v421.py','run_options_once_v421.py','check_compact_options_v423.py','prepare_compact_v424.py','check_compact_v424.py','run_compact_v425.py','prepare_profile_compact_v427.py','profile_compact_v427.py','run_compact_profile_v427.py','prepare_compact_fleet_v428.py','run_compact_fleet_v428.py','prepare_compact_final.py','run_compact_v429.py','compare_compact.py','publish_compact_options.py']
for name in recipes:shutil.copy2(Path('build/performance')/name,root/name)
for version in ['426','430']:
 source=Path('build/performance/retrieved-compact-options-v'+version);target=base/('gpu-compact-options-v'+version)
 r=json.loads((source/'report.json').read_text());assert r['complete'] and not r.get('error')
 s=comparison(source) if version=='426' else dict(campaigns=[campaign(source,'default',True)],scope='Final default-on complete campaign and integration tests, not a paired comparison.')
 s.update(hardware='Lambda H100',source_sha256=r['source_sha256'],runtime_sha256=r['runtime_sha256'],source_base='Frozen paired-ephemerides v420 repository plus recorded compact overlay; v426 builds native library, v430 reuses it with default-on Python overlay.')
 if version=='430':assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==sha for p,sha in r['source_sha256'].items())
 (target/'summary.json').write_text(json.dumps(s,indent=2))
 for name in ['report.json','pytest.log','memcheck.log','synccheck.log']+(['probe.log','racecheck.log','configure.log','build.log'] if version=='426' else []):shutil.copy2(source/name,target/name)
 shutil.copytree(source/'source-overlay',target/'source-overlay')
 for prefix in ['run_','archive_','retrieve_']:
  name=prefix+'compact_v'+version+'.py';shutil.copy2(Path('build/performance')/name,target/name)
 shutil.copy2(Path('build/performance')/('prepare_compact_v426.py' if version=='426' else 'prepare_compact_final.py'),target)
for folder in [root,base/'gpu-compact-options-v426',base/'gpu-compact-options-v430']:
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
