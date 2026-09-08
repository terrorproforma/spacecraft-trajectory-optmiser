from pathlib import Path
import json,hashlib,tarfile,shutil,statistics
base=Path('results/lambda/2026-09-08');root=base/'gpu-paired-ephemerides-local-v419';root.mkdir(exist_ok=False)
archives={}
names=['search-profile-v414','paired-ephemerides-v415','paired-ephemerides-v416','paired-fleet-v418','paired-ephemerides-v419','search-profile-v419']
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
def comparison(cs):
 logical=lambda c:{k:v for k,v in c['screening'].items() if k!='completed_element_hops'}
 assert all(logical(c)==logical(cs[0]) for c in cs)
 b=statistics.median(c['seconds'] for c in cs if not c['candidate']);a=statistics.median(c['seconds'] for c in cs if c['candidate'])
 return dict(campaigns=cs,baseline_median_seconds=b,candidate_median_seconds=a,speedup=b/a,less_time_percent=100*(1-a/b),additional_gpu_ephemeris_hops=cs[1]['screening']['completed_element_hops']-cs[0]['screening']['completed_element_hops'],scope='ABBA, two complete CLI runs per mode; overlapping timing ranges do not establish a significant overall speedup. All logical counts match except the expected increase in GPU-built element hops.')
cs=[campaign(Path('build/performance/paired-ephemerides-v416'),n,c) for n,c in [('baseline0',False),('candidate0',True),('candidate1',True),('baseline1',False)]]
summary=comparison(cs);summary['hardware']='RTX 5090';summary['source_base_commit']='2bea87387abc488297553413578daab895621aa6'
summary['wider']=campaign(Path('build/performance/paired-fleet-v418'),'',True)
summary['default']=campaign(Path('build/performance/search-profile-v419'),'',True)
for version in ['v414','v419']:summary['profile_'+version]=json.loads(Path('build/performance/search-profile-'+version+'/timers.json').read_text())
validation=json.loads(Path('build/performance/paired-ephemerides-v419/report.json').read_text());assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==sha for p,sha in validation['source_sha256'].items())
summary['default_validation']=validation
(root/'summary.json').write_text(json.dumps(summary,indent=2))
for name in ['profile_search_v414.py','run_search_profile_v414.py','check_paired_ephemerides_v415.py','run_paired_ephemerides_v416.py','run_paired_fleet_v418.py','check_paired_ephemerides_v419.py','profile_search_v419.py','run_search_profile_v419.py','publish_paired_ephemerides.py']:shutil.copy2(Path('build/performance')/name,root/name)
for version in ['v417','v420']:
 source=Path('build/performance/retrieved-paired-ephemerides-'+version);target=base/('gpu-paired-ephemerides-'+version)
 r=json.loads((source/'report.json').read_text());assert r['complete'] and not r.get('error')
 cs=[campaign(source,c['name'],c['candidate']) for c in r['campaigns']]
 summary=comparison(cs) if version=='v417' else dict(campaigns=cs,scope='Single final default-on confirmation; not a paired speed comparison.')
 summary.update(hardware='Lambda H100',source_sha256=r['source_sha256'],runtime_sha256=r['runtime_sha256'],source_base='Frozen v412 repository plus included Python overlay; native v412 library unchanged.')
 if version=='v420':assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==sha for p,sha in r['source_sha256'].items())
 (target/'summary.json').write_text(json.dumps(summary,indent=2))
 for name in ['report.json','pytest.log','memcheck.log','synccheck.log']:shutil.copy2(source/name,target/name)
 shutil.copytree(source/'source-overlay',target/'source-overlay')
 for prefix in ['run_','prepare_','archive_','retrieve_']:
  name=prefix+'paired_ephemerides_'+version+'.py';shutil.copy2(Path('build/performance')/name,target/name)
for folder in [root,base/'gpu-paired-ephemerides-v417',base/'gpu-paired-ephemerides-v420']:
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
