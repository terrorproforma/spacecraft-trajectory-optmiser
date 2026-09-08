from pathlib import Path
import json,hashlib,tarfile,shutil,statistics
base=Path('results/lambda/2026-09-08');root=base/'gpu-earth-beam-local-v443';root.mkdir(exist_ok=False)
local=Path('build/performance')
names=['earth-beam-v432','earth-beam-v433','earth-beam-v435','earth-beam-micro-v436','earth-beam-fleet-v438','earth-beam-v440','earth-beam-v441','earth-beam-default-v441','earth-beam-v443','earth-beam-default-v443']
names += ['earth-beam-style-v445']
archives={}
for name in names:
 source=local/name;r=json.loads((source/'report.json').read_text());assert r['complete'] and not r.get('error')
 manifest={p.relative_to(source).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(source.rglob('*')) if p.is_file()}
 with tarfile.open(root/(name+'.tar.gz'),'w:gz') as t:
  for member in manifest:t.add(source/member,arcname=member,recursive=False)
 archives[name+'.tar.gz']=manifest
(root/'archive-manifests.json').write_text(json.dumps(archives,indent=2))

def campaign(source,name,candidate):
 r=json.loads((source/name/'output/run_report.json').read_text());b=r['best'];assert b['accepted'] and b['official']['ok'] and b['independent']['ok']
 return dict(name=name,candidate=candidate,seconds=r['wall_seconds_total'],score=b['independent']['weighted_score_fixed_bonus_kg'],screening=r['screening'],official=b['official'],independent=b['independent'])

def comparison(sources):
 cs=[]
 for source in sources:
  r=json.loads((source/'report.json').read_text())
  for c in r['campaigns']:
   row=campaign(source,c['name'],c['candidate']);row['batch']=source.name;cs.append(row)
 logical=lambda c:{k:v for k,v in c['screening'].items() if k not in ['completed_element_hops','completed_earth_beam_rows','earth_beam_download_bytes']}
 assert all(logical(c)==logical(cs[0]) for c in cs)
 assert max(c['score'] for c in cs)-min(c['score'] for c in cs)<1e-6
 b=statistics.median(c['seconds'] for c in cs if not c['candidate']);a=statistics.median(c['seconds'] for c in cs if c['candidate'])
 return dict(campaigns=cs,baseline_median_seconds=b,candidate_median_seconds=a,speedup=b/a,less_time_percent=100*(1-a/b),scope='Two ABBA batches, four runs per mode combined. All original logical counters match except expected migration to device ephemerides. Overlapping timings and variable refinement mean this is not a universal speedup claim.')

def micro(source):
 r=json.loads((source/'measurement.json').read_text());rows=[x for x in r['rows'] if not x['name'].startswith('warm')]
 b=statistics.median(c['seconds'] for c in rows if not c['candidate']);a=statistics.median(c['seconds'] for c in rows if c['candidate'])
 r.update(baseline_median_seconds=b,candidate_median_seconds=a,speedup=b/a,less_time_percent=100*(1-a/b));return r

s=comparison([local/'earth-beam-v433',local/'earth-beam-v440']);s.update(hardware='RTX 5090',source_base_commit='2fa4348bd4f3c0d1f2bb842d48948cc4bdad459e')
s['stage_analysis']=[json.loads((local/n/'analysis.json').read_text()) for n in ['earth-beam-v433','earth-beam-v440']]
s['isolated']=micro(local/'earth-beam-micro-v436')
s['isolated']['recipe_note']='The saved reproducer adds an optional output-directory environment override after the local measurement. That measurement used the unchanged default directory; numerical operations and inputs are unchanged.'
s['wider']=campaign(local/'earth-beam-fleet-v438','',True)
s['default']=campaign(local/'earth-beam-default-v443','default',True)
s['native_validation']=json.loads((local/'earth-beam-v441/report.json').read_text())
validation=json.loads((local/'earth-beam-style-v445/report.json').read_text());assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==sha for p,sha in validation['source_sha256'].items())
s['final_validation']=validation
s['test_style_note']='After v443/v444, the test zip explicitly uses strict=True following its equal-length assertion. Production code is unchanged; the final local 34-test suite and Ruff pass.'
(root/'summary.json').write_text(json.dumps(s,indent=2))
recipes=['prepare_earth_beam_v431.py','check_earth_beam_v431.py','prepare_earth_beam_v432.py','check_earth_beam_v432.py','prepare_earth_beam_v435.py','check_earth_beam_v435.py','prepare_earth_beam_runs.py','run_earth_beam_v433.py','earth_beam_micro_v436.py','prepare_earth_beam_micro_v436.py','run_earth_beam_micro_v436.py','prepare_earth_beam_followup.py','run_earth_beam_fleet_v438.py','prepare_earth_beam_repeats.py','run_earth_beam_v440.py','prepare_earth_beam_final.py','check_earth_beam_v441.py','run_earth_beam_default_v441.py','analyze_earth_beam.py','publish_earth_beam.py']
recipes += ['prepare_earth_beam_empty_final.py','check_earth_beam_v443.py','run_earth_beam_default_v443.py']
recipes += ['prepare_earth_beam_style_v445.py','check_earth_beam_style_v445.py']
for name in recipes:shutil.copy2(local/name,root/name)
for version in ['434','437','439','442','444']:
 source=local/('retrieved-earth-beam-v'+version);target=base/('gpu-earth-beam-v'+version)
 r=json.loads((source/'report.json').read_text());assert r['complete'] and not r.get('error')
 if version=='434':s=comparison([source,local/'retrieved-earth-beam-v439'])
 elif version=='437':s=dict(isolated=micro(source),scope='Native scoring/ranking probe and isolated first-level timing. No complete mission generated in this follow-up.')
 elif version=='439':s=dict(campaigns=[campaign(source,c['name'],c['candidate']) for c in r['campaigns']],stage_analysis=json.loads((source/'analysis.json').read_text()),scope='Second ABBA batch; combined comparison is in v434 summary.')
 else:s=dict(campaigns=[campaign(source,'default',True)],scope='Final default-on validation, not a paired speed comparison.')
 s.update(hardware='Lambda H100',source_sha256=r['source_sha256'],runtime_sha256=r['runtime_sha256'],source_base='v430 repository plus recorded overlay for v434; v437 reuses v434 source/runtime with native test and isolated timing script; v439 repeats frozen v434 numerical source; final v442 rebuilds default source and counter membership fix.')
 if version=='434':s['stage_analysis']=[json.loads((local/('retrieved-earth-beam-v'+v)/'analysis.json').read_text()) for v in ['434','439']]
 if version=='444':
  for p,sha in r['source_sha256'].items():
   if p=='tests/test_gtoc12_gpu_beam.py':
    old=(source/'source-overlay'/p).read_text();assert old.replace('zip(actual, expected):','zip(actual, expected, strict=True):')==Path(p).read_text()
   else:assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==sha,p
 if version=='444':s['source_base']='Frozen v442 source/runtime plus final Python guard for zero-limit evaluation telemetry; tests and default campaign repeat on unchanged native binary.'
 (target/'summary.json').write_text(json.dumps(s,indent=2))
 for p in source.glob('*.log'):shutil.copy2(p,target/p.name)
 shutil.copy2(source/'report.json',target/'report.json');shutil.copytree(source/'source-overlay',target/'source-overlay')
 for prefix in ['run_','archive_','retrieve_']:
  name=prefix+'earth_beam_v'+version+'.py';shutil.copy2(local/name,target/name)
 prepare={'434':'prepare_earth_beam_runs.py','437':'prepare_earth_beam_followup.py','439':'prepare_earth_beam_repeats.py','442':'prepare_earth_beam_final.py','444':'prepare_earth_beam_empty_final.py'}[version]
 shutil.copy2(local/prepare,target/prepare)
for folder in [root]+[base/('gpu-earth-beam-v'+v) for v in ['434','437','439','442','444']]:
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
