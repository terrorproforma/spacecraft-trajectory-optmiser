from pathlib import Path
import json,hashlib,tarfile,shutil,statistics
root=Path('results/lambda/2026-09-08/gpu-grid-cache-local-v401');root.mkdir(exist_ok=False)
names=['pipeline-timers-v393','pipeline-timers-v394','grid-cache-v395','grid-cache-pipeline-v396','grid-cache-v398','pipeline-timers-v399','grid-cache-v401']
archives={}
for name in names:
 source=Path('build/performance')/name
 report=json.loads((source/'report.json').read_text())
 assert report.get('complete',report.get('returncode')==0) and not report.get('error'),name
 manifest={p.relative_to(source).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(source.rglob('*')) if p.is_file()}
 with tarfile.open(root/(name+'.tar.gz'),'w:gz') as t:
  for member in manifest:t.add(source/member,arcname=member,recursive=False)
 archives[name+'.tar.gz']=manifest
(root/'archive-manifests.json').write_text(json.dumps(archives,indent=2))
runroot=Path('build/performance/grid-cache-pipeline-v396')
r=json.loads((runroot/'report.json').read_text());campaigns=r['campaigns'];assert len(campaigns)==4
for c in campaigns:
 run=json.loads((runroot/c['name']/'output/run_report.json').read_text())
 assert run['best']['official']['ok'] and run['best']['independent']['ok']
 assert c['screening']==campaigns[0]['screening']
baseline=statistics.median(c['seconds'] for c in campaigns if not c['candidate'])
candidate=statistics.median(c['seconds'] for c in campaigns if c['candidate'])
summary=dict(hardware='Local NVIDIA RTX 5090',baseline_median_seconds=baseline,candidate_median_seconds=candidate,speedup=baseline/candidate,time_reduction_percent=100*(1-candidate/baseline),campaigns=campaigns,
 cache_observation=json.loads(Path('build/performance/pipeline-timers-v399/cache-observations.json').read_text())[-1],
 scope='ABBA complete one-ship campaigns with identical binary and cache explicitly off/on; two samples per mode. v398 adds a 256-entry bound, and v401 enables caching by default. Final default configuration passes native parity/eviction/sanitizer tests and 114 selected Python tests. Wider default-mode fleet run is separate. Logical screening requests include cache hits.')
(root/'summary.json').write_text(json.dumps(summary,indent=2))
recipes=['time_pipeline_v393.py','run_pipeline_timers_v393.py','time_pipeline_v394.py','run_pipeline_timers_v394.py','check_grid_cache_v395.py','run_grid_cache_pipeline_v396.py','check_grid_cache_v398.py','time_pipeline_v399.py','run_pipeline_timers_v399.py','check_grid_cache_v401.py','publish_grid_cache_local_v401.py']
for name in recipes:shutil.copy2(Path('build/performance')/name,root/name)
(root/'.gitattributes').write_text('* -text whitespace=cr-at-eol\n')
(root/'files-sha256.json').write_text(json.dumps({p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob('*') if p.is_file()},indent=2))
print(json.dumps(dict(baseline=baseline,candidate=candidate,speedup=baseline/candidate,reduction=summary['time_reduction_percent'],cache=summary['cache_observation'])))
