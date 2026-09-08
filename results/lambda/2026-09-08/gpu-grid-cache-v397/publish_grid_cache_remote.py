from pathlib import Path
import json,hashlib,shutil,statistics,sys
version=sys.argv[1]
source=Path('build/performance/retrieved-grid-cache-'+version)
root=Path('results/lambda/2026-09-08/gpu-grid-cache-'+version)
r=json.loads((source/'report.json').read_text());assert r['complete'] and not r.get('error')
campaigns=r['campaigns']
for c in campaigns:
 run=json.loads((source/c['name']/'output/run_report.json').read_text())
 assert run['best']['accepted'] and run['best']['official']['ok'] and run['best']['independent']['ok']
 c['official']=run['best']['official'];c['independent']=run['best']['independent']
summary=dict(hardware='Lambda NVIDIA H100 80 GB',complete=True,campaigns=campaigns,source_sha256=r['source_sha256'],runtime_sha256=r['runtime_sha256'],
 scope='v397 is the paired cache-off/cache-on experiment. v400 validates the additional entry-count cap. v402 validates the final default-on source, including release-mode native tests. All final mission checkers pass. Base source is frozen v387 plus the included six-file overlay; logical screening requests include cache hits.')
if len(campaigns)==4:
 assert all(c['screening']==campaigns[0]['screening'] for c in campaigns)
 b=statistics.median(c['seconds'] for c in campaigns if not c['candidate']);a=statistics.median(c['seconds'] for c in campaigns if c['candidate'])
 summary.update(baseline_median_seconds=b,candidate_median_seconds=a,speedup=b/a,time_reduction_percent=100*(1-a/b),timing_conclusion='About 1.3% less time in two samples per mode; insufficient evidence of a significant H100 end-to-end speedup.')
(root/'summary.json').write_text(json.dumps(summary,indent=2))
for name in ['report.json','pytest.log','probe.log','eviction.log','memcheck.log','synccheck.log','racecheck.log']:
 shutil.copy2(source/name,root/name)
shutil.copytree(source/'source-overlay',root/'source-overlay')
for name in ['run_grid_cache_'+version+'.py','prepare_grid_cache_'+version+'.py','archive_grid_cache_'+version+'.py','retrieve_grid_cache_'+version+'.py','publish_grid_cache_remote.py']:
 shutil.copy2(Path('build/performance')/name,root/name)
(root/'.gitattributes').write_text('* -text whitespace=cr-at-eol\n')
(root/'files-sha256.json').write_text(json.dumps({p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob('*') if p.is_file() and p.name!='files-sha256.json'},indent=2))
print(version,summary.get('speedup','default/cap validation'),(root/'pytest.log').read_text()[-80:])
