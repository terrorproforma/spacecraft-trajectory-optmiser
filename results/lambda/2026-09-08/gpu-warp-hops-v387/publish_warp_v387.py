from pathlib import Path
import json,hashlib,shutil,statistics
source=Path('build/performance/retrieved-warp-v387')
root=Path('results/lambda/2026-09-08/gpu-warp-hops-v387')
r=json.loads((source/'report.json').read_text());assert r['complete'] and not r.get('error')
audit=json.loads((source/'audit.json').read_text());assert audit['all_fields_exact_equal'] and audit['compared_outputs']==216
native=json.loads((source/'native-v390/report.json').read_text());assert native['complete']
campaigns=r['campaigns'];assert len(campaigns)==4
assert all(c['screening']==campaigns[0]['screening'] for c in campaigns)
for c in campaigns:
 run=json.loads((source/c['name']/'output/run_report.json').read_text())
 assert run['best']['official']['ok'] and run['best']['independent']['ok']
 c['official']=run['best']['official'];c['independent']=run['best']['independent']
baseline=statistics.median(c['seconds'] for c in campaigns if not c['candidate'])
candidate=statistics.median(c['seconds'] for c in campaigns if c['candidate'])
summary=dict(hardware='Lambda NVIDIA H100 80 GB',complete=True,
 baseline_median_seconds=baseline,candidate_median_seconds=candidate,speedup=baseline/candidate,
 time_reduction_percent=100*(1-candidate/baseline),campaigns=campaigns,
 same_search_counts=True,all_mission_checkers_pass=True,accuracy_audit=audit,
 native_graph_probe=native,source_sha256=r['source_sha256'],runtime_sha256=r['runtime_sha256'],
 scope='ABBA complete one-ship campaigns, two samples per mode. Frozen source from v381 plus the included four-file overlay; the full base source is preserved in gpu-fleet-recovery-v381/raw.tar.gz. Both final mission checkers pass in every campaign. Fleet incumbent unchanged.')
(root/'summary.json').write_text(json.dumps(summary,indent=2))
for name in ['report.json','audit.json','pytest.log','memcheck.log','synccheck.log']:
 shutil.copy2(source/name,root/name)
shutil.copytree(source/'native-v390',root/'native-v390')
shutil.copytree(source/'source-overlay',root/'source-overlay')
for name in ['run_warp_hops_v387.py','prepare_warp_hops_v387.py','compose_warp_hops_v387.py','add_recovery_tests_v387.py','check_warp_native_v390.py','archive_warp_v387.py','retrieve_warp_v387.py','publish_warp_v387.py']:
 shutil.copy2(Path('build/performance')/name,root/name)
(root/'.gitattributes').write_text('* -text\n')
files={p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob('*') if p.is_file() and p.name!='files-sha256.json'}
(root/'files-sha256.json').write_text(json.dumps(files,indent=2))
print(json.dumps(dict(baseline=baseline,candidate=candidate,speedup=baseline/candidate,time_reduction_percent=summary['time_reduction_percent'])))
print((root/'pytest.log').read_text()[-200:])
print((root/'native-v390/racecheck.log').read_text()[-250:])
