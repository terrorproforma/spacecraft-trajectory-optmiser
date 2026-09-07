from pathlib import Path
import json,tarfile,hashlib,statistics
root=Path('/home/ubuntu/spacepdhcg-retry-comparison-v315');r=json.loads((root/'report.json').read_text());assert r['complete'] and len(r['rows'])==8
summary=[]
for row in r['rows']:
 path=Path('/home/ubuntu/spacepdhcg-retry-campaign-v'+str(row['version']));run=json.loads((path/'output/run_report.json').read_text())
 b=run['best'];assert b['official']['ok'] and b['independent']['ok'] and abs(b['independent']['total_mass_kg']-548.2546201232033)<1e-6
 summary.append(dict(version=row['version'],mode=row['mode'],seconds=run['wall_seconds_total'],search_seconds=run['ships'][0]['search']['wall_seconds'],score=b['independent']['total_mass_kg'],official=b['official']['ok'],independent=b['independent']['ok'],max_position_error_km=b['independent']['max_position_error_km'],max_velocity_error_km_s=b['independent']['max_velocity_error_km_s'],max_mass_error_kg=b['independent']['max_mass_error_kg'],screening=run['screening']))
medians={mode:statistics.median(x['seconds'] for x in summary if x['mode']==mode) for mode in ['baseline','retry']}
out=dict(rows=summary,median_seconds=medians,relative_time_change=medians['retry']/medians['baseline']-1,scope='One fixed one-ship campaign; both modes passed all four runs. Does not establish a failure-rate improvement.')
(root/'summary.json').write_text(json.dumps(out,indent=2))
archive=root/'evidence.tar.gz'
with tarfile.open(archive,'w:gz') as t:
 for name in ['run.py','report.json','runner.log','summary.json']:t.add(root/name,arcname='comparison/'+name)
 for row in r['rows']:
  path=Path('/home/ubuntu/spacepdhcg-retry-campaign-v'+str(row['version']));t.add(path,arcname='v'+str(row['version']))
print(json.dumps(dict(path=str(archive),bytes=archive.stat().st_size,sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),medians=medians,relative_time_change=out['relative_time_change'])))
print([(x['version'],x['mode'],x['seconds']) for x in summary])
