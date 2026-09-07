from pathlib import Path
import json,hashlib,tarfile,statistics
root=Path('/home/ubuntu/spacepdhcg-gpu-execution-v328');report=json.loads((root/'report.json').read_text());assert report['complete'] and len(report['rows'])==4
summary=[]
for row in report['rows']:
 path=root/f"v{row['version']}";output=json.loads((path/'output/run_report.json').read_text())
 refs=json.loads((path/'output/ship_01/refinements.json').read_text());legs=[l for r in refs for l in r['refined']['legs']]
 conics=[r for l in legs for r in l.get('conic_reports',[])];graphs=sum(r['solve_seconds'] is None for r in conics)
 assert bool(graphs)==(row['mode']=='graph')
 summary.append(dict(**row,initial_refinement_legs=len(legs),initial_conic_reports=len(conics),initial_graph_reports=graphs,screening=output['screening'],independent_details=output['best']['independent']))
medians={mode:statistics.median(r['seconds'] for r in summary if r['mode']==mode) for mode in ['graph','dispatch']}
audit=dict(rows=summary,median_seconds=medians,time_reduction_percent=100*(1-medians['graph']/medians['dispatch']),scope='Initial top-three refinement graph counts only; timing and final certificates cover complete campaign including retiming.')
(root/'summary.json').write_text(json.dumps(audit,indent=2))
files=[p for p in root.rglob('*') if p.is_file() and 'repo' not in p.relative_to(root).parts]
files+= [root/'repo'/p for p in report['source_sha256']]+[root/'repo/execution-source-sha256.json']
manifest={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
(root/'sha256.json').write_text(json.dumps(manifest,indent=2))
archive=Path('/tmp/gpu-execution-v328-results.tar.gz')
with tarfile.open(archive,'w:gz') as t:
 for p in files+[root/'sha256.json']:t.add(p,arcname=str(p.relative_to(root)))
print(json.dumps(dict(files=len(files),archive=str(archive),bytes=archive.stat().st_size,sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),medians=medians,time_reduction_percent=audit['time_reduction_percent'])))

