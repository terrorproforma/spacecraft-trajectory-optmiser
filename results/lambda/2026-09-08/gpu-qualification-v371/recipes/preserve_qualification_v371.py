from pathlib import Path
import json,shutil,gzip,hashlib,tarfile
root=Path('results/lambda/2026-09-08/gpu-qualification-v371');root.mkdir(exist_ok=False)
(root/'.gitattributes').write_text('* -text\n')
def copy(source,target):
 source=Path(source);target=root/target;target.parent.mkdir(parents=True,exist_ok=True)
 if source.stat().st_size>250000:
  target=target.with_name(target.name+'.gz')
  with target.open('wb') as f:
   with gzip.GzipFile(filename='',mode='wb',fileobj=f,mtime=0) as z:z.write(source.read_bytes())
 else:shutil.copyfile(source,target)
for name in ['interior-nt-replay-v369b','division-trace-v370','kkt-equilibration-replay-v371','kkt-equilibration-probe-v371']:
 for p in sorted((Path('build/performance')/name).iterdir()):
  if p.is_file():copy(p,Path(name)/p.name)
recipes=['prepare_interior_nt_v369.py','repair_interior_nt_v369.py','replay_interior_nt_v369b.py','prepare_division_trace_v370.py','division_trace_v370.cuh','build_division_trace_v370.py','replay_division_trace_v370.py','audit_division_trace_v370.py','kkt_equilibration_v371.cuh','prepare_kkt_equilibration_v371.py','build_kkt_equilibration_v371.py','replay_kkt_equilibration_v371.py','kkt_equilibration_probe_v371.cu','audit_kkt_equilibration_v371.py']
for name in recipes:copy(Path('build/performance')/name,Path('recipes')/name)
for version,name in [(369,'interior-nt'),(370,'division-trace'),(371,'kkt-equilibration')]:
 frozen=Path(f'/home/angus/build-qoco-{name}-v{version}')
 for p in [frozen/'report.json',*frozen.glob('build*.log')]:copy(p,Path(f'frozen-v{version}')/p.name)
 (root/f'frozen-v{version}'/'source-sha256.json').write_text(json.dumps({str(p.relative_to(frozen/'source')):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((frozen/'source').rglob('*')) if p.is_file()},indent=2))
 # Preserve the full prepared source (no binaries/build directory) for independent rebuilding.
 with (root/f'frozen-v{version}'/'source.tar.gz').open('wb') as f:
  with gzip.GzipFile(filename='',mode='wb',fileobj=f,mtime=0) as z:
   with tarfile.open(fileobj=z,mode='w') as t:
    for p in sorted((frozen/'source').rglob('*')):
     if p.is_file():t.add(p,arcname=str(p.relative_to(frozen/'source')))
trace=Path('/home/angus/build-qoco-division-trace-v370')
for name in ['replay.log','replay.err']:copy(trace/name,Path('division-trace-v370')/name)
with (root/'division-trace-v370'/'snapshots.tar.gz').open('wb') as f:
 with gzip.GzipFile(filename='',mode='wb',fileobj=f,mtime=0) as z:
  with tarfile.open(fileobj=z,mode='w') as t:
   for p in sorted((trace/'snapshots').glob('*.bin')):t.add(p,arcname=p.name)
summary={}
import statistics
for name in ['interior-nt-replay-v369b','kkt-equilibration-replay-v371']:
 r=json.loads((root/name/'report.json').read_text());assert r['complete']
 summary[name]=[dict(name=row['name'],qualified=row['qualified'],replays=len(row['audits']),seconds_including_process_and_audit=row['seconds'],median_ipm_iterations=statistics.median(a['iterations'] for a in row['audits']),median_ir_iterations=statistics.median(a['ir_iterations'] for a in row['audits'])) for row in r['rows']]
(root/'summary.json').write_text(json.dumps(summary,indent=2))
print(root)
