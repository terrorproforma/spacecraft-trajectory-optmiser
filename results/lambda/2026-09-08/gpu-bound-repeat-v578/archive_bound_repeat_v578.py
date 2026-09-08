from pathlib import Path
import hashlib,json,statistics,tarfile
base=Path('/home/ubuntu/spacepdhcg-bound-types-v569/repo/build/performance')
first=base/'bound-types-v569';repeat=base/'bound-repeat-v575'
r=json.loads((repeat/'report.json').read_text());assert r['complete'] and not r.get('error')
groups={mode:[json.loads((root/mode/'results.json').read_text()) for root in [first,repeat]] for mode in ['baseline','candidate']}
reference=groups['baseline'][0];qualified={x['index'] for x in reference if x['status']=='converged'}
assert len(qualified)==205
max_delta=0.0
for pairs in groups.values():
 for rows in pairs:
  assert [x['index'] for x in rows]==list(range(225))
  assert {x['index'] for x in rows if x['status']=='converged'}==qualified
  assert all(x.get('certified') for x in rows if x['status']=='converged')
  max_delta=max(max_delta,max(abs(rows[i]['certificate']['final_mass_kg']-reference[i]['certificate']['final_mass_kg']) for i in qualified))
times={mode:[sum(x['seconds'] for x in rows) for rows in pairs] for mode,pairs in groups.items()}
medians={mode:statistics.median(values) for mode,values in times.items()}
analysis=dict(scope='Two 225-leg passes per mode on the same frozen H100 binary, first A/B then B/A after campaign validation. Solver time excludes certification. Repeated due to convergence/timing variability.',seconds=times,medians=medians,less_time_percent=100*(1-medians['candidate']/medians['baseline']),qualified_per_pass=205,max_mass_delta_kg=max_delta,source_core_sha256='88134a3877f876700b6ca97b95d862817630ca07b090f8687f7bc1787b0a4917')
assert max_delta<1e-5
(repeat/'analysis.json').write_text(json.dumps(analysis,indent=2))
files={path.relative_to(repeat).as_posix():path for path in sorted(repeat.rglob('*')) if path.is_file() and '__pycache__' not in path.parts}
manifest={name:hashlib.sha256(path.read_bytes()).hexdigest() for name,path in files.items()}
meta=repeat/'archive-manifest.json';meta.write_text(json.dumps(manifest,indent=2))
output=Path('/home/ubuntu/bound-repeat-results-v578.tar.gz');assert not output.exists()
with tarfile.open(output,'w:gz') as t:
 for name,path in files.items():t.add(path,arcname=name,recursive=False)
 t.add(meta,arcname='archive-manifest.json',recursive=False)
print(json.dumps(dict(path=str(output),sha256=hashlib.sha256(output.read_bytes()).hexdigest(),bytes=output.stat().st_size,analysis=analysis)))
