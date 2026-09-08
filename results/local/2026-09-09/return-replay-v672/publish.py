from pathlib import Path
import hashlib,json,shutil,tarfile
import numpy as np
root=Path('/home/angus/spacepdhcg-return-replay-v672');dest=Path('results/local/2026-09-09/return-replay-v672');dest.mkdir(parents=True,exist_ok=False)
audit=dict(groups={},scope='CPU-created native numerical envelopes, conic reports and independent leg certificates. GPU-generated seeds and assembled first-conic matrices were not captured.',physics_tolerances_changed=False,fleet_promotion=False)
reference=None
for group in ('output','pool-off','ruiz5'):
 report=json.loads((root/group/'report.json').read_text());assert report['complete'] and report['actual_native_calls']==3
 rows=[]
 for row in report['rows']:
  directory=root/group/('repeat_%02d_mesh_candidate1'%row['repeat'])
  envelope=json.loads((directory/'native-input.json').read_text())
  assert hashlib.sha256(json.dumps(envelope,sort_keys=True).encode()).hexdigest()==row['native_input_sha256']
  with np.load(directory/'native-input-arrays.npz') as values:
   arrays={name:values[name] for name in values.files}
   for name,value in arrays.items():assert hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()==envelope['arrays'][name]['sha256']
   if reference is None:reference=arrays
   else:
    assert arrays.keys()==reference.keys()
    for name,value in arrays.items():assert np.array_equal(value,reference[name]),name
  conic=json.loads((directory/'conic-reports.json').read_text())
  if group=='pool-off':assert conic[0]['workspace_creations']==1
  rows.append(dict(repeat=row['repeat'],status=row['status'],iterations=row['iterations'],accepted_iterations=row['accepted_iterations'],solve_seconds=row['solve_seconds'],certified=row['pipeline_leg_certified'],native_input_sha256=row['native_input_sha256'],first_conic=conic[0]))
 audit['groups'][group]=dict(rows=rows,certified=sum(r['certified'] for r in rows),input_hashes=sorted({r['native_input_sha256'] for r in rows}),feature_flags=report['feature_flags'])
 assert sum(r['certified'] for r in rows)==2
assert audit['groups']['output']['input_hashes']==audit['groups']['pool-off']['input_hashes']
(root/'audit.json').write_text(json.dumps(audit,indent=2))
files={p.relative_to(root).as_posix():p for p in root.rglob('*') if p.is_file() and p.suffix!='.lock'}
manifest={n:dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for n,p in sorted(files.items())}
archive=dest/'raw.tar.gz'
with tarfile.open(archive,'w:gz') as tar:
 for name,path in sorted(files.items()):tar.add(path,arcname=name,recursive=False)
with tarfile.open(archive) as tar:
 for m in tar.getmembers():assert m.isfile() and hashlib.sha256(tar.extractfile(m).read()).hexdigest()==manifest[m.name]['sha256']
(dest/'raw-manifest.json').write_text(json.dumps(manifest,indent=2))
for name in ('fixture.json','ruiz5-fixture.json','run.py','ruiz5-run.py','worker.py','pool-off-worker.py','ruiz5-worker.py','audit.json','worker-report.json','pool-off-worker-report.json','ruiz5-worker-report.json'):
 shutil.copy2(root/name,dest/name)
for group in ('output','pool-off','ruiz5'):shutil.copy2(root/group/'report.json',dest/(group+'-report.json'))
shutil.copy2('build/performance/publish_return_replay_v672.py',dest/'publish.py')
(dest/'.gitattributes').write_text('* -text whitespace=cr-at-eol,-blank-at-eof\n')
(dest/'README.md').write_text('''# Reproduced fixed-input return-leg convergence failure

The local GPU mesh campaign's second alternative return leg failed after 40
iterations. Its retained best fleet still passes both full-fleet physics checks.
This diagnostic reconstructs the pinned boundary at MJD 69230–69805 from asteroid
13077 to Earth with exactly the recorded initial mass, 1339.295993103377 kg.

Three calls at unchanged settings produce converged / converged / infeasible.
Disabling retained QOCO workspaces produces the same outcome sequence, with fresh
workspace creation verified in all three conic reports. Five Ruiz equilibration
iterations with objective preservation produce converged / infeasible / converged.
Thus neither disabling the pool nor this scaling setting fixes the failure.
None of these diagnostic settings is promoted as a production workaround.

All nine calls have byte-identical captured numerical arrays. The six unscaled
calls share one complete numerical input-envelope hash; the scaled envelope hash
differs because the equilibration setting changes. The captured envelope excludes
wall-clock deadlines. All six converged legs pass the existing independent CPU
rollout and unchanged acceptance gates; the three failed legs are rejected.
No fleet or score is promoted by this diagnostic.

Differences already appear in first-conic numerical outputs. This isolates
variability below the Python mission-input boundary but does not identify its
root cause: GPU-generated seeds, assembled matrices and vendor factorization
were not compared. Capturing those is the next isolation step. A solver status
of infeasible here is a failed local refinement, not proof that the transfer
has no physically feasible solution.

`audit.json` verifies saved array hashes, envelope hashes and disabled-pool
workspace creation. `raw.tar.gz` retains all nine numerical input packets,
solutions, conic and SCvx histories, logs and exact drivers. `raw-manifest.json`
hashes every member. The source and binaries are the frozen mesh build archived
in `results/lambda/2026-09-09/gpu-joint-mesh-v670/local-raw.tar.gz` (SHA-256
421cd1a63bbd6921ad12194be75b404250bb5be15a58d72da48d2a6efbd89ca6).
Core: 6fd02e87f3e148c0e62aac2bee731bc58397605a1d29ef3aebec95b0b4303c6c.
QOCO: 0cc27a1d8bde1edd74f7ef00e04ec64c2d6c0f8fe526a1d94b04d509a74b3315.
''')
checks={p.relative_to(dest).as_posix():dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in dest.rglob('*') if p.is_file()}
(dest/'sha256.json').write_text(json.dumps(checks,indent=2))
print(json.dumps(dict(archived_files=len(manifest),archive_bytes=archive.stat().st_size,statuses={g:[r['status'] for r in a['rows']] for g,a in audit['groups'].items()})))
