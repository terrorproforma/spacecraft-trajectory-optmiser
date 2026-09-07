from pathlib import Path
import tarfile,hashlib,json
base=Path('/home/ubuntu/spacepdhcg-finish-barriers-v266');mission=Path('/home/ubuntu/spacepdhcg-finish-cert-v267');trace=Path('/home/ubuntu/spacepdhcg-finish-profile-v268')
for r in [base,mission,trace]:
 report=json.loads((r/'report.json').read_text());assert report['complete'] and 'error' not in report
r=json.loads((mission/'report.json').read_text());assert r['refinement']['certified'] and r['independent']['ok'] and r['official']['ok']
archive=Path('/tmp/finish-barriers-v266-evidence.tar.gz')
with tarfile.open(archive,'w:gz') as t:
 for name in ['report.json','run.py','configure.log','build.log','comparison.log','tests.log','memcheck.log','initcheck.log','synccheck.log','racecheck.log','comparison']:t.add(base/name,arcname='validation/'+name)
 t.add(base/'repo/source-sha256.json',arcname='validation/source-sha256.json')
 for name in ['report.json','run.py','runner.log','plan.json','output']:t.add(mission/name,arcname='mission-v267/'+name)
 for p in trace.iterdir():
  if p.is_file() and p.suffix!='.sqlite':t.add(p,arcname='profile-v268/'+p.name)
 rejected=Path('/home/ubuntu/spacepdhcg-parallel-finish-v260')
 for name in ['report.json','racecheck.log','tests.log','memcheck.log','initcheck.log','synccheck.log']:
  t.add(rejected/name,arcname='rejected-v260/'+name)
 t.add(rejected/'repo/source-sha256.json',arcname='rejected-v260/source-sha256.json')
 for name in json.loads((rejected/'repo/source-sha256.json').read_text()):
  t.add(rejected/'repo'/name,arcname='rejected-v260/source/'+name)
print(json.dumps(dict(path=str(archive),sha256=hashlib.sha256(archive.read_bytes()).hexdigest())))
