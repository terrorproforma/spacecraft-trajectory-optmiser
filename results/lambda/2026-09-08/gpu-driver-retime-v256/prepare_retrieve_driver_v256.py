from pathlib import Path
import tarfile,hashlib,json
base=Path('/home/ubuntu/spacepdhcg-driver-retime-v256');mission=Path('/home/ubuntu/spacepdhcg-driver-cert-v257');trace=Path('/home/ubuntu/spacepdhcg-driver-profile-v258')
for r in [base,mission,trace]:
    report=json.loads((r/'report.json').read_text());assert report['complete'] and 'error' not in report
r=json.loads((mission/'report.json').read_text());assert r['refinement']['certified'] and r['independent']['ok'] and r['official']['ok'];assert r['retime_gpu']['completed_retime_driver_calls']==1
archive=Path('/tmp/driver-retime-v256-evidence.tar.gz')
with tarfile.open(archive,'w:gz') as t:
    for name in ['report.json','run.py','configure.log','build.log','driver-comparison.log','tests.log','memcheck.log','initcheck.log','synccheck.log','driver-comparison']:
        t.add(base/name,arcname='validation/'+name)
    t.add(base/'repo/source-sha256.json',arcname='validation/source-sha256.json')
    for name in ['report.json','run.py','runner.log','plan.json','output']:t.add(mission/name,arcname='mission-v257/'+name)
    for p in trace.iterdir():
        if p.is_file() and p.suffix!='.sqlite':t.add(p,arcname='profile-v258/'+p.name)
print(json.dumps(dict(path=str(archive),sha256=hashlib.sha256(archive.read_bytes()).hexdigest())))
