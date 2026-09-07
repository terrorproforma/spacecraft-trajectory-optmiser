from pathlib import Path
import tarfile, hashlib, json
base=Path('/home/ubuntu/spacepdhcg-forward-final-v251')
initial=Path('/home/ubuntu/spacepdhcg-forward-retime-v249')
mission=Path('/home/ubuntu/spacepdhcg-forward-cert-v252')
for r in [base, initial, mission]:
    report=json.loads((r/'report.json').read_text())
    assert report['complete'] and 'error' not in report
r=json.loads((mission/'report.json').read_text())
assert r['refinement']['certified'] and r['independent']['ok'] and r['official']['ok']
assert r['retime_gpu']['completed_retime_forward_calls']>0
archive=Path('/tmp/forward-final-v251-evidence.tar.gz')
with tarfile.open(archive,'w:gz') as t:
    for name in ['report.json','run.py','configure.log','build.log']:
        t.add(initial/name,arcname='native-build-v249/'+name)
    t.add(initial/'repo/source-sha256.json',arcname='native-build-v249/source-sha256.json')
    for name in ['report.json','run.py','benchmark.log','tests.log','memcheck.log','initcheck.log','synccheck.log','measurement','forward-comparison','boundary-profile.json']:
        t.add(base/name,arcname='validation/'+name)
    t.add(base/'repo/source-sha256.json',arcname='validation/source-sha256.json')
    for name in ['report.json','run.py','runner.log','plan.json','output']:
        t.add(mission/name,arcname='mission-v252/'+name)
print(json.dumps(dict(path=str(archive),sha256=hashlib.sha256(archive.read_bytes()).hexdigest())))
