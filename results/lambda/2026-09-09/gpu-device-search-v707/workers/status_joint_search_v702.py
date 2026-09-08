from pathlib import Path
import json
import os
root=Path('/home/ubuntu/spacepdhcg-joint-search-v702')
r=json.loads((root/'report.json').read_text())
alive={}
for key in ('pid','child_pid'):
    try:os.kill(r[key],0);alive[key]=True
    except ProcessLookupError:alive[key]=False
print(json.dumps(dict(alive=alive,complete=r['complete'],success=r['success'],stage=r.get('stage'),stages=[{k:s[k] for k in ('name','returncode','seconds')} for s in r['stages']],error=r.get('error'))))
for name in ('pytest.log','memcheck.log','racecheck.log','synccheck.log','benchmark.json'):
    p=root/name
    if p.exists():
        if name.endswith('.json'):
            b=json.loads(p.read_text());print(json.dumps(dict(benchmark_complete=b['complete'],comparisons=b['comparisons'])))
        else:print(name,p.read_text()[-1300:])

