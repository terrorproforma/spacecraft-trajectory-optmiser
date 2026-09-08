from pathlib import Path
import json
import os
remote=Path('/home/ubuntu').exists()
root=Path('/home/ubuntu' if remote else '/home/angus')/'spacepdhcg-search-campaign-v703'
r=json.loads((root/'report.json').read_text())
alive={}
for key in ('pid','child_pid'):
    if key not in r:continue
    try:os.kill(r[key],0);alive[key]=True
    except ProcessLookupError:alive[key]=False
print(json.dumps(dict(alive=alive,complete=r['complete'],success=r['success'],stage=r.get('stage'),error=r.get('error'),runs=[{k:v for k,v in row.items() if k in ('name','process_seconds','native_solves','nonconverged','orders')} for row in r['runs']])))
for name in ('baseline0','candidate0'):
    p=root/name/'report.json'
    if p.exists():
        d=json.loads(p.read_text());print(json.dumps(dict(name=name,complete=d.get('complete'),stage=d.get('stage'),orders=d.get('orders'),status=d.get('status'),score=d.get('best',{}).get('score_kg'))))
    p=root/name/'native-solves.jsonl'
    if p.exists():
        rows=p.read_text().splitlines();print(json.dumps(dict(name=name,solves=len(rows),last=json.loads(rows[-1]) if rows else None)))
