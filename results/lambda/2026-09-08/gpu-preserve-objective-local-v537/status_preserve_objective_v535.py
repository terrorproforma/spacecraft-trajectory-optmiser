from pathlib import Path
import json,subprocess
root=Path('/home/ubuntu/spacepdhcg-preserve-objective-v535')
r=json.loads((root/'report.json').read_text())
print(json.dumps({k:r.get(k) for k in ['complete','stage','error','pid','child_pid','stages']}))
pids=[r['pid']]+([r['child_pid']] if 'child_pid' in r else [])
path=root/'repo/build/performance/preserve-objective-v536/report.json'
if path.exists():
 q=json.loads(path.read_text());pids.extend([q['pid']]+([q['child_pid']] if 'child_pid' in q else []))
 print(json.dumps({k:q.get(k) for k in ['complete','stage','error']}))
 print(json.dumps([dict(enabled=row['enabled'],iterations=row['iterations'],qualified=sum(a['qualified'] for a in row['audits'])) for row in q['qp']]))
 for label in ['baseline','candidate']:
  path=root/'repo/build/performance/preserve-objective-v536'/label/'results.json'
  if path.exists():
   try:
    rows=json.loads(path.read_text());print(label,len(rows),sum(row['status']=='converged' for row in rows))
   except json.JSONDecodeError:print(label,'report being written')
print(subprocess.run(['ps','-p',','.join(map(str,set(pids))),'-o','pid,etime,args'],capture_output=True,text=True).stdout)
