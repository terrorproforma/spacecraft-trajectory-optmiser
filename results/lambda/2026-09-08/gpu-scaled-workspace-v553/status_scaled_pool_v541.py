from pathlib import Path
import json,subprocess
root=Path('/home/ubuntu/spacepdhcg-scaled-pool-v541')
pids=[]
for path in [root/'report.json',root/'repo/build/performance/scaled-pool-v542/report.json',Path('/home/ubuntu/spacepdhcg-scaled-pool-campaign-v544/report.json')]:
 if not path.exists():continue
 r=json.loads(path.read_text());print(str(path),json.dumps({k:r.get(k) for k in ['complete','stage','error','pid','child_pid']}))
 pids.append(r['pid']);pids += [r['child_pid']] if 'child_pid' in r else []
 if path.parent.name=='scaled-pool-v542':
  for label in ['baseline','candidate']:
   result=path.parent/label/'results.json'
   if result.exists():
    try:
     rows=json.loads(result.read_text());print(label,len(rows),sum(q['status']=='converged' for q in rows))
    except json.JSONDecodeError:print(label,'report being written')
print(subprocess.run(['ps','-p',','.join(map(str,set(pids))),'-o','pid,etime,args'],capture_output=True,text=True).stdout)
