from pathlib import Path
import json,subprocess
root=Path('/home/ubuntu/spacepdhcg-workspace-pool-replay-v480');r=json.loads((root/'report.json').read_text())
print(json.dumps({k:r.get(k) for k in ['pid','child_pid','stage','complete','error']}))
print(subprocess.run(['ps','-p',str(r['pid'])+','+str(r.get('child_pid',0)),'-o','pid,stat,etime,comm'],capture_output=True,text=True).stdout)
for name in ['baseline','candidate']:
 path=root/name/'results.json'
 try:
  rows=json.loads(path.read_text());print(json.dumps(dict(name=name,cases=len(rows),seconds=sum(x['seconds'] for x in rows),converged=sum(x['status']=='converged' for x in rows),last_index=rows[-1]['index'] if rows else None)))
 except (FileNotFoundError,json.JSONDecodeError):print(name,'report not ready')
