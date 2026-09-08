from pathlib import Path
import json,subprocess
root=Path('/home/ubuntu/spacepdhcg-device-init-v557');pids=[]
for path in [root/'report.json',root/'repo/build/performance/device-init-v556/report.json',root/'repo/build/performance/device-init-v559/report.json']:
 if not path.exists():continue
 r=json.loads(path.read_text());pids += [r['pid']]+([r['child_pid']] if r.get('child_pid') else [])
 print(path, {k:r.get(k) for k in ['stage','complete','pid','child_pid']},str(r.get('error',''))[:250])
 if r.get('error'):
  log=path.parent/(r['stage']+'.log')
  if log.exists():print(log.read_text()[-1200:])
print(subprocess.run(['ps','-p',','.join(map(str,set(pids))),'-o','pid,etime,args'],capture_output=True,text=True).stdout)
