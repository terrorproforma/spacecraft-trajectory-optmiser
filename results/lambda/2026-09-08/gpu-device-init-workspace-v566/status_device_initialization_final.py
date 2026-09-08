from pathlib import Path
import json,subprocess
pids=[]
for root in [Path('/home/ubuntu/spacepdhcg-device-init-v557/repo/build/performance')/tag for tag in ['device-init-v559','device-init-v563']]+[Path('/home/ubuntu/spacepdhcg-device-init-campaign-v561')]:
 path=root/'report.json'
 if not path.exists():continue
 r=json.loads(path.read_text());print(root.name,{k:r.get(k) for k in ['stage','complete','error']})
 pids += [r['pid']]+([r['child_pid']] if r.get('child_pid') else [])
 for label in ['baseline','candidate']:
  path=root/label/'summary.json'
  if path.exists():print(label,path.read_text())
 if (root/'pytest.log').exists():print((root/'pytest.log').read_text()[-350:])
print(subprocess.run(['ps','-p',','.join(map(str,set(pids))),'-o','pid,etime,args'],capture_output=True,text=True).stdout)
