from pathlib import Path
import json,subprocess
root=Path('/home/ubuntu/spacepdhcg-preserve-campaign-v538')
r=json.loads((root/'report.json').read_text())
print(json.dumps({k:r.get(k) for k in ['complete','stage','error','pid','child_pid','campaigns']}))
pids=[r['pid']]+([r['child_pid']] if 'child_pid' in r else [])
print(subprocess.run(['ps','-p',','.join(map(str,pids)),'-o','pid,etime,args'],capture_output=True,text=True).stdout)
path=root/'pytest.log'
if path.exists():print('\n'.join(path.read_text().splitlines()[-3:]))
