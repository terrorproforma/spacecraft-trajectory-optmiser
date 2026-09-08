from pathlib import Path
import json,subprocess
root=Path('/home/ubuntu/spacepdhcg-conditioning-v520');r=json.loads((root/'report.json').read_text())
print(json.dumps({k:r.get(k) for k in ['complete','error','stage','pid','child_pid']}))
print(subprocess.run(['ps','-p',str(r['pid'])+','+str(r.get('child_pid',0)),'-o','pid,etime,args'],text=True,capture_output=True).stdout)
for folder in sorted(root.glob('origin*')):
 if (folder/'summary.json').exists():print(folder.name,json.loads((folder/'summary.json').read_text()))
