from pathlib import Path
import json,subprocess
for tag in ['conditioning-legs-v522','conditioning-campaign-v524']:
 root=Path('/home/ubuntu/spacepdhcg-'+tag);r=json.loads((root/'report.json').read_text())
 print(tag,{k:r.get(k) for k in ['complete','stage','error','pid','child_pid']})
 print(subprocess.run(['ps','-p',str(r['pid'])+','+str(r.get('child_pid',0)),'-o','pid,etime,args'],text=True,capture_output=True).stdout)
 for name in ['baseline','candidate']:
  q=root/name/'summary.json'
  if q.exists():print(name,json.loads(q.read_text()))
