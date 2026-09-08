from pathlib import Path
import json, subprocess
root=Path('/home/ubuntu/spacepdhcg-resident-options-v507')
r=json.loads((root/'report.json').read_text())
print(json.dumps({k:r.get(k) for k in ['pid','child_pid','stage','complete','error']},indent=2))
print(json.dumps([{k:v for k,v in c.items() if k!='screening'} for c in r.get('campaigns',[])],indent=2))
print(subprocess.run(['ps','-p',str(r['pid'])+','+str(r.get('child_pid',0)),'-o','pid,stat,etime,comm'],capture_output=True,text=True).stdout)
if r.get('error'):print((root/(r.get('stage','runner')+'.log')).read_text()[-1500:])
