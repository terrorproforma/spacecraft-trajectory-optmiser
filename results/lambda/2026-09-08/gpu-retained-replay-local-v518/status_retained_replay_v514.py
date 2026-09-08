from pathlib import Path
import json,subprocess
root=Path('/home/ubuntu/spacepdhcg-retained-replay-v514')
r=json.loads((root/'report.json').read_text());print(json.dumps({k:r.get(k) for k in ['complete','stage','error','pid','child_pid','stages']}))
print(subprocess.run(['ps','-p',str(r['pid'])+','+str(r.get('child_pid',0)),'-o','pid,etime,args'],text=True,capture_output=True).stdout)
log=root/(r.get('stage','runner')+'.log')
if log.exists():print('\n'.join(log.read_text() .splitlines()[-4:]))
