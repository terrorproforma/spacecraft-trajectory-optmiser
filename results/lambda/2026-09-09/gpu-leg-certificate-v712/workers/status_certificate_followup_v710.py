from pathlib import Path
import json,subprocess
home=Path('/home/ubuntu' if Path('/home/ubuntu').exists() else '/home/angus')
root=home/'spacepdhcg-certificate-v709/followup-v710'
if (root/'report.json').exists():
    r=json.loads((root/'report.json').read_text());print(json.dumps(r))
    print(subprocess.run(['ps','-p',','.join(str(r[k]) for k in ('pid','child_pid') if k in r),'-o','pid,stat,etime,comm'],capture_output=True,text=True).stdout)
for name in ('worker','pytest','benchmark'):
    if (root/(name+'.log')).exists():print(name, '\n'.join((root/(name+'.log')).read_text().splitlines()[-8:]))
