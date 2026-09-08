from pathlib import Path
import json
import subprocess
home=Path('/home/ubuntu' if Path('/home/ubuntu').exists() else '/home/angus')
root=home/'spacepdhcg-campaign-profile-v708'
report=json.loads((root/'profile.json').read_text())
print(json.dumps(report))
print(subprocess.run(['ps','-p',','.join(str(report[k]) for k in ('pid','child_pid') if k in report),'-o','pid,stat,etime,comm'],capture_output=True,text=True).stdout)
if report['complete'] and (root/'profile.txt').exists():
    print('\n'.join((root/'profile.txt').read_text().splitlines()[:59]))
