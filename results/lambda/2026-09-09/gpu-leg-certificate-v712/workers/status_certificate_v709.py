from pathlib import Path
import json,subprocess
home=Path('/home/ubuntu' if Path('/home/ubuntu').exists() else '/home/angus')
root=home/'spacepdhcg-certificate-v709'
report=json.loads((root/'report.json').read_text())
print(json.dumps({k:v for k,v in report.items() if k not in ('stages','campaigns')}))
print(subprocess.run(['ps','-p',','.join(str(report[k]) for k in ('pid','child_pid') if k in report),'-o','pid,stat,etime,comm'],capture_output=True,text=True).stdout)
print(json.dumps([dict(name=r['name'],returncode=r['returncode'],seconds=r['seconds']) for r in report['stages']]))
for r in report['campaigns']:print(json.dumps(dict(mode=r['mode'],seconds=r['seconds'],score=r['best']['score_kg'],ok=r['best']['ok'])))
for name in ('pytest','pipeline-gpu'):
    if (root/(name+'.log')).exists():print('\n'.join((root/(name+'.log')).read_text().splitlines()[-3:]))
