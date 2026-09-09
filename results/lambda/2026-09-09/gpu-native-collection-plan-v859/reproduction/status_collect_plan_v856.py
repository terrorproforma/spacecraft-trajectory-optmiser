from pathlib import Path
import subprocess
code="""from pathlib import Path
import json,subprocess
home=Path.home();out={}
for name in ('spacepdhcg-native-collect-plan-v856','spacepdhcg-native-collect-plan-bench-v857'):
    path=home/name/'report.json'
    if not path.exists():continue
    r=json.loads(path.read_text())
    item={k:v for k,v in r.items() if k in ('pid','complete','success','stage','child_pid','error','medians','core_sha256','source_manifest_sha256')}
    item['runs']=[dict(name=x['name'],routes=[dict(ship=y['ship'],seconds=y['seconds'],difference=y['max_numeric_difference']) for y in x['routes']]) for x in r.get('runs',[])]
    item['processes']=subprocess.run(['ps','-p',','.join(str(r[k]) for k in ('pid','child_pid') if k in r),'-o','pid,etime,stat,args'],capture_output=True,text=True).stdout
    out[name]=item
print(json.dumps(out,indent=2))
"""
r=subprocess.run(['ssh','-i','/home/angus/.ssh/spacepdhcg-expansion-key.pem','-o','BatchMode=yes','ubuntu@192.222.55.229','python3 -'],input=code,text=True,capture_output=True,check=True,timeout=25)
print('H100',r.stdout,r.stderr)
print('LOCAL');exec(code)
