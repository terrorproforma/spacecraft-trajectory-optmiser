from pathlib import Path
import subprocess,sys
name=sys.argv[1]
assert name in ['spacepdhcg-family-refinement-v851','spacepdhcg-family-penalty-v852','spacepdhcg-family-retiming-v853','spacepdhcg-family-retiming-v854','spacepdhcg-family-returns-v855']
code='''from pathlib import Path
import json,subprocess
root=Path.home()/NAME
r=json.loads((root/'report.json').read_text())
o={k:v for k,v in r.items() if k in ('pid','complete','success','native_solves','cache_hits','fleet_checks','stage','rank','last_leg','seconds','error','probes')}
o['attempts']=[dict(rank=a['rank'],certified=a['certified'],legs=a['legs'],certified_legs=a['certified_legs'],seconds=a['seconds'],failures=[{k:v for k,v in f.items() if k in ('leg','from','to','status','diagnostic')} for f in a['failures']],verified_gain=a.get('verified_gain')) for a in r.get('attempts',[])]
o['process']=subprocess.run(['ps','-p',str(r['pid']),'-o','pid,etime,stat,args'],capture_output=True,text=True).stdout
print(json.dumps(o,indent=2))
'''.replace('NAME',repr(name))
r=subprocess.run(['ssh.exe','-i','C:/Users/Angus/.ssh/spacepdhcg-expansion-key.pem','-o','BatchMode=yes','-o','ConnectTimeout=12','ubuntu@192.222.55.229','python3 -'],input=code,text=True,capture_output=True,check=True,timeout=25)
print(r.stdout,r.stderr)
