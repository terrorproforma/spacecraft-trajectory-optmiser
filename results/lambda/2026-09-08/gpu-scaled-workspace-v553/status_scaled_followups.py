from pathlib import Path
import json,subprocess
pids=[]
for name in ['scaled-pool-campaign-v548','scaled-followup-v549','scaled-followup-v550']:
 root=Path('/home/ubuntu/spacepdhcg-'+name)
 path=root/'report.json'
 if not path.exists():print(name,'no report');continue
 r=json.loads(path.read_text());pids.extend([r['pid']]+([r['child_pid']] if r.get('child_pid') else []))
 print(name,json.dumps({k:r[k] for k in ['complete','stage','error','cases'] if k in r}))
 for label in ['baseline','candidate']:
  p=root/label/'summary.json'
  if p.exists():print(label,p.read_text())
 if name.endswith('550'):
  for f in root.glob('*.log'):
   if f.name=='runner.log':continue
   t=f.read_text();print(f.name,t[:900],t[-500:])
print(subprocess.run(['ps','-p',','.join(map(str,pids)),'-o','pid,etime,args'],capture_output=True,text=True).stdout)
