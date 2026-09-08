from pathlib import Path
import subprocess,json
print(subprocess.run(['nvidia-smi','--query-gpu=name,driver_version,utilization.gpu,memory.used','--format=csv'],capture_output=True,text=True).stdout)
for p in Path('/home/ubuntu/spacepdhcg-recovery-v152/cudss/include').glob('*version*'):
 print(p,p.read_text()[-2500:])
for version in [549,550,552]:
 root=Path(f'/home/ubuntu/spacepdhcg-scaled-followup-v{version}')
 r=json.loads((root/'report.json').read_text());print(version,{k:r.get(k) for k in ['stage','complete','error','cases']})
 for label in ['previous','candidate','synccheck','racecheck','memcheck']:
  f=root/(label+'.log')
  if f.exists():
   t=f.read_text();print(label,t[:650],t[-400:])
