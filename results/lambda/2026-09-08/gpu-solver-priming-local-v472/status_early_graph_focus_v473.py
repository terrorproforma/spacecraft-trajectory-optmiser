from pathlib import Path
import subprocess,json
root=Path('/home/ubuntu/spacepdhcg-early-graph-focus-v473')
r=json.loads((root/'report.json').read_text())
print(json.dumps({k:r.get(k) for k in ['pid','child_pid','stage','complete','error','stages']},indent=2))
print(subprocess.run(['ps','-p',str(r['pid'])+','+str(r.get('child_pid',0)),'-o','pid,stat,etime,comm'],capture_output=True,text=True).stdout)
for name in ['baseline0','candidate0','candidate1','baseline1']:
 path=root/name/'results.json'
 if path.exists():
  print(name,json.dumps([{k:v for k,v in row.items() if k in ['index','status','seconds','iterations','diagnostic','priming']} for row in json.loads(path.read_text())]))
