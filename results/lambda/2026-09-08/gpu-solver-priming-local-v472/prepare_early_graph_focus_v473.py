from pathlib import Path
import json

p=Path('build/performance')
old=(p/'run_early_graph_v471.py').read_text()
head=old[:old.index(' shutil.copytree(')]
body=old[old.index(" cmake='/home/ubuntu/"):]
body=body.replace(" for name,mode in [('baseline','0'),('candidate','1')]:\n  run(name,[py,'-c',replay,mode,str(root/name)],1200)"," for name,mode in [('baseline0','0'),('candidate0','1'),('candidate1','1'),('baseline1','0')]:\n  run(name,[py,'-c',replay,mode,str(root/name),'44,201,98'],300)")
head=head.replace("root=Path('/home/ubuntu/spacepdhcg-early-graph-v471');repo=root/'repo';baseline=Path('/home/ubuntu/spacepdhcg-gpu-execution-v328/repo')", "root=Path('/home/ubuntu/spacepdhcg-early-graph-focus-v473');repo=Path('/home/ubuntu/spacepdhcg-early-graph-v471/repo')")
source=head+" report['source_report']=json.loads(Path('/home/ubuntu/spacepdhcg-early-graph-v471/report.json').read_text())\n"+body
(p/'run_early_graph_focus_v473.py').write_text(source)
launch="""from pathlib import Path
import subprocess,json
root=Path('/home/ubuntu/spacepdhcg-early-graph-focus-v473');root.mkdir(exist_ok=False)
(root/'run.py').write_text(SOURCE)
with (root/'runner.log').open('x') as log:
 child=subprocess.Popen(['python3',str(root/'run.py')],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps(dict(pid=child.pid,root=str(root))))
""".replace('SOURCE',repr(source))
(p/'launch_early_graph_focus_v473.py').write_text(launch)
status="""from pathlib import Path
import subprocess,json
root=Path('/home/ubuntu/spacepdhcg-early-graph-focus-v473')
r=json.loads((root/'report.json').read_text())
print(json.dumps({k:r.get(k) for k in ['pid','child_pid','stage','complete','error','stages']},indent=2))
print(subprocess.run(['ps','-p',str(r['pid'])+','+str(r.get('child_pid',0)),'-o','pid,stat,etime,comm'],capture_output=True,text=True).stdout)
for name in ['baseline0','candidate0','candidate1','baseline1']:
 path=root/name/'results.json'
 if path.exists():
  print(name,json.dumps([{k:v for k,v in row.items() if k in ['index','status','seconds','iterations','diagnostic','priming']} for row in json.loads(path.read_text())]))
"""
(p/'status_early_graph_focus_v473.py').write_text(status)
