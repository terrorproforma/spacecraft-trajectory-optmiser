from pathlib import Path
import subprocess,json,os
root=Path('/home/ubuntu/spacepdhcg-nonfinite-confirm-v341');report=dict(pid=os.getpid(),complete=False,rows=[])
def save(): (root/'report.json').write_text(json.dumps(report,indent=2))
save()
try:
 for v,source_v in [(341,340),(342,339)]:
  old=Path('/home/ubuntu/spacepdhcg-nonfinite-replay-v338')/f'v{source_v}'
  path=root/f'v{v}';path.mkdir(exist_ok=False)
  source=(old/'run.py').read_text().replace(str(old),str(path)).replace('gpu_nonfinite_'+str(source_v),'gpu_nonfinite_'+str(v))
  (path/'run.py').write_text(source)
  with (path/'runner.log').open('x') as log:
   child=subprocess.Popen(['python3',str(path/'run.py')],stdout=log,stderr=subprocess.STDOUT)
   row=dict(version=v,candidate=source_v==339,pid=child.pid,complete=False);report['rows'].append(row);save()
   row['returncode']=child.wait(timeout=1900)
  run=json.loads((path/'report.json').read_text());assert run['complete'] and run['returncode']==0
  output=json.loads((path/'output/run_report.json').read_text())
  row.update(complete=True,seconds=output['wall_seconds_total'],score=output['best']['score_kg'],official=output['best']['official']['ok'],independent=output['best']['independent']['ok']);save();print(row,flush=True)
 report['complete']=True
except Exception as e:report['error']=repr(e)
save()
