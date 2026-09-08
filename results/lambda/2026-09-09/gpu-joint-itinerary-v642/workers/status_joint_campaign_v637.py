from pathlib import Path
import json,subprocess
root=Path('/home/ubuntu/spacepdhcg-joint-selection-v632/campaign-v636')
report=json.loads((root/'report.json').read_text())
print(json.dumps({k:report.get(k) for k in ('complete','success','stage','child_pid','exception')}))
for run in report['runs']:print(json.dumps(dict(name=run['name'],seconds=run['process_seconds'],score=run['best']['score_kg'],download_bytes=run['screening']['joint_result_download_bytes'])))
path=root/report['stage']/'report.json'
if path.exists():
    r=json.loads(path.read_text());print(json.dumps({k:r.get(k) for k in ('complete','status','stage','native_solves_completed','seconds')}))
print(subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader'],text=True))
