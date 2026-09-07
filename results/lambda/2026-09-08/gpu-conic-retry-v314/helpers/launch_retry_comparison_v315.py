from pathlib import Path
import subprocess,json
root=Path('/home/ubuntu/spacepdhcg-retry-comparison-v315');root.mkdir(exist_ok=False)
validation=json.loads(Path('/home/ubuntu/spacepdhcg-conic-retry-v314/report.json').read_text())
assert validation['complete'] and validation['arc_qualified']==24
runner=r'''from pathlib import Path
import subprocess,json,time
root=Path('/home/ubuntu/spacepdhcg-retry-comparison-v315')
template=Path('/home/ubuntu/spacepdhcg-collect-resident-campaign-v299/run.py').read_text()
report=dict(pid=__import__('os').getpid(),complete=False,rows=[])
(root/'report.json').write_text(json.dumps(report,indent=2))
for i,mode in enumerate(['retry','baseline','baseline','retry','retry','baseline','baseline','retry']):
 version=316+i;path=Path('/home/ubuntu/spacepdhcg-retry-campaign-v'+str(version));path.mkdir(exist_ok=False)
 integrated='/home/ubuntu/spacepdhcg-conic-retry-v314' if mode=='retry' else '/home/ubuntu/spacepdhcg-collect-resident-v298'
 source=template.replace('/home/ubuntu/spacepdhcg-collect-resident-campaign-v299',str(path)).replace('gpu_collect_resident_campaign_v299','gpu_conic_'+mode+'_v'+str(version)).replace('/home/ubuntu/spacepdhcg-collect-resident-v298',integrated)
 source=source.replace("report['kernel_source_sha256']=json.loads((repo/'source-sha256.json').read_text())", "report['kernel_source_sha256']=json.loads((repo/'source-sha256.json').read_text())\nif (repo/'retry-source-sha256.json').exists():report['kernel_source_sha256'].update(json.loads((repo/'retry-source-sha256.json').read_text()))")
 source=source.replace("source_changes=['resident float32 collection tables and device gather into DP']", "source_changes="+repr(['resident float32 collection tables and device gather into DP']+(['bounded cold retries before conic trust shrink'] if mode=='retry' else [])))
 (path/'run.py').write_text(source)
 with (path/'runner.log').open('x') as log:
  child=subprocess.Popen(['python3',str(path/'run.py')],stdout=log,stderr=subprocess.STDOUT)
  row=dict(version=version,mode=mode,pid=child.pid,complete=False);report['rows'].append(row);(root/'report.json').write_text(json.dumps(report,indent=2))
  row['returncode']=child.wait(timeout=1900)
 run=json.loads((path/'report.json').read_text());assert run['complete'] and run['returncode']==0
 output=json.loads((path/'output/run_report.json').read_text());row.update(complete=True,score=output['best']['score_kg'],official=output['best']['official']['ok'],independent=output['best']['independent']['ok'])
 (root/'report.json').write_text(json.dumps(report,indent=2));print(row,flush=True)
report['complete']=True;(root/'report.json').write_text(json.dumps(report,indent=2))
'''
(root/'run.py').write_text(runner)
with (root/'runner.log').open('x') as log:p=subprocess.Popen(['python3',str(root/'run.py')],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
print(p.pid)
