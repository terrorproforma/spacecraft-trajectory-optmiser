from pathlib import Path
import subprocess
root=Path('/home/ubuntu/spacepdhcg-nonfinite-replay-v338');root.mkdir(exist_ok=False)
runner=r'''from pathlib import Path
import sys,subprocess,os,time,json,fcntl,hashlib,ast
root=Path('/home/ubuntu/spacepdhcg-nonfinite-replay-v338')
report=dict(pid=os.getpid(),complete=False,rows=[],campaigns=[])
def save(): (root/'report.json').write_text(json.dumps(report,indent=2))
save()
try:
 validation=json.loads(Path('/home/ubuntu/spacepdhcg-nonfinite-ir-v337/report.json').read_text());assert validation['complete']
 sys.path.insert(0,'/home/ubuntu/spacepdhcg-nonfinite-ir-v337/overlay')
 from audit import problem,audit
 qp=Path('/home/ubuntu/spacepdhcg-arc-snapshot-v307/snapshots/qp-153736-000000.txt');data=problem(qp)
 binary=Path('/home/ubuntu/spacepdhcg-diagnose-v174/final/qoco_snapshot_replay')
 for node in ast.parse(Path('/home/ubuntu/spacepdhcg-diagnose-v174/diagnose.py').read_text()).body:
  if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='runtime' for t in node.targets):runtime=ast.literal_eval(node.value)
 report['qp_sha256']=hashlib.sha256(qp.read_bytes()).hexdigest();save()
 with open('/home/ubuntu/.spacepdhcg-gpu.lock','a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  for name,candidate in [('baseline',False),('guard',True),('guard_repeat',True),('baseline_repeat',False)]:
   lib=Path('/home/ubuntu/spacepdhcg-nonfinite-ir-v337/final/libqoco.so') if candidate else Path('/home/ubuntu/spacepdhcg-diagnose-v174/final/libqoco.so')
   env={k:v for k,v in os.environ.items() if not k.startswith('SPACEPDHCG_TEST_')}
   env.update(LD_LIBRARY_PATH=str(lib.parent)+':'+runtime+'/lib:/usr/local/cuda/lib64',SPACEPDHCG_TEST_QOCO_IPM_GRAPH='1')
   assert str(lib) in subprocess.check_output(['ldd',str(binary)],env=env,text=True)
   start=time.perf_counter();r=subprocess.run([str(binary),str(qp),'16'],env=env,capture_output=True,text=True,timeout=120)
   (root/(name+'.log')).write_text(r.stdout);(root/(name+'.err')).write_text(r.stderr);r.check_returncode()
   records=[json.loads(s[10:]) for s in r.stdout.splitlines() if s.startswith('QP_REPLAY ')];assert len(records)==16
   audits=[dict(iterations=x['iterations'],ir_iterations=x['ir_iterations'],status=x['status'],**audit(data,x)) for x in records]
   row=dict(name=name,library_sha256=hashlib.sha256(lib.read_bytes()).hexdigest(),seconds=time.perf_counter()-start,qualified=sum(x['qualified'] for x in audits),audits=audits)
   report['rows'].append(row);save();print(name,row['qualified'],'/16',flush=True)
 # Complete campaigns obtain the same GPU lock themselves.
 template=Path('/home/ubuntu/spacepdhcg-gpu-execution-v328/v332/run.py').read_text()
 for version,candidate in [(339,True),(340,False)]:
  path=root/f'v{version}';path.mkdir(exist_ok=False)
  source=template.replace('/home/ubuntu/spacepdhcg-gpu-execution-v328/v332',str(path)).replace('gpu_execution_graph_v332','gpu_nonfinite_'+str(version))
  if candidate:source=source.replace('/home/ubuntu/spacepdhcg-diagnose-v174/final/libqoco.so','/home/ubuntu/spacepdhcg-nonfinite-ir-v337/final/libqoco.so')
  source=source.replace("report['kernel_source_sha256']=", "report['nonfinite_candidate']="+repr(candidate)+"\nreport['qoco_guard_source_sha256']="+repr(validation['source_sha256'] if candidate else {})+"\nreport['kernel_source_sha256']=",1)
  (path/'run.py').write_text(source)
  with (path/'runner.log').open('x') as log:
   child=subprocess.Popen(['python3',str(path/'run.py')],stdout=log,stderr=subprocess.STDOUT)
   row=dict(version=version,candidate=candidate,pid=child.pid,complete=False);report['campaigns'].append(row);save()
   row['returncode']=child.wait(timeout=1900)
  run=json.loads((path/'report.json').read_text());assert run['complete'] and run['returncode']==0
  output=json.loads((path/'output/run_report.json').read_text())
  row.update(complete=True,seconds=output['wall_seconds_total'],score=output['best']['score_kg'],official=output['best']['official']['ok'],independent=output['best']['independent']['ok']);save();print(row,flush=True)
 report['complete']=True
except Exception as e:report['error']=repr(e)
save();print('complete',report['complete'],report.get('error'),flush=True)
'''
(root/'run.py').write_text(runner)
with (root/'runner.log').open('x') as log:p=subprocess.Popen(['/home/ubuntu/spacepdhcg/v1/.venv/bin/python',str(root/'run.py')],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
print(p.pid)
