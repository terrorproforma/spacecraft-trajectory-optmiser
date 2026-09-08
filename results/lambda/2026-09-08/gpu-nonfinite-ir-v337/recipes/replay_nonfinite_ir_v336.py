from pathlib import Path
import sys,subprocess,os,time,json,fcntl,hashlib
sys.path.insert(0,str(Path('build/performance/qp-ir-v309').resolve()))
from audit import problem,audit
root=Path('build/performance/nonfinite-replay-v336');root.mkdir(exist_ok=False)
qp=Path('build/performance/qp-ir-v309/qp.txt').resolve();data=problem(qp);binary=Path('build/performance/qp-regularization-v169/qoco_snapshot_replay').resolve()
report=dict(pid=os.getpid(),complete=False,qp_sha256=hashlib.sha256(qp.read_bytes()).hexdigest(),rows=[])
with open('/home/angus/.spacepdhcg-gpu.lock','a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 for name,version in [('baseline_graph',137),('guard_graph',336),('guard_graph_repeat',336),('baseline_graph_repeat',137)]:
  lib=Path('/home/angus/build-qoco-nonfinite-ir-v336/final/libqoco.so') if version==336 else Path('/home/angus/build-qoco-gpu-device-ir-v137/final/libqoco.so')
  env={k:v for k,v in os.environ.items() if not k.startswith('SPACEPDHCG_TEST_')}
  env.update(LD_LIBRARY_PATH=str(lib.parent)+':/home/angus/cudss-isolated-0.8.0.10/nvidia/cu12/lib:/usr/local/cuda-12.8/lib64',SPACEPDHCG_TEST_QOCO_IPM_GRAPH='1')
  assert str(lib) in subprocess.check_output(['ldd',str(binary)],env=env,text=True)
  start=time.perf_counter();r=subprocess.run([str(binary),str(qp),'16'],env=env,capture_output=True,text=True,timeout=120)
  (root/(name+'.log')).write_text(r.stdout);(root/(name+'.err')).write_text(r.stderr);r.check_returncode()
  records=[json.loads(s[10:]) for s in r.stdout.splitlines() if s.startswith('QP_REPLAY ')];assert len(records)==16
  audits=[dict(iterations=x['iterations'],ir_iterations=x['ir_iterations'],status=x['status'],**audit(data,x)) for x in records]
  row=dict(name=name,library_sha256=hashlib.sha256(lib.read_bytes()).hexdigest(),seconds=time.perf_counter()-start,qualified=sum(x['qualified'] for x in audits),audits=audits)
  report['rows'].append(row);(root/'report.json').write_text(json.dumps(report,indent=2));print(name,row['qualified'],'/16',flush=True)
report['complete']=True;(root/'report.json').write_text(json.dumps(report,indent=2))
