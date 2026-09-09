from pathlib import Path
import base64,json,subprocess
repo=Path(__file__).resolve().parents[2]
script=(repo/'build/performance/run_family_returns_v855.py').read_bytes()
compile(script,'run.py','exec')
code='''from pathlib import Path
import base64,hashlib,json,os,subprocess
home=Path.home();root=home/'spacepdhcg-family-returns-v855';assert not root.exists()
prior=home/'spacepdhcg-family-departures-v850';report=json.loads((prior/'report.json').read_text())
assert report['complete'] and report['success'] and not Path('/proc/'+str(report['pid'])).exists()
core=home/'spacepdhcg-return-cache-v844/final/libspacepdhcg_cuda.so'
qoco=home/'spacepdhcg-retry-conditioning-v686/final/libqoco.so'
assert core.is_file() and qoco.is_file()
root.mkdir();(root/'run.py').write_bytes(base64.b64decode(PAYLOAD))
env={k:v for k,v in os.environ.items() if not k.startswith(('SPACEPDHCG_TEST_','QOCO_REPLAY_'))}
env.update(SPACEPDHCG_GTOC12_CUDA_LIBRARY=str(core),SPACEPDHCG_QOCO_LIBRARY=str(qoco),SPACEPDHCG_GTOC12_DATA=str(home/'spacepdhcg/gtoc12/benchmarks/gtoc12/data'),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',LD_LIBRARY_PATH=str(qoco.parent)+':'+str(core.parent)+':'+str(home/'spacepdhcg-recovery-v152/cudss/lib')+':/usr/local/cuda/lib64',SPACEPDHCG_TEST_GTOC12_CONDITIONING_RETRY='1')
manifest={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [core,qoco,root/'run.py',prior/'report.json',prior/'ship-18-unit/plans.json',prior/'ship-18-bonus/plans.json',prior/'fleet.txt']}
(root/'inputs.json').write_text(json.dumps(manifest,indent=2))
with (root/'worker.log').open('x') as log:
    child=subprocess.Popen([str(home/'spacepdhcg/v1/.venv/bin/python'),str(root/'run.py')],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
(root/'launch.json').write_text(json.dumps({'pid':child.pid,'environment':{k:v for k,v in env.items() if k.startswith('SPACEPDHCG_') or k=='LD_LIBRARY_PATH'}},indent=2))
print(json.dumps({'root':str(root),'pid':child.pid}))
'''.replace('PAYLOAD',repr(base64.b64encode(script).decode()))
args=['ssh.exe','-i','C:/Users/Angus/.ssh/spacepdhcg-expansion-key.pem','-o','BatchMode=yes','-o','ConnectTimeout=12','ubuntu@192.222.55.229','python3 -']
r=subprocess.run(args,input=code,text=True,capture_output=True,timeout=40,check=True)
print(r.stdout,r.stderr)
(repo/'build/performance/family-returns-launch-v855.json').write_text(r.stdout)
