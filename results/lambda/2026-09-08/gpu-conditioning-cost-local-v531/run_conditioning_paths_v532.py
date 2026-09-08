
from pathlib import Path
import hashlib,json,os,subprocess,tarfile,time
root=Path('/home/ubuntu/spacepdhcg-conditioning-paths-v532');os.chdir(root)
report=dict(pid=os.getpid(),complete=False,stages=[])
def save():
 t=root/'report.tmp';t.write_text(json.dumps(report,indent=2));t.replace(root/'report.json')
save()
try:
 with tarfile.open('/tmp/conditioning-paths-v532.tar.gz') as t:
  for m in t.getmembers():
   path=(root/m.name).resolve();assert path.is_relative_to(root) and m.isfile()
   path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(t.extractfile(m).read())
 manifest=json.loads((root/'source-sha256.json').read_text())
 assert all(hashlib.sha256((root/n).read_bytes()).hexdigest()==h for n,h in manifest.items())
 scripts=['replay_conditioning_paths_v529.py','inspect_conditioning_v530.py','isolate_conditioning_cost_v531.py']
 for name in scripts:
  path=root/'build/performance'/name;s=path.read_text()
  s=s.replace('/home/angus/build-qoco-soc-step-v358/source','/home/ubuntu/spacepdhcg-step-final-v359/qoco')
  s=s.replace('/home/angus/build-qoco-soc-step-v358/final','/home/ubuntu/spacepdhcg-step-final-v359/final')
  s=s.replace('/home/angus/cudss-isolated-0.8.0.10/nvidia/cu12/lib','/home/ubuntu/spacepdhcg-recovery-v152/cudss/lib')
  s=s.replace('/usr/local/cuda-12.8','/usr/local/cuda').replace('sm_120','sm_90')
  s=s.replace('/home/angus/.spacepdhcg-gpu.lock','/home/ubuntu/.spacepdhcg-gpu.lock')
  if name=='isolate_conditioning_cost_v531.py':s=s.replace("binary=(prior/'qoco_snapshot_replay').resolve()","binary=Path('build/performance/conditioning-paths-v529/qoco_snapshot_replay').resolve()")
  path.write_text(s)
 report['executed_sources']={str(path.relative_to(root)):hashlib.sha256(path.read_bytes()).hexdigest() for path in (root/'build/performance').glob('*.py')}
 headers=Path('/home/ubuntu/spacepdhcg-step-final-v359/qoco')
 report['headers_sha256']={str(path.relative_to(headers)):hashlib.sha256(path.read_bytes()).hexdigest() for directory in ['include','algebra/cuda','lib/qdldl/include','lib/amd'] for path in (headers/directory).glob('*.h')}
 for name in scripts:
  start=time.perf_counter()
  with (root/(name+'.log')).open('x') as log:
   child=subprocess.Popen(['/home/ubuntu/spacepdhcg/v1/.venv/bin/python',str(root/'build/performance'/name)],stdout=log,stderr=subprocess.STDOUT)
   report.update(stage=name,child_pid=child.pid);save();rc=child.wait(timeout=600)
  report['stages'].append(dict(name=name,returncode=rc,seconds=time.perf_counter()-start));save();assert rc==0
 for name in ['conditioning-paths-v529','conditioning-matrices-v530','conditioning-cost-v531']:
  r=json.loads((root/'build/performance'/name/'report.json').read_text());assert r['complete'] and not r.get('error'),r
 report['complete']=True
except Exception as error:report['error']=repr(error)
save()
