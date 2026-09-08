from pathlib import Path
import json,hashlib,os,subprocess,tarfile,shutil
p=Path('build/performance');tag='conditioning-v520'
r=(p/'run_retained_replay_legs_v517.py').read_text().replace('retained-replay-legs-v517',tag).replace('replay_workspace_pool.py','conditioning_cases.py')
a=r.index(" for name,mode in");b=r.index(" report['complete']=True",a)
r=r[:a]+" for origin,ruiz in [(0,0),(1,0),(0,2),(1,2),(0,5),(1,5)]:\n  name=f'origin{origin}-ruiz{ruiz}'\n  run(name,[py,'-c',replay,str(root/name),str(origin),str(ruiz)],600)\n"+r[b:]
r=r.replace("def save():(root/'report.json').write_text(json.dumps(report,indent=2))", "def save():\n temp=root/'report.tmp';temp.write_text(json.dumps(report,indent=2));temp.replace(root/'report.json')")
(p/'run_conditioning_v520.py').write_text(r)
sources={q.as_posix() for q in Path('src/spacepdhcg/gtoc12').glob('*.py')}|{'cpp/cuda/src/native_qoco_adapter.cpp','cpp/cuda/src/gtoc12_scvx.cu','cpp/cuda/src/gtoc12_qoco.cu','build/performance/conditioning_cases.py','build/performance/grid-cache-fleet-v403/scvx-calls.json'}
manifest={name:hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in sorted(sources)}
mp=p/(tag+'-source-sha256.json');mp.write_text(json.dumps(manifest,indent=2))
local=p/'conditioning-v519'
for name in sorted(sources):
 dest=local/'source'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(name,dest)
shutil.copy2(p/'run_conditioning_v519.py',local/'run.py')
(local/'source-sha256.json').write_text(json.dumps(manifest,indent=2))
(local/'provenance.json').write_text(json.dumps(dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),qoco_sha256=hashlib.sha256(Path('/home/angus/build-qoco-soc-step-v358/final/libqoco.so').read_bytes()).hexdigest(),scope='Eight-case diagnostic, 30-second per-leg pilot budget in every mode. Pool disabled for an equal comparison across Ruiz modes; original mathematical tolerances unchanged.'),indent=2))
archive=Path('/tmp/'+tag+'.tar.gz')
with tarfile.open(archive,'w:gz') as t:
 for name in sorted(sources):t.add(name,arcname=name)
 t.add(mp,arcname='fused-tables-source-sha256.json')
key=Path('/tmp/traj-key.pem')
if not key.exists():
 fd=os.open(str(key),os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
 with os.fdopen(fd,'wb') as out:out.write(Path('traj-key.pem').read_bytes())
subprocess.run(['scp','-q','-i',str(key),'-o','BatchMode=yes',str(archive),'ubuntu@192.222.55.229:'+str(archive)],check=True,timeout=55)
launch="from pathlib import Path\nimport subprocess,json\nroot=Path('/home/ubuntu/spacepdhcg-"+tag+"');root.mkdir(exist_ok=False)\n(root/'run.py').write_text("+repr(r)+")\nwith (root/'runner.log').open('x') as log:child=subprocess.Popen(['python3',str(root/'run.py')],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)\nprint(json.dumps(dict(pid=child.pid,root=str(root))))\n"
(p/'launch_conditioning_v520.py').write_text(launch)
