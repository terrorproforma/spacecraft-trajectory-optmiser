from pathlib import Path
import ast,hashlib,json,os,subprocess,tarfile
p=Path('build/performance')
worker=(p/'check_bound_types_v575.py').read_text().replace('bound-types-v575','bound-types-v576')
a=worker.index('frozen=');b=worker.index('\nsources=',a)
worker=worker[:a]+"core=Path('/home/ubuntu/spacepdhcg-bound-types-v576/core-build/cuda/libspacepdhcg_cuda.so')"+worker[b:]
(p/'check_bound_types_v576.py').write_text(worker)
tree=ast.parse((p/'prepare_device_initialization_v557.py').read_text())
s=next(n.value.value for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='script' for t in n.targets))
s=s.replace('device-init-v557','bound-types-v576').replace('validate_device_initialization_v556.py','check_bound_types_v576.py').replace('device-init-v556/report.json','bound-types-v576/report.json')
s=s.replace("shutil.copytree('/home/ubuntu/spacepdhcg-scaled-pool-v545/repo'", "shutil.copytree('/home/ubuntu/spacepdhcg-bound-types-v569/repo'")
s=s.replace("r['complete'] and not r.get('error'),r", "r['complete'] and not r.get('error'),r")
names=['cpp/cuda/src/native_qoco_adapter.cpp','tests/test_gtoc12_gpu_bound_types.py','build/performance/check_bound_types_v576.py']
archive=Path('/tmp/bound-types-v576.tar.gz')
with tarfile.open(archive,'w:gz') as t:
 for name in names:t.add(name,arcname=name)
manifest={name:hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in names}
key=Path('/tmp/traj-key.pem')
if not key.exists():
 fd=os.open(str(key),os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
 with os.fdopen(fd,'wb') as out:out.write(Path('traj-key.pem').read_bytes())
subprocess.run(['scp','-q','-i',str(key),'-o','BatchMode=yes',str(archive),'ubuntu@192.222.55.229:'+str(archive)],check=True,timeout=55)
code="from pathlib import Path\nimport subprocess,json\nroot=Path('/home/ubuntu/spacepdhcg-bound-types-v576');root.mkdir(exist_ok=False)\n"
code+="(root/'run.py').write_text("+repr(s)+")\n(root/'source-sha256.json').write_text("+repr(json.dumps(manifest,indent=2))+")\n"
code+="with (root/'runner.log').open('x') as log:child=subprocess.Popen(['python3',str(root/'run.py')],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)\nprint(json.dumps(dict(pid=child.pid)))\n"
(p/'launch_bound_types_v576.py').write_text(code)
