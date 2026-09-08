from pathlib import Path
import ast,hashlib,json,os,subprocess,tarfile
p=Path('build/performance')
worker=(p/'validate_device_initialization_v558.py').read_text().replace('device-init-v558','device-init-v565').replace('validate_device_initialization_v558.py','validate_device_initialization_v565.py')
worker=worker.replace("env['SPACEPDHCG_TEST_QOCO_DEVICE_INITIALIZATION']='1'", "env.pop('SPACEPDHCG_TEST_QOCO_DEVICE_INITIALIZATION',None)")
worker=worker.replace("files=['tests/test_gtoc12_gpu_qoco.py'", "files=['tests/test_gtoc12_gpu_scvx.py','tests/test_gtoc12_gpu_qoco.py'")
a=worker.index(" tests=['tests/'+name");b=worker.index('\n for name in tests:',a)
worker=worker[:a]+" tests=['tests/'+name for name in ['test_gtoc12_gpu_device_initialization.py','test_gtoc12_gpu_qoco.py','test_gtoc12_gpu_scvx.py','test_gtoc12_gpu_retained_replay.py','test_gtoc12_gpu_workspace_pool.py']]"+worker[b:]
a=worker.index('  replay=');b=worker.index(" report['complete']=True",a)
worker=worker[:a]+"  selection='tests/test_gtoc12_gpu_device_initialization.py::test_device_initialization_matches_fresh_reference[0-0-lagrange-1]'\n  run('memcheck',['/usr/local/cuda-12.8/bin/compute-sanitizer','--tool','memcheck','--error-exitcode','99',py,'-c',boot,selection,'-x','-s','-q'])\n"+worker[b:]
(p/'validate_device_initialization_v565.py').write_text(worker)
tree=ast.parse((p/'prepare_device_initialization_v557.py').read_text())
s=next(n.value.value for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='script' for t in n.targets))
s=s.replace('device-init-v557','device-init-v565').replace('validate_device_initialization_v556.py','validate_device_initialization_v565.py').replace('device-init-v556/report.json','device-init-v565/report.json')
names=['cpp/cuda/src/native_qoco_adapter.cpp','tests/test_gtoc12_gpu_device_initialization.py','tests/test_gtoc12_gpu_qoco.py','tests/test_gtoc12_gpu_scvx.py','build/performance/validate_device_initialization_v565.py','build/performance/replay_device_initialization.py']
archive=Path('/tmp/device-init-v565.tar.gz')
with tarfile.open(archive,'w:gz') as t:
 for name in names:t.add(name,arcname=name)
manifest={name:hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in names}
key=Path('/tmp/traj-key.pem')
if not key.exists():
 fd=os.open(str(key),os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
 with os.fdopen(fd,'wb') as out:out.write(Path('traj-key.pem').read_bytes())
subprocess.run(['scp','-q','-i',str(key),'-o','BatchMode=yes',str(archive),'ubuntu@192.222.55.229:'+str(archive)],check=True,timeout=55)
code="from pathlib import Path\nimport subprocess,json\nroot=Path('/home/ubuntu/spacepdhcg-device-init-v565');root.mkdir(exist_ok=False)\n"
code+="(root/'run.py').write_text("+repr(s)+")\n(root/'source-sha256.json').write_text("+repr(json.dumps(manifest,indent=2))+")\n"
code+="with (root/'runner.log').open('x') as log:child=subprocess.Popen(['python3',str(root/'run.py')],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)\nprint(json.dumps(dict(pid=child.pid)))\n"
(p/'launch_device_initialization_v565.py').write_text(code)
