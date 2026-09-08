from pathlib import Path
p=Path('build/performance');local=Path('/home/angus/spacepdhcg-joint-selection-v630')
worker=(p/'check_joint_compatibility_v640.py').read_text()
for old,new in {'/home/angus/spacepdhcg-joint-selection-v630':'/home/ubuntu/spacepdhcg-joint-selection-v632','/home/angus/worktrees/spacepdhcg-literature-venv/bin/python':'/home/ubuntu/spacepdhcg/v1/.venv/bin/python','/home/angus/worktrees/spacepdhcg-release/benchmarks/gtoc12/data':'/home/ubuntu/spacepdhcg/gtoc12/benchmarks/gtoc12/data','/home/angus/.spacepdhcg-gpu.lock':'/home/ubuntu/.spacepdhcg-gpu.lock'}.items():worker=worker.replace(old,new)
code="from pathlib import Path\nimport json,shutil,subprocess\nroot=Path('/home/ubuntu/spacepdhcg-joint-selection-v632');source=root/'final-python-v640'\nshutil.copytree(root/'repo',source,ignore=shutil.ignore_patterns('.git','__pycache__','.pytest_cache','build'))\n"
for name in ('src/spacepdhcg/gtoc12/gpu_joint.py','tests/test_gtoc12_gpu_joint_compatibility.py'):
    code+="(source/"+repr(name)+").write_bytes("+repr((local/'final-python-v640'/name).read_bytes())+")\n"
code+="(root/'compatibility-source.json').write_bytes("+repr((local/'compatibility-source.json').read_bytes())+")\n"
code+="(root/'compatibility.patch').write_bytes("+repr((local/'compatibility.patch').read_bytes())+")\n"
code+="(root/'check-compatibility.py').write_text("+repr(worker)+")\n"
waiter="from pathlib import Path\nimport hashlib,json,subprocess,time\nroot=Path('/home/ubuntu/spacepdhcg-joint-selection-v632')\nmanifest=json.loads((root/'compatibility-source.json').read_text())\nfor name,digest in manifest.items():assert hashlib.sha256((root/'final-python-v640'/name).read_bytes()).hexdigest()==digest,name\nwhile not json.loads((root/'campaign-v636/report.json').read_text())['complete']:time.sleep(10)\nraise SystemExit(subprocess.call(['/home/ubuntu/spacepdhcg/v1/.venv/bin/python',str(root/'check-compatibility.py')]))\n"
code+="(root/'wait-check-compatibility.py').write_text("+repr(waiter)+")\n"
code+="with (root/'compatibility-runner.log').open('x') as log:child=subprocess.Popen(['python3',str(root/'wait-check-compatibility.py')],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)\nprint(json.dumps(dict(pid=child.pid)))\n"
(p/'launch_joint_compatibility_v641.py').write_text(code)
