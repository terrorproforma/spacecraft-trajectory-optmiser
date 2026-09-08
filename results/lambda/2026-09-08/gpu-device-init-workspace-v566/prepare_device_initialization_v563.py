from pathlib import Path
import hashlib,json,runpy
p=Path('build/performance');h=runpy.run_path(str(p/'prepare_scaled_followups.py'))
s=(p/'validate_device_initialization_v562.py').read_text().replace('device-init-v562','device-init-v563').replace('validate_device_initialization_v562.py','validate_device_initialization_v563.py')
s=s.replace('/home/angus/build-spacepdhcg-device-init-v555/final','/home/ubuntu/spacepdhcg-device-init-v557/core-build/cuda')
s=h['adapt'](s)
files={'build/performance/validate_device_initialization_v563.py':s,'tests/test_gtoc12_gpu_qoco.py':Path('tests/test_gtoc12_gpu_qoco.py').read_text(),'tests/test_gtoc12_gpu_scvx.py':Path('tests/test_gtoc12_gpu_scvx.py').read_text()}
code="from pathlib import Path\nimport hashlib,json,subprocess\nrepo=Path('/home/ubuntu/spacepdhcg-device-init-v557/repo')\n"
code+="files="+repr(files)+"\nfor name,text in files.items():(repo/name).write_text(text)\n"
code+="(repo/'build/performance/device-init-v563-overlay.json').write_text(json.dumps({n:hashlib.sha256((repo/n).read_bytes()).hexdigest() for n in files},indent=2))\n"
code+="with (repo/'build/performance/device-init-v563-runner.log').open('x') as log:child=subprocess.Popen(['python3','build/performance/validate_device_initialization_v563.py'],cwd=repo,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)\nprint(json.dumps(dict(pid=child.pid)))\n"
(p/'launch_device_initialization_v563.py').write_text(code)
