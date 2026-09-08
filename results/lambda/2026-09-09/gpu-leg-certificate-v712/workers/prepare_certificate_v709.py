from pathlib import Path
import base64
import hashlib
import json
import shutil
import subprocess

owned=['src/spacepdhcg/gtoc12/'+name+'.py' for name in ('cli','gpu_verifier','low_thrust','pipeline')]
owned+=['tests/'+name+'.py' for name in ('test_gtoc12_certificate_backend','test_gtoc12_gpu_scvx','test_gtoc12_gpu_verifier')]
files={name:base64.b64encode(Path(name).read_bytes()).decode() for name in owned}
worker=base64.b64encode(Path('build/performance/worker_certificate_v709.py').read_bytes()).decode()
script='''from pathlib import Path
import base64,hashlib,json,shutil,subprocess
home=Path('/home/ubuntu' if Path('/home/ubuntu').exists() else '/home/angus')
root=home/'spacepdhcg-certificate-v709';root.mkdir(exist_ok=False)
repo=root/'repo'
shutil.copytree(home/'spacepdhcg-search-campaign-v703/repo',repo,ignore=shutil.ignore_patterns('__pycache__','.pytest_cache'))
shutil.copytree(home/'spacepdhcg-joint-search-v702/repo/tests',repo/'tests',dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__'))
'''
script+='files='+repr(files)+'\n'
script+='''for name,encoded in files.items():
    target=repo/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(base64.b64decode(encoded))
runner=repo/'build/performance/orphan-recovery-v595/run.py'
text=runner.read_text()
old='qoco_ruiz_iterations=0, outer_loop_backend="cuda", seed_backend="cuda")'
assert text.count(old)==1
text=text.replace(old,old[:-1]+', certification_backend=os.environ["SPACEPDHCG_BENCH_CERTIFICATE_BACKEND"])')
runner.write_text(text)
manifest={p.relative_to(repo).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for directory in ('src','tests','build') for p in (repo/directory).rglob('*') if p.is_file()}
(root/'source-manifest.json').write_text(json.dumps(manifest,indent=2))
'''
script+='(root/"worker.py").write_bytes(base64.b64decode('+repr(worker)+'))\n'
script+='''with (root/'worker.log').open('x') as log:
    child=subprocess.Popen(['python3',str(root/'worker.py')],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps(dict(root=str(root),pid=child.pid,files=len(manifest))))
'''
Path('build/performance/launch_certificate_v709.py').write_text(script)
exec(compile(script,'launch-v709','exec'))
remote=subprocess.run(['ssh','-i','/tmp/traj-key.pem','-o','BatchMode=yes','-o','ConnectTimeout=10','ubuntu@192.222.55.229','python3 -'],input=script,text=True,capture_output=True,timeout=55)
print(remote.stdout);print(remote.stderr);remote.check_returncode()
