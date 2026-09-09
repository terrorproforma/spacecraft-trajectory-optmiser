from pathlib import Path
import hashlib,io,json,shutil,subprocess,tarfile
home=Path.home();root=home/'spacepdhcg-native-collect-plan-v856';root.mkdir();repo=root/'repo';repo.mkdir()
base='f2511e510938d78d61d2c492a78d2b0ba79793a6'
owned=['cpp/cuda/include/spacepdhcg/cuda/gtoc12_collect_dp_c_api.h','cpp/cuda/src/gtoc12_collect_dp.cu',
       'src/spacepdhcg/gtoc12/gpu_collect_dp.py','src/spacepdhcg/gtoc12/collectdp.py',
       'tests/test_gtoc12_gpu_collect_plan.py','tests/test_gtoc12_gpu_resident_collect_tables.py']
with tarfile.open(fileobj=io.BytesIO(subprocess.check_output(['git','archive',base,'.gitignore','cpp','src','tests','scripts','benchmarks','pyproject.toml','third_party']))) as tar:tar.extractall(repo,filter='data')
for name in owned:shutil.copyfile(name,repo/name)
fixture='results/lambda/2026-09-08/gpu-collect-profile-v272/v272/timing.json'
(repo/fixture).parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(fixture,repo/fixture)
(root/'source-manifest.json').write_text(json.dumps(dict(base=base,owned=owned,files={p.relative_to(repo).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in repo.rglob('*') if p.is_file()}),indent=2))
worker=(home/'spacepdhcg-return-cache-v844/worker.py').read_text().replace('Freeze shared native return option runtime','Freeze native mining and collection plan runtime')
worker=worker.replace("'tests/test_gtoc12_gpu_resident_options.py']", "'tests/test_gtoc12_gpu_resident_options.py','tests/test_gtoc12_gpu_collect_plan.py']")
worker=worker.replace("    report['success']=True", "    with (home/'.spacepdhcg-gpu.lock').open('a') as lock:\n        fcntl.flock(lock,fcntl.LOCK_EX)\n        run('leaks',[cuda+'/bin/compute-sanitizer','--tool','memcheck','--leak-check','full','--error-exitcode','86',py,'-c',boot,'-q','tests/test_gtoc12_gpu_collect_plan.py'])\n    report['success']=True")
(root/'worker.py').write_text(worker)
archive=root/'source.tar.gz'
with tarfile.open(archive,'w:gz') as tar:
    tar.add(repo,arcname='repo')
    for name in ('worker.py','source-manifest.json'):tar.add(root/name,arcname=name)
key='/home/angus/.ssh/spacepdhcg-expansion-key.pem'
subprocess.run(['scp','-q','-i',key,str(archive),'ubuntu@192.222.55.229:/tmp/native-collect-plan-source-v856.tar.gz'],check=True,timeout=55)
remote="""from pathlib import Path
import hashlib,json,subprocess,tarfile
root=Path.home()/'spacepdhcg-native-collect-plan-v856';root.mkdir()
with tarfile.open('/tmp/native-collect-plan-source-v856.tar.gz') as tar:tar.extractall(root,filter='data')
for p,h in json.loads((root/'source-manifest.json').read_text())['files'].items():assert hashlib.sha256((root/'repo'/p).read_bytes()).hexdigest()==h,p
with (root/'worker.log').open('x') as log:print(subprocess.Popen(['python3',str(root/'worker.py')],stdout=log,stderr=subprocess.STDOUT,start_new_session=True).pid)
"""
Path('build/performance/launch_collect_plan_h100_v856.py').write_text(remote)
r=subprocess.run(['ssh','-i',key,'-o','BatchMode=yes','ubuntu@192.222.55.229','python3 -'],input=remote,text=True,capture_output=True,check=True,timeout=40);print('h100',r.stdout,r.stderr)
with (root/'worker.log').open('x') as log:print('local',subprocess.Popen(['python3',str(root/'worker.py')],stdout=log,stderr=subprocess.STDOUT,start_new_session=True).pid)
