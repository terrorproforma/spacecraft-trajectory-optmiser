from pathlib import Path
import hashlib,io,json,os,shutil,subprocess,tarfile
root=Path.home()/'spacepdhcg-catalogue-v808';root.mkdir();repo=root/'repo';repo.mkdir()
base=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
with tarfile.open(fileobj=io.BytesIO(subprocess.check_output(['git','archive',base,'cpp','src','tests','scripts','benchmarks','pyproject.toml','third_party']))) as tar:tar.extractall(repo,filter='data')
owned=['src/spacepdhcg/gtoc12/data.py','src/spacepdhcg/gtoc12/gpu_completion_model.py','tests/test_gtoc12_gpu_completion_model.py','tests/test_gtoc12_catalogue_immutability.py']
for p in owned:shutil.copyfile(p,repo/p)
fixture='results/lambda/2026-09-08/gpu-collect-profile-v272/v272/timing.json';(repo/fixture).parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(fixture,repo/fixture)
(root/'source-manifest.json').write_text(json.dumps(dict(base=base,files={p.relative_to(repo).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in repo.rglob('*') if p.is_file()}),indent=2))
worker=Path('build/performance/worker_collect_v800.py').read_text()
worker=worker.replace("'-m','Freeze CUDA insertion batches'","'-m','Freeze integrated endpoint and catalogue fingerprint runtime'")
worker=worker.replace("'spacepdhcg_cuda','-j3'","'spacepdhcg_cuda','gtoc12_scvx_test','-j3'")
worker=worker.replace("env['SPACEPDHCG_GTOC12_JOINT_ARCHIVE']", "env['SPACEPDHCG_COMPLETION_TEST_LIBRARY']=str(root/'final/libspacepdhcg_cuda.so')\nenv['SPACEPDHCG_GTOC12_JOINT_ARCHIVE']")
worker=worker.replace("        run('pytest',[py,'-c',boot,'-q',*tests])", "        run('controller',[str(root/'build/cuda/gtoc12_scvx_test')])\n        tests += ['tests/test_gtoc12_gpu_completion_model.py','tests/test_gtoc12_gpu_completion.py','tests/test_gtoc12_catalogue_immutability.py','tests/test_gtoc12_rules_and_data.py','tests/test_gtoc12_completion_costs.py','tests/test_gtoc12_completion_capture.py','tests/test_gtoc12_zoh_trajectory_seed.py']\n        run('pytest',[py,'-c',boot,'-q',*tests])")
worker=worker.replace("'tests/test_gtoc12_gpu_collect_workspace.py','tests/test_gtoc12_gpu_resident_collect_tables.py'])", "'tests/test_gtoc12_gpu_collect_workspace.py','tests/test_gtoc12_gpu_resident_collect_tables.py','tests/test_gtoc12_gpu_completion_model.py'])\n            run(mode+'-controller',[cuda+'/bin/compute-sanitizer','--tool',mode,'--error-exitcode','86',str(root/'build/cuda/gtoc12_scvx_test')])")
(root/'worker.py').write_text(worker)
with (root/'worker.log').open('x') as log:pid=subprocess.Popen(['python3',str(root/'worker.py')],stdout=log,stderr=subprocess.STDOUT,start_new_session=True).pid
archive=Path('/tmp/catalogue-source-v808.tar.gz')
with tarfile.open(archive,'w:gz') as tar:
    tar.add(repo,arcname='repo',filter=lambda m:None if '/.git/' in m.name or m.name.endswith('/.git') else m)
    for p in ('worker.py','source-manifest.json'):tar.add(root/p,arcname=p)
key=Path('/tmp/traj-key.pem')
if not key.exists():
    fd=os.open(key,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
    with os.fdopen(fd,'wb') as out:out.write(Path('traj-key.pem').read_bytes())
subprocess.run(['scp','-q','-i',str(key),str(archive),'ubuntu@192.222.55.229:/tmp/catalogue-source-v808.tar.gz'],check=True,timeout=55)
remote="from pathlib import Path\nimport hashlib,json,subprocess,tarfile\nroot=Path.home()/'spacepdhcg-catalogue-v808';root.mkdir()\nwith tarfile.open('/tmp/catalogue-source-v808.tar.gz') as tar:tar.extractall(root,filter='data')\nfor p,h in json.loads((root/'source-manifest.json').read_text())['files'].items():assert hashlib.sha256((root/'repo'/p).read_bytes()).hexdigest()==h,p\nwith (root/'worker.log').open('x') as log:print(subprocess.Popen(['python3',str(root/'worker.py')],stdout=log,stderr=subprocess.STDOUT,start_new_session=True).pid)"
Path('build/performance/launch_catalogue_h100_v808.py').write_text(remote)
r=subprocess.run(['ssh','-i',str(key),'-o','BatchMode=yes','ubuntu@192.222.55.229','python3 -'],input=remote,text=True,capture_output=True,check=True,timeout=55)
print(json.dumps(dict(base=base,local_pid=pid,h100=r.stdout,remote_stderr=r.stderr)))
