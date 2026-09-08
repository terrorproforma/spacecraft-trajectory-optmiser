from pathlib import Path
import hashlib,json,os,shutil,subprocess,tarfile
workspace=Path.cwd(); build=Path('/home/angus/spacepdhcg-joint-selection-v630'); source=build/'repo'
fixture=Path('results/gtoc12/runs/return_sweep_v2/ships/cluster_fleet_v7_clusters_family_0007/ship_01/route_summary.json')
(source/fixture).parent.mkdir(parents=True,exist_ok=True)
assert not (source/fixture).exists()
shutil.copy2(workspace/fixture,source/fixture)
(build/'fixture-addition.json').write_text(json.dumps(dict(path=str(fixture),sha256=hashlib.sha256((source/fixture).read_bytes()).hexdigest()),indent=2))
# Freeze auxiliary validation inputs without changing the already compiled source.
for name in ('check_joint_selection_v631.py','joint-benchmark-v593/benchmark_joint.py'):
    path=source/'build/performance'/name;path.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(workspace/'build/performance'/name,path)
worker=(source/'build/performance/check_joint_selection_v631.py').read_text()
mapping={'/home/angus/spacepdhcg-joint-selection-v630':'/home/ubuntu/spacepdhcg-joint-selection-v632','/home/angus/worktrees/spacepdhcg-literature-venv/bin/python':'/home/ubuntu/spacepdhcg/v1/.venv/bin/python','/home/angus/worktrees/spacepdhcg-release/benchmarks/gtoc12/data':'/home/ubuntu/spacepdhcg/gtoc12/benchmarks/gtoc12/data','/home/angus/build-qoco-scaled-pool-v540/final':'/home/ubuntu/spacepdhcg-scaled-pool-v545/final','/home/angus/cudss-isolated-0.8.0.10/nvidia/cu12/lib':'/home/ubuntu/spacepdhcg-recovery-v152/cudss/lib','/usr/local/cuda-12.8':'/usr/local/cuda','/home/angus/.spacepdhcg-gpu.lock':'/home/ubuntu/.spacepdhcg-gpu.lock','sm_120':'sm_90'}
for old,new in mapping.items():worker=worker.replace(old,new)
(workspace/'build/performance/check_joint_selection_v632.py').write_text(worker)
archive=Path('/tmp/joint-selection-v632.tar.gz')
with tarfile.open(archive,'w:gz') as tar:
    for path in sorted(source.rglob('*')):
        if path.is_file() and not any(p in ('.git','__pycache__','.pytest_cache') for p in path.relative_to(source).parts):
            tar.add(path,arcname=str(path.relative_to(source)))
    tar.add(workspace/'build/performance/check_joint_selection_v632.py',arcname='build/performance/check_joint_selection_v632.py')
manifest={str(path.relative_to(source)):hashlib.sha256(path.read_bytes()).hexdigest() for path in source.rglob('*') if path.is_file() and not any(p in ('.git','__pycache__','.pytest_cache') for p in path.relative_to(source).parts)}
manifest['build/performance/check_joint_selection_v632.py']=hashlib.sha256(worker.encode()).hexdigest()
(workspace/'build/performance/joint-source-v632.json').write_text(json.dumps(manifest,indent=2))
key=Path('/tmp/traj-key.pem')
if not key.exists():
    fd=os.open(str(key),os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    with os.fdopen(fd,'wb') as out:out.write((workspace/'traj-key.pem').read_bytes())
subprocess.run(['scp','-q','-i',str(key),'-o','BatchMode=yes',str(archive),'ubuntu@192.222.55.229:'+str(archive)],check=True,timeout=55)
print(json.dumps(dict(archive=str(archive),bytes=archive.stat().st_size,sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),files=len(manifest))))
