from pathlib import Path
import hashlib,json,subprocess

repo=Path(__file__).resolve().parents[2]
out=repo/'results/lambda/2026-09-09/gpu-family-departures-v850'
out.mkdir(exist_ok=True)
ssh=['ssh.exe','-i','C:/Users/Angus/.ssh/spacepdhcg-expansion-key.pem','-o','BatchMode=yes','-o','ConnectTimeout=12','ubuntu@192.222.55.229']
code='''from pathlib import Path
import hashlib,json,subprocess,tarfile
home=Path.home();root=home/'spacepdhcg-family-departures-v850';prior=home/'spacepdhcg-family-departures-v849'
report=json.loads((root/'report.json').read_text())
assert report['complete'] and report['success'],report
assert not Path('/proc/'+str(report['pid'])).exists()
assert not Path('/proc/'+str(json.loads((prior/'report.json').read_text())['pid'])).exists()
gpu=subprocess.run(['nvidia-smi','--query-gpu=uuid,name,utilization.gpu,memory.used','--format=csv,noheader'],capture_output=True,text=True,check=True)
processes=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],capture_output=True,text=True,check=True)
(root/'terminal.json').write_text(json.dumps(dict(worker_exited=True,initial_worker_exited=True,gpu=gpu.stdout,compute_processes=processes.stdout),indent=2))
# Include the pinned bonus table for independent Decimal score accounting.
bonus=home/'spacepdhcg/gtoc12/benchmarks/gtoc12/data/bonus_coefficients.txt'
assert hashlib.sha256(bonus.read_bytes()).hexdigest()==report['bonus_sha256']
(root/'bonus_coefficients.txt').write_bytes(bonus.read_bytes())
archive=home/'spacepdhcg-family-departures-v850.tar.gz'
assert not archive.exists()
files={}
with tarfile.open(archive,'w:gz') as tar:
    for base,prefix in [(root,'continued'),(prior,'initial')]:
        for p in sorted(base.rglob('*')):
            if p.is_file():
                assert not p.is_symlink()
                name=prefix+'/'+str(p.relative_to(base));raw=p.read_bytes()
                files[name]=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
                tar.add(p,arcname=name,recursive=False)
manifest=dict(archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),archive_bytes=archive.stat().st_size,files=files)
(home/'spacepdhcg-family-departures-v850-manifest.json').write_text(json.dumps(manifest,indent=2))
print(json.dumps(dict(complete=True,candidates=sum(r.get('candidates',0) for r in report['routes']),refinement_candidates=sum(len(r.get('refinement_candidates',[])) for r in report['routes']),seconds=report['seconds'],manifest={k:v for k,v in manifest.items() if k!='files'})))
'''
r=subprocess.run(ssh+['python3 -'],input=code,text=True,capture_output=True,timeout=50,check=True)
print(r.stdout,r.stderr)
for name,target in [('spacepdhcg-family-departures-v850.tar.gz','evidence.tar.gz'),('spacepdhcg-family-departures-v850-manifest.json','manifest.json')]:
    assert not (out/target).exists()
    subprocess.run(['scp.exe','-i','C:/Users/Angus/.ssh/spacepdhcg-expansion-key.pem','-o','BatchMode=yes',
        'ubuntu@192.222.55.229:/home/ubuntu/'+name,str(out/target)],check=True,timeout=100)
manifest=json.loads((out/'manifest.json').read_text())
assert hashlib.sha256((out/'evidence.tar.gz').read_bytes()).hexdigest()==manifest['archive_sha256']
print('Retrieved and verified',manifest['archive_bytes'],'bytes')
