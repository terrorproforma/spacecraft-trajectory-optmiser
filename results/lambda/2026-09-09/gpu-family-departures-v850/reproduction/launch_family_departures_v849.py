from pathlib import Path
import base64, hashlib, json, subprocess

repo=Path(__file__).resolve().parents[2]
files={
    'run.py': (repo/'build/performance/run_family_departures_v849.py').read_bytes(),
    'fleet.txt': (repo/'results/local/2026-09-09/fleet-addition-v633/Result.txt').read_bytes(),
}
compile(files['run.py'], 'run.py', 'exec')
payload={name:base64.b64encode(raw).decode() for name,raw in files.items()}
remote='''from pathlib import Path
import base64,hashlib,json,os,subprocess
home=Path.home(); root=home/'spacepdhcg-family-departures-v849'
assert not root.exists()
runtime=home/'spacepdhcg-return-cache-v844'
report=json.loads((runtime/'report.json').read_text());assert report['complete'] and report['success']
root.mkdir()
payload=PAYLOAD
for name,raw in payload.items():(root/name).write_bytes(base64.b64decode(raw))
(root/'fit.json').write_bytes((home/'spacepdhcg-expansion-wide-v837/fit.json').read_bytes())
core=runtime/'final/libspacepdhcg_cuda.so'
manifest={str(p.relative_to(root)):dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in root.iterdir()}
manifest['runtime']={'core_sha256':hashlib.sha256(core.read_bytes()).hexdigest(),'source_manifest_sha256':hashlib.sha256((runtime/'source-manifest.json').read_bytes()).hexdigest()}
(root/'input-manifest.json').write_text(json.dumps(manifest,indent=2))
env={k:v for k,v in os.environ.items() if not k.startswith(('SPACEPDHCG_TEST_','QOCO_REPLAY_'))}
env.update(SPACEPDHCG_GTOC12_CUDA_LIBRARY=str(core),SPACEPDHCG_GTOC12_DATA=str(home/'spacepdhcg/gtoc12/benchmarks/gtoc12/data'),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',LD_LIBRARY_PATH='/usr/local/cuda/lib64')
with (root/'worker.log').open('x') as log:
    child=subprocess.Popen([str(home/'spacepdhcg/v1/.venv/bin/python'),str(root/'run.py')],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
(root/'launch.json').write_text(json.dumps({'pid':child.pid,'manifest':manifest},indent=2))
print(json.dumps({'root':str(root),'pid':child.pid,'runtime':manifest['runtime']}))
'''.replace('PAYLOAD',repr(payload))
ssh=['ssh.exe','-i','C:/Users/Angus/.ssh/spacepdhcg-expansion-key.pem','-o','BatchMode=yes','-o','ConnectTimeout=12','ubuntu@192.222.55.229','python3 -']
r=subprocess.run(ssh,input=remote,text=True,capture_output=True,timeout=40,check=True)
print(r.stdout,r.stderr)
(repo/'build/performance/family-departures-launch-v849.json').write_text(r.stdout)
