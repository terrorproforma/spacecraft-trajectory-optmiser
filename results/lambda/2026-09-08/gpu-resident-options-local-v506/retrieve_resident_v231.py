from pathlib import Path
import json,os,subprocess,tarfile,hashlib,shutil
files=[n for n in json.loads(Path('build/performance/resident-v231-source-files.json').read_text()) if not n.startswith('results/')]
remote='''from pathlib import Path
import json,hashlib,tarfile
root=Path('/home/ubuntu/spacepdhcg-resident-v231');repo=root/'repo'
assert json.loads((root/'report.json').read_text())['complete']
mission=Path('/home/ubuntu/spacepdhcg-resident-cert-v232');report=json.loads((mission/'report.json').read_text())
assert report['complete'] and report['official']['ok'] and report['independent']['ok']
(root/'source-sha256.json').write_text(json.dumps({n:hashlib.sha256((repo/n).read_bytes()).hexdigest() for n in FILES},indent=2))
with tarfile.open(root/'evidence.tar.gz','w:gz') as t:
 for name in ['report.json','run.py','source-sha256.json','configure.log','build.log','benchmark.log','tests.log','memcheck.log','initcheck.log','final-cache-tests.log','measurement']:t.add(root/name,arcname='validation/'+name)
 for name in ['report.json','run.py','runner.log','plan.json','output']:t.add(mission/name,arcname='mission-v232/'+name)
print(hashlib.sha256((root/'evidence.tar.gz').read_bytes()).hexdigest())
'''.replace('FILES',repr(files))
out=Path('results/lambda/2026-09-08/gpu-resident-retime-v231');out.mkdir(exist_ok=False)
key=Path('/tmp/traj-key.pem')
if not key.exists():
 fd=os.open(key,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
 with os.fdopen(fd,'wb') as f:f.write(Path('traj-key.pem').read_bytes())
r=subprocess.run(['ssh','-i',str(key),'-o','BatchMode=yes','ubuntu@192.222.55.229','python3 -'],input=remote,text=True,capture_output=True,timeout=55)
if r.returncode:print(r.stderr);r.check_returncode()
archive=out/'h100-evidence.tar.gz';subprocess.run(['scp','-i',str(key),'-o','BatchMode=yes','ubuntu@192.222.55.229:/home/ubuntu/spacepdhcg-resident-v231/evidence.tar.gz',str(archive)],check=True,timeout=55)
assert hashlib.sha256(archive.read_bytes()).hexdigest()==r.stdout.strip()
with tarfile.open(archive) as t:t.extractall(out/'h100',filter='data')
for n,d in json.loads((out/'h100/validation/source-sha256.json').read_text()).items():assert hashlib.sha256(Path(n).read_text().encode()).hexdigest()==d,n
local=out/'local';local.mkdir()
for source,name in [('build/performance/resident-v230/benchmark.json','benchmark.json'),('build/performance/resident-v230-tests-final.log','tests.log'),('build/performance/resident-v230-build.log','build.log')]:shutil.copy(source,local/name)
shutil.copy('build/performance/benchmark_resident_v230.py',out/'benchmark_resident_v230.py')
shutil.copy('build/performance/final_resident_v231.py',out/'final-cache-validation.py')
shutil.copy('results/lambda/2026-09-08/gpu-native-campaign-v209/v209/output/ship_01/refinements.json',out/'input-refinements.json')
frozen=Path('/home/angus/build-spacepdhcg-resident-v230/final');frozen.mkdir(parents=True,exist_ok=False)
shutil.copy('/home/angus/build-spacepdhcg-gtoc12-v141/cuda/libspacepdhcg_cuda.so',frozen/'libspacepdhcg_cuda.so')
assert hashlib.sha256((frozen/'libspacepdhcg_cuda.so').read_bytes()).hexdigest()==json.loads((local/'benchmark.json').read_text())['library_sha256']
timing={}
for gpu,path in [('RTX5090',local/'benchmark.json'),('H100',out/'h100/validation/measurement/benchmark.json')]:
 m=json.loads(path.read_text())['medians'];timing[gpu]=dict(host_tables_seconds=m['False'],resident_tables_seconds=m['True'],speedup=m['False']/m['True'])
summary=dict(scope='Resident unswept fixed-order retiming tables; same schedules and objective, unchanged full-mission physics checks.',timing=timing,tests_per_gpu=84,h100_final_cache_tests=21,h100_memcheck_errors=0,h100_initcheck_errors=0,incumbent_weighted_kg=12805.194102488575,mission=json.loads((out/'h100/mission-v232/report.json').read_text())['independent'])
(out/'summary.json').write_text(json.dumps(summary,indent=2))
hashes={str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest() for p in out.rglob('*') if p.is_file()}
(out/'evidence-sha256.json').write_text(json.dumps(hashes,indent=2));print(json.dumps(summary,indent=2));print('Verified',len(hashes),'artifacts and',len(files),'sources')
