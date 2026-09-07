from pathlib import Path
import subprocess,tarfile,json,hashlib,shutil,os
out=Path('results/lambda/2026-09-08/gpu-conic-retry-v314');out.mkdir(exist_ok=False)
key=Path('/tmp/traj-key.pem')
if not key.exists():
 fd=os.open(str(key),os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
 with os.fdopen(fd,'wb') as f:f.write(Path('traj-key.pem').read_bytes())
archive=out/'diagnostic-evidence.tar.gz'
subprocess.run(['scp','-q','-i',str(key),'-o','BatchMode=yes','ubuntu@192.222.55.229:/home/ubuntu/spacepdhcg-conic-retry-v314/diagnostic-evidence.tar.gz',str(archive)],check=True,timeout=55)
assert hashlib.sha256(archive.read_bytes()).hexdigest()=='db7b35d85207ae19894d515c38cc11a882e26b0e47863436499dacc8428a9239'
with tarfile.open(archive) as t:
 for m in t.getmembers():
  if m.isdir():continue
  target=(out/m.name).resolve();assert target.is_relative_to(out.resolve()) and m.isfile()
  target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(t.extractfile(m).read())
hashes=json.loads((out/'diagnostics/sha256.json').read_text());assert all(hashlib.sha256((out/'diagnostics'/p).read_bytes()).hexdigest()==h for p,h in hashes.items())
local=out/'rtx5090';local.mkdir()
shutil.copytree('build/performance/conic-retry-v313',local/'checks')
shutil.copytree('build/performance/arc-retry-v313',local/'arc')
for name in ['manifest.json','build.log']:shutil.copy2(Path('/home/angus/build-spacepdhcg-conic-retry-v313')/name,local/name)
prototype=local/'rejected-precise-ir-v310';prototype.mkdir()
shutil.copy2('build/performance/precise-ir-v310/report.json',prototype/'report.json')
shutil.copy2('build/performance/qoco_precise_ir_v310.cuh',prototype/'qoco_precise_ir_v310.cuh')
for name in ['build_precise_ir_v310.py','resume_precise_ir_v310.py','replay_precise_ir_v310.py']:shutil.copy2(Path('build/performance')/name,prototype/name)
for name in ['build-0.log','resume-0.log','resume-1.log','report.json']:shutil.copy2(Path('/home/angus/build-qoco-precise-ir-v310')/name,prototype/('build-'+name))
print('Verified',len(hashes),'downloaded artifacts; copied local validation and rejected prototype evidence')
