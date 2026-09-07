from pathlib import Path
import os,subprocess,tarfile,hashlib,json,shutil
key=Path('/tmp/traj-key.pem')
if not key.exists():
 fd=os.open(str(key),os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
 with os.fdopen(fd,'wb') as f:f.write(Path('traj-key.pem').read_bytes())
root=Path('results/lambda/2026-09-08/gpu-execution-v328');root.mkdir(exist_ok=False)
archive=Path('build/performance/gpu-execution-v328/results.tar.gz')
subprocess.run(['scp','-i',str(key),'-o','BatchMode=yes','ubuntu@192.222.55.229:/tmp/gpu-execution-v328-results.tar.gz',str(archive)],check=True)
h=hashlib.sha256(archive.read_bytes()).hexdigest();assert h=='af87bb1504779a19ca71f294f5840e161a754e2e6fbf26819fe7310d001e6ade'
with tarfile.open(archive) as t:
 for m in t.getmembers():
  p=(root/m.name).resolve();assert p.is_relative_to(root.resolve()) and m.isfile()
  p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(t.extractfile(m).read())
manifest=json.loads((root/'sha256.json').read_text());assert all(hashlib.sha256((root/p).read_bytes()).hexdigest()==h for p,h in manifest.items())
(root/'retrieval.json').write_text(json.dumps(dict(archive_sha256=h,files_verified=len(manifest)),indent=2))
shutil.copytree('build/performance/gpu-execution-v328',root/'local',ignore=shutil.ignore_patterns('*.tar.gz'))
(root/'recipes').mkdir()
for name in ['launch_gpu_execution_v328.py','test_gpu_execution_v328.py','package_gpu_execution_v328.py','retrieve_gpu_execution_v328.py']:
 shutil.copyfile(Path('build/performance')/name,root/'recipes'/name)
print('verified',len(manifest),'files')
