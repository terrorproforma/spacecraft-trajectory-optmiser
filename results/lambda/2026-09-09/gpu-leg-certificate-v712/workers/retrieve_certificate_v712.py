from pathlib import Path
import os,subprocess
out=Path('build/performance/retrieved-certificate-v712');out.mkdir(exist_ok=False)
key=Path('/tmp/traj-key.pem')
if not key.exists():
    fd=os.open(key,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
    with os.fdopen(fd,'wb') as file:file.write(Path('traj-key.pem').read_bytes())
for suffix in ('.tar.gz','.tar.manifest.json','.tar.record.json'):
    subprocess.run(['scp','-q','-i',str(key),'-o','BatchMode=yes','ubuntu@192.222.55.229:/home/ubuntu/certificate-v712-h100'+suffix,str(out)],check=True,timeout=55)
print(out)
