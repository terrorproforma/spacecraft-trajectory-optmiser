from pathlib import Path
import hashlib
import json
import os
import shutil
import subprocess

out=Path('build/performance/retrieved-device-search-v707');out.mkdir(exist_ok=False)
key=Path('/tmp/traj-key.pem')
if not key.exists():
    fd=os.open(str(key),os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    with os.fdopen(fd,'wb') as file:file.write(Path('traj-key.pem').read_bytes())
for suffix in ('.tar.gz','.tar.manifest.json','.tar.record.json'):
    name='device-search-v707-h100'+suffix
    subprocess.run(['scp','-q','-i',str(key),'-o','BatchMode=yes','ubuntu@192.222.55.229:/home/ubuntu/'+name,str(out/name)],check=True,timeout=55)
for machine,prefix in [('local',Path('/home/angus')),('h100',out)]:
    p=prefix/('device-search-v707-'+machine+'.tar.gz');record=json.loads(p.with_suffix('.record.json').read_text())
    assert p.stat().st_size==record['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==record['sha256']
    print(machine,json.dumps(record),flush=True)
