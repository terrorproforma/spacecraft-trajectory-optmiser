from pathlib import Path
import os,subprocess
key=Path('/tmp/traj-key.pem')
if not key.exists():
    fd=os.open(key,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
    with os.fdopen(fd,'wb') as out:out.write(Path('traj-key.pem').read_bytes())
root=Path.home()/'spacepdhcg-collect-composition-v807';target='ubuntu@192.222.55.229'
assert not (root/'worker.log').exists()
subprocess.run(['ssh','-i',str(key),'-o','BatchMode=yes',target,'mkdir /home/ubuntu/spacepdhcg-collect-composition-v807'],check=True,timeout=20)
subprocess.run(['scp','-i',str(key),*[str(root/n) for n in ('run.py','launch.py','Result.txt','plan.json')],target+':/home/ubuntu/spacepdhcg-collect-composition-v807/'],check=True,timeout=40)
subprocess.run(['ssh','-i',str(key),'-o','BatchMode=yes',target,'python3 /home/ubuntu/spacepdhcg-collect-composition-v807/launch.py'],check=True,timeout=20)
exec((root/'launch.py').read_text())
