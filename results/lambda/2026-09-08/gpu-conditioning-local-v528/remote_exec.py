import os,subprocess,sys
from pathlib import Path
key=Path('/tmp/traj-key.pem')
if not key.exists():
 fd=os.open(str(key),os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
 with os.fdopen(fd,'wb') as out: out.write(Path('traj-key.pem').read_bytes())
r=subprocess.run(['ssh','-i',str(key),'-o','BatchMode=yes','-o','ConnectTimeout=10','ubuntu@192.222.55.229','python3 -'],input=Path(sys.argv[1]).read_text(),text=True,capture_output=True,timeout=55)
print(r.stdout);print(r.stderr,file=sys.stderr);sys.exit(r.returncode)
