from pathlib import Path
import subprocess,fcntl
out=Path('/home/angus/residual-operator-v334b');out.mkdir(exist_ok=False)
files=sorted(Path('/home/angus/build-qoco-linear-snapshot-v333/snapshots').glob('*.bin'))
with open('/home/angus/.spacepdhcg-gpu.lock','a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 r=subprocess.run(['/home/angus/residual-probe-v334b',str(out),*map(str,files)],capture_output=True,text=True,timeout=120)
 print(r.stdout,r.stderr);r.check_returncode()
