from pathlib import Path
import subprocess,shutil,json,hashlib
root=Path('/home/angus/build-qoco-interior-nt-v369');p=root/'source/src/cone.cu';s=p.read_text()
anchor='#include "qoco_nt_precision.cuh"';assert s.count(anchor)==1
s=s.replace(anchor,'#include "qoco_soc_step.cuh"\n'+anchor);p.write_text(s)
with (root/'build-retry.log').open('x') as log:
 r=subprocess.run(['/home/angus/spacecraft-trajectory-optmiser/.venv/bin/cmake','--build',str(root/'build'),'--target','qoco','-j','8'],stdout=log,stderr=subprocess.STDOUT)
r.check_returncode();libs=list((root/'build').rglob('libqoco.so'));assert len(libs)==1
(root/'final').mkdir();out=root/'final/libqoco.so';shutil.copyfile(libs[0],out)
(root/'report.json').write_text(json.dumps(dict(complete=True,binary=str(out),sha256=hashlib.sha256(out.read_bytes()).hexdigest(),first_build_error='NT precision header appeared before its DD type declaration; fixed include ordering before any candidate replay',api_sha256=hashlib.sha256((root/'source/src/qoco_api.c').read_bytes()).hexdigest()),indent=2));print((root/'report.json').read_text())
r=Path('build/performance/replay_interior_nt_v369.py').read_text().replace('interior-nt-replay-v369','interior-nt-replay-v369b')
Path('build/performance/replay_interior_nt_v369b.py').write_text(r)
