from pathlib import Path
import subprocess,shutil,hashlib,json
root=Path('/home/angus/build-qoco-precise-ir-v310')
cmake='/home/angus/spacecraft-trajectory-optmiser/.venv/bin/cmake'
commands=[[cmake,'-S',str(root/'source'),'-B',str(root/'build'),'-DCUDSS_LIB=/home/angus/cudss-isolated-0.8.0.10/nvidia/cu12/lib/libcudss.so','-DCMAKE_CUDA_FLAGS=--default-stream per-thread -I/home/angus/cudss-isolated-0.8.0.10/nvidia/cu12/include','-DCMAKE_C_FLAGS=-Werror=implicit-function-declaration'],[cmake,'--build',str(root/'build'),'--target','qoco','-j','8']]
for i,cmd in enumerate(commands):
 with (root/f'resume-{i}.log').open('w') as f:r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
 if r.returncode:print((root/f'resume-{i}.log').read_text()[-6000:]);r.check_returncode()
libs=list((root/'build').rglob('libqoco.so'));assert len(libs)==1
(root/'final').mkdir(exist_ok=True);out=root/'final/libqoco.so';shutil.copy2(libs[0],out)
(root/'report.json').write_text(json.dumps(dict(complete=True,binary=str(out),sha256=hashlib.sha256(out.read_bytes()).hexdigest()),indent=2))
print((root/'report.json').read_text())
