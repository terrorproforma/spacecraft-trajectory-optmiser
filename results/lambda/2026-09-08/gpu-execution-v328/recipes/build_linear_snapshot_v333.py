from pathlib import Path
import sys,shutil,subprocess,hashlib,json
sys.path.insert(0,str(Path('scripts/gpu').resolve()))
from prepare_qoco_gpu import patch_restore_inaccurate_best
root=Path('/home/angus/build-qoco-linear-snapshot-v333');root.mkdir(exist_ok=False)
source=root/'source';shutil.copytree('/home/angus/build-qoco-gpu-device-ir-v137/source',source,ignore=shutil.ignore_patterns('.git','build','__pycache__'))
backend=source/'algebra/cuda/cudss_backend.cu'
text=backend.read_text()
anchor='#include "qoco_device_ir.cuh"'
assert text.count(anchor)==1
text=text.replace(anchor,anchor+'\n'+Path('build/performance/linear_snapshot_v333.cuh').read_text())
anchor='  cudss_solve_system(linsys_data, b, x);\n  if (!getenv'
assert text.count(anchor)==1
text=text.replace(anchor,'  cudss_solve_system(linsys_data, b, x);\n  static int diagnostic_index=0;\n  const int index=diagnostic_index++;\n  dump_linear(linsys_data,work,b,x,index,0);\n  if (!getenv')
anchor='    qoco_ir_solve(linsys_data, work, b, x, ir_tol, max_ir_iters);\n    return;'
assert text.count(anchor)==1
text=text.replace(anchor,'    qoco_ir_solve(linsys_data, work, b, x, ir_tol, max_ir_iters);\n    dump_linear(linsys_data,work,b,x,index,1);\n    return;')
backend.write_text(text)
cmake='/home/angus/spacecraft-trajectory-optmiser/.venv/bin/cmake'
flags=['-DQOCO_ALGEBRA_BACKEND=cuda','-DCMAKE_CUDA_ARCHITECTURES=120','-DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc','-DCMAKE_BUILD_TYPE=Release','-DQOCO_BUILD_TYPE=Release','-DBUILD_QOCO_DEMO=OFF','-DCUDSS_LIB=/home/angus/cudss-isolated-0.8.0.10/nvidia/cu12/lib/libcudss.so','-DCMAKE_CUDA_FLAGS=--default-stream per-thread -I/home/angus/cudss-isolated-0.8.0.10/nvidia/cu12/include','-DCMAKE_C_FLAGS=-Werror=implicit-function-declaration']
for i,cmd in enumerate([[cmake,'-S',str(source),'-B',str(root/'build'),*flags],[cmake,'--build',str(root/'build'),'--target','qoco','-j','8']]):
 with (root/f'build-{i}.log').open('x') as log:r=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
 if r.returncode:print((root/f'build-{i}.log').read_text()[-5000:]);r.check_returncode()
libs=list((root/'build').rglob('libqoco.so'));assert len(libs)==1
(root/'final').mkdir();out=root/'final/libqoco.so';shutil.copy2(libs[0],out)
(root/'report.json').write_text(json.dumps(dict(complete=True,binary=str(out),sha256=hashlib.sha256(out.read_bytes()).hexdigest(),api_sha256=hashlib.sha256((source/'src/qoco_api.c').read_bytes()).hexdigest()),indent=2));print((root/'report.json').read_text())
