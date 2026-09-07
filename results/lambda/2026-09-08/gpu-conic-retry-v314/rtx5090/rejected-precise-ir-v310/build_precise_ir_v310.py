from pathlib import Path
import subprocess,shutil,os,json,time,hashlib
root=Path('/home/angus/build-qoco-precise-ir-v310');root.mkdir(exist_ok=False)
source=root/'source';shutil.copytree('/home/angus/build-qoco-gpu-device-ir-v137/source',source,ignore=shutil.ignore_patterns('.git','build','__pycache__'))
header=Path('build/performance/qoco_precise_ir_v310.cuh').read_text();(source/'algebra/cuda/qoco_precise_ir.cuh').write_text(header)
algebra=source/'algebra/cuda/cuda_linalg.cu';algebra.write_text(algebra.read_text()+'\n#include "qoco_precise_ir.cuh"\n')
backend=source/'algebra/cuda/cudss_backend.cu';text=backend.read_text()
anchor='static QOCOFloat compute_linsys_residual('
assert text.count(anchor)==1
text=text.replace(anchor,'extern "C" void qoco_gpu_precise_ir(QOCOWorkspace*,const double*,const double*,double*,double*,double);\n\n'+anchor)
anchor='  QOCOFloat* nt_scaling = get_data_vectorf(work->nt_scaling);'
assert text.count(anchor)==1
text=text.replace(anchor,'''  if (!getenv("SPACEPDHCG_TEST_QOCO_PRECISE_IR_DISABLE")) {
    qoco_gpu_precise_ir(work, b, x, residual_scratch, linsys_data->d_rhs_matrix_data,
                        linsys_data->kkt_static_reg_P);
    return download ? inf_norm(residual_scratch, linsys_data->Kn) : 0.0;
  }
'''+anchor)
backend.write_text(text)
cmake='/home/angus/spacecraft-trajectory-optmiser/.venv/bin/cmake'
env=os.environ.copy();env['CUDSS_ROOT']='/home/angus/cudss-isolated-0.8.0.10/nvidia/cu12'
commands=[[cmake,'-S',str(source),'-B',str(root/'build'),'-DQOCO_ALGEBRA_BACKEND=cuda','-DCMAKE_CUDA_ARCHITECTURES=120','-DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc','-DQOCO_BUILD_TYPE=Release','-DCMAKE_BUILD_TYPE=Release','-DBUILD_QOCO_DEMO=OFF'],[cmake,'--build',str(root/'build'),'-j','8']]
for i,cmd in enumerate(commands):
 with (root/f'build-{i}.log').open('w') as f:r=subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT)
 if r.returncode:print((root/f'build-{i}.log').read_text()[-7000:]);r.check_returncode()
libs=list((root/'build').rglob('libqoco.so'));assert len(libs)==1
(root/'final').mkdir();out=root/'final/libqoco.so';shutil.copy2(libs[0],out)
(root/'report.json').write_text(json.dumps(dict(complete=True,binary=str(out),sha256=hashlib.sha256(out.read_bytes()).hexdigest()),indent=2))
print((root/'report.json').read_text())
