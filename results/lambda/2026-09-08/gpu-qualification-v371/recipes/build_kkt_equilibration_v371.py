from pathlib import Path
import sys,shutil,subprocess,hashlib,json
sys.path.insert(0,str(Path('scripts/gpu').resolve()))
from prepare_qoco_gpu import patch_restore_inaccurate_best
root=Path('/home/angus/build-qoco-kkt-equilibration-v371');root.mkdir(exist_ok=False)
source=root/'source';shutil.copytree('/home/angus/build-qoco-soc-step-v358/source',source,ignore=shutil.ignore_patterns('.git','build','__pycache__'))


p=source/'algebra/cuda/cudss_backend.cu';text=p.read_text()
def once(old,new):
 global text
 assert text.count(old)==1,(old,text.count(old))
 text=text.replace(old,new)
once('  QOCOFloat* d_csr_val;', '  QOCOFloat* d_csr_val;\n  double *d_factor_val, *d_factor_scale, *d_factor_max;')
once('static LinSysData* cudss_setup(', '#include "qoco_kkt_equilibration.cuh"\nstatic LinSysData* cudss_setup(')
(p.parent/'qoco_kkt_equilibration.cuh').write_text(Path('build/performance/kkt_equilibration_v371.cuh').read_text())
once('  // Determine data types', '''
  CUDA_CHECK(cudaMalloc(&linsys_data->d_factor_val,kkt_nnz*sizeof(double)));
  CUDA_CHECK(cudaMalloc(&linsys_data->d_factor_scale,linsys_data->Kn*sizeof(double)));
  CUDA_CHECK(cudaMalloc(&linsys_data->d_factor_max,linsys_data->Kn*sizeof(double)));
  CUDA_CHECK(cudaMemcpy(linsys_data->d_factor_val,csr_val,kkt_nnz*sizeof(double),cudaMemcpyDeviceToDevice));
  qoco_kkt_equilibration::identity<<<(linsys_data->Kn+255)/256,256>>>(linsys_data->d_factor_scale,linsys_data->Kn);
  CUDA_CHECK(cudaGetLastError());
  // Determine data types''')
once('(int64_t)kkt_nnz, csr_row_ptr, NULL, csr_col_ind, csr_val,','(int64_t)kkt_nnz, csr_row_ptr, NULL, csr_col_ind, linsys_data->d_factor_val,')
import re
text,count=re.subn(r'(cudssMatrixSetValues\(linsys_data->K_csr,\s*)linsys_data->d_csr_val',r'\1linsys_data->d_factor_val',text);assert count==2,count
once('  (void)kkt_dynamic_reg;', '  (void)kkt_dynamic_reg;\n  qoco_kkt_equilibration::apply(linsys_data);')
anchor='''  CUDSS_CHECK(qoco_ir_execute(linsys_data->ir, 
      linsys_data->handle, CUDSS_PHASE_SOLVE, linsys_data->config,
      linsys_data->data, linsys_data->K_csr, linsys_data->d_xyz_matrix,
      linsys_data->d_rhs_matrix));'''
once(anchor,'  qoco_kkt_equilibration::rhs(linsys_data);\n'+anchor+'\n  qoco_kkt_equilibration::solution(linsys_data);')
once('  cudaFree(linsys_data->d_csr_val);','  cudaFree(linsys_data->d_factor_val);\n  cudaFree(linsys_data->d_factor_scale);\n  cudaFree(linsys_data->d_factor_max);\n  cudaFree(linsys_data->d_csr_val);')
p.write_text(text)
# Parent IPM graph: surround both initial and correction child solves.
p=p.parent/'qoco_ipm_graph.cuh';t=p.read_text();anchor='    qoco_ipm_child(c.linear);';assert t.count(anchor)==2
t=t.replace(anchor,'    qoco_kkt_equilibration::rhs(s,cudaStreamPerThread);\n'+anchor+'\n    qoco_kkt_equilibration::solution(s,cudaStreamPerThread);');p.write_text(t)
# Standalone retained IR graph uses its dedicated runtime stream.
p=p.parent/'qoco_device_ir.cuh';t=p.read_text();anchor='''    CUDSS_CHECK(g_cuda_funcs.cudssExecute(s->handle, CUDSS_PHASE_SOLVE, s->config, s->data,
                                         s->K_csr, s->d_xyz_matrix, s->d_rhs_matrix));''';assert t.count(anchor)==1
t=t.replace(anchor,'    qoco_kkt_equilibration::rhs(s,stream);\n'+anchor+'\n    qoco_kkt_equilibration::solution(s,stream);');p.write_text(t)
cmake='/home/angus/spacecraft-trajectory-optmiser/.venv/bin/cmake'
flags=['-DQOCO_ALGEBRA_BACKEND=cuda','-DCMAKE_CUDA_ARCHITECTURES=120','-DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc','-DCMAKE_BUILD_TYPE=Release','-DQOCO_BUILD_TYPE=Release','-DBUILD_QOCO_DEMO=OFF','-DCUDSS_LIB=/home/angus/cudss-isolated-0.8.0.10/nvidia/cu12/lib/libcudss.so','-DCMAKE_CUDA_FLAGS=--default-stream per-thread -I/home/angus/cudss-isolated-0.8.0.10/nvidia/cu12/include','-DCMAKE_C_FLAGS=-Werror=implicit-function-declaration']
for i,cmd in enumerate([[cmake,'-S',str(source),'-B',str(root/'build'),*flags],[cmake,'--build',str(root/'build'),'--target','qoco','-j','8']]):
 with (root/f'build-{i}.log').open('x') as log:r=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
 if r.returncode:print((root/f'build-{i}.log').read_text()[-5000:]);r.check_returncode()
libs=list((root/'build').rglob('libqoco.so'));assert len(libs)==1
(root/'final').mkdir();out=root/'final/libqoco.so';shutil.copy2(libs[0],out)
(root/'report.json').write_text(json.dumps(dict(complete=True,binary=str(out),sha256=hashlib.sha256(out.read_bytes()).hexdigest(),api_sha256=hashlib.sha256((source/'src/qoco_api.c').read_bytes()).hexdigest()),indent=2));print((root/'report.json').read_text())
