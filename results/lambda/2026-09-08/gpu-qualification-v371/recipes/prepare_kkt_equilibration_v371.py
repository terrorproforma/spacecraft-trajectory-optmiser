from pathlib import Path
s=Path('build/performance/build_interior_nt_v369.py').read_text().replace('build-qoco-interior-nt-v369','build-qoco-kkt-equilibration-v371').replace('build-qoco-interior-step-v367','build-qoco-soc-step-v358')
a=s.index('header=Path(');b=s.index('cmake=',a)
patch=r"""
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
"""
Path('build/performance/build_kkt_equilibration_v371.py').write_text(s[:a]+patch+s[b:])
r=Path('build/performance/replay_interior_nt_v369b.py').read_text().replace('interior-nt-replay-v369b','kkt-equilibration-replay-v371').replace('build-qoco-interior-nt-v369','build-qoco-kkt-equilibration-v371').replace('369','371').replace('interior_nt','kkt_equilibration')
Path('build/performance/replay_kkt_equilibration_v371.py').write_text(r)
