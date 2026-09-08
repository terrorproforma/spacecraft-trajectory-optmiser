// Experimental per-factorization symmetric KKT equilibration.
// Preserve raw K and all original-equation iterative-refinement residuals.
namespace qoco_kkt_equilibration {
__global__ void identity(double* scale,int n) {
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)scale[i]=1.;
}
__global__ void row_max(const int* rows,const int* columns,const double* raw,
 const double* scale,double* maxima,int n) {
 int row=blockIdx.x*blockDim.x+threadIdx.x;if(row>=n)return;
 for(int k=rows[row];k<rows[row+1];++k){
  int col=columns[k];double value=fabs((raw[k]*scale[row])*scale[col]);
  // Positive IEEE-754 doubles have the same order as their unsigned bits.
  // Preserve nonfinite input as a nonfinite norm; never turn it into a zero row.
  if(!isfinite(value))value=INFINITY;
  atomicMax(reinterpret_cast<unsigned long long*>(maxima+row),__double_as_longlong(value));
  atomicMax(reinterpret_cast<unsigned long long*>(maxima+col),__double_as_longlong(value));
 }
}
__global__ void update(double* scale,const double* maxima,int n){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n){
  double v=maxima[i];if(v>0.&&isfinite(v))scale[i]/=sqrt(v);
 }
}
__global__ void matrix(const int* rows,const int* columns,const double* raw,
 const double* scale,double* out,int n){
 int row=blockIdx.x*blockDim.x+threadIdx.x;if(row>=n)return;
 for(int k=rows[row];k<rows[row+1];++k)out[k]=(raw[k]*scale[row])*scale[columns[k]];
}
__global__ void vector(double* data,const double* scale,int n){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)data[i]*=scale[i];
}
static void apply(LinSysData* s){
 int blocks=(s->Kn+255)/256;
 identity<<<blocks,256>>>(s->d_factor_scale,s->Kn);
 for(int pass=0;pass<4;++pass){
  CUDA_CHECK(cudaMemsetAsync(s->d_factor_max,0,s->Kn*sizeof(double)));
  row_max<<<blocks,256>>>(s->d_csr_rows,s->d_csr_columns,s->d_csr_val,s->d_factor_scale,s->d_factor_max,s->Kn);
  update<<<blocks,256>>>(s->d_factor_scale,s->d_factor_max,s->Kn);
 }
 matrix<<<blocks,256>>>(s->d_csr_rows,s->d_csr_columns,s->d_csr_val,s->d_factor_scale,s->d_factor_val,s->Kn);
 CUDA_CHECK(cudaGetLastError());
}
static void rhs(LinSysData* s,cudaStream_t stream=0){
 vector<<<(s->Kn+255)/256,256,0,stream>>>(s->d_rhs_matrix_data,s->d_factor_scale,s->Kn);CUDA_CHECK(cudaGetLastError());
}
static void solution(LinSysData* s,cudaStream_t stream=0){
 vector<<<(s->Kn+255)/256,256,0,stream>>>(s->d_xyz_matrix_data,s->d_factor_scale,s->Kn);CUDA_CHECK(cudaGetLastError());
}
}
