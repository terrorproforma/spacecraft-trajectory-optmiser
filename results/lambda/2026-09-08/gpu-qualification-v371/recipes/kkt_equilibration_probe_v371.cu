#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#define CUDA_CHECK(call) do{auto e=(call);if(e!=cudaSuccess){fprintf(stderr,"%d %s\n",__LINE__,cudaGetErrorString(e));exit(1);}}while(0)
struct LinSysData {int Kn; int *d_csr_rows,*d_csr_columns; double *d_csr_val,*d_factor_scale,*d_factor_max,*d_factor_val,*d_rhs_matrix_data,*d_xyz_matrix_data;};
#include "kkt_equilibration_v371.cuh"
template<class T> std::vector<T> read(FILE* f,int n){std::vector<T> a(n);if(fread(a.data(),sizeof(T),n,f)!=size_t(n))exit(2);return a;}
template<class T> T* upload(const std::vector<T>& a){T* d;CUDA_CHECK(cudaMalloc(&d,a.size()*sizeof(T)));CUDA_CHECK(cudaMemcpy(d,a.data(),a.size()*sizeof(T),cudaMemcpyHostToDevice));return d;}
double* allocate(int n){double* d;CUDA_CHECK(cudaMalloc(&d,n*sizeof(double)));return d;}
void output(FILE* f,double* d,int n){std::vector<double>a(n);CUDA_CHECK(cudaMemcpy(a.data(),d,n*sizeof(double),cudaMemcpyDeviceToHost));if(fwrite(a.data(),sizeof(double),n,f)!=size_t(n))exit(2);}
int main(int argc,char** argv){
 if(argc!=3)return 2;FILE* f=fopen(argv[1],"rb");if(!f)return 2;
 auto h=read<int>(f,2);int n=h[0],nnz=h[1];auto rows=read<int>(f,n+1),cols=read<int>(f,nnz);
 auto raw=read<double>(f,nnz),rhs=read<double>(f,n),sol=read<double>(f,n);fclose(f);
 LinSysData s{};s.Kn=n;s.d_csr_rows=upload(rows);s.d_csr_columns=upload(cols);s.d_csr_val=upload(raw);
 s.d_factor_scale=allocate(n);s.d_factor_max=allocate(n);s.d_factor_val=allocate(nnz);
 s.d_rhs_matrix_data=upload(rhs);s.d_xyz_matrix_data=upload(sol);
 f=fopen(argv[2],"wb");if(!f)return 2;
 for(int repetition=0;repetition<2;++repetition){
  if(repetition){for(double& x:raw)x*=4.;CUDA_CHECK(cudaMemcpy(s.d_csr_val,raw.data(),nnz*sizeof(double),cudaMemcpyHostToDevice));}
  CUDA_CHECK(cudaMemcpy(s.d_rhs_matrix_data,rhs.data(),n*sizeof(double),cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(s.d_xyz_matrix_data,sol.data(),n*sizeof(double),cudaMemcpyHostToDevice));
  qoco_kkt_equilibration::apply(&s);qoco_kkt_equilibration::rhs(&s);qoco_kkt_equilibration::solution(&s);
  output(f,s.d_factor_scale,n);output(f,s.d_factor_val,nnz);output(f,s.d_rhs_matrix_data,n);output(f,s.d_xyz_matrix_data,n);output(f,s.d_csr_val,nnz);
 }
 fclose(f);
 for(auto ptr:{s.d_csr_val,s.d_factor_scale,s.d_factor_max,s.d_factor_val,s.d_rhs_matrix_data,s.d_xyz_matrix_data})CUDA_CHECK(cudaFree(ptr));
 CUDA_CHECK(cudaFree(s.d_csr_rows));CUDA_CHECK(cudaFree(s.d_csr_columns));
}
