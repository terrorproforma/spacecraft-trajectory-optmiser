
#include <cuda_runtime.h>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cfloat>
using QOCOFloat=double;using QOCOInt=int;
#define qoco_sqrt(a) sqrt(a)
#define safe_div(a,b) (((b)!=0.0)?((a)/(b)):DBL_MAX)
#include "arithmetic.cuh"
__device__ double soc_residual2(const double* u,int n){double r=u[0]*u[0];for(int j=1;j<n;++j)r-=u[j]*u[j];return r;}
__device__ double qoco_dot_dev(const double* u,const double* v,int n){double r=0;for(int j=0;j<n;++j)r+=u[j]*v[j];return r;}
__device__ void scale_arrayf_dev(const double* u,double* v,double f,int n){for(int j=0;j<n;++j)v[j]=u[j]*f;}
namespace legacy {
#include "old_kernel.cuh"
}
namespace candidate {
#include "new_kernel.cuh"
}
#define CHECK(call) do{auto e=(call);if(e!=cudaSuccess){fprintf(stderr,"%d: %s\n",__LINE__,cudaGetErrorString(e));exit(1);}}while(0)
template<class T> std::vector<T> read(FILE* f,int n){std::vector<T> a(n);if(fread(a.data(),sizeof(T),n,f)!=size_t(n))exit(2);return a;}
template<class T> T* upload(const std::vector<T>& a){T* d;CHECK(cudaMalloc(&d,a.size()*sizeof(T)));CHECK(cudaMemcpy(d,a.data(),a.size()*sizeof(T),cudaMemcpyHostToDevice));return d;}
double* allocate(int n){double* d;CHECK(cudaMalloc(&d,n*sizeof(double)));return d;}
void output(FILE* f,double* d,int n){std::vector<double> h(n);CHECK(cudaMemcpy(h.data(),d,n*sizeof(double),cudaMemcpyDeviceToHost));fwrite(h.data(),sizeof(double),n,f);}
int main(int argc,char** argv){
 if(argc!=3)return 2;FILE* f=fopen(argv[1],"rb");if(!f)return 2;
 auto h=read<int>(f,2);auto q=read<int>(f,h[0]);auto s=read<double>(f,h[1]);auto z=read<double>(f,h[1]);fclose(f);
 int tri=0;for(auto n:q)tri+=n*(n+1)/2;
 auto dq=upload(q);auto ds=upload(s);auto dz=upload(z);
 auto W=allocate(tri),WtW=allocate(tri),Winv=allocate(tri),nt=allocate(h[1]+h[0]),sb=allocate(h[1]),zb=allocate(h[1]);
 f=fopen(argv[2],"wb");if(!f)return 2;
 legacy::compute_nt_scaling_kernel<<<(h[0]+127)/128,128>>>(W,WtW,nt,Winv,ds,dz,sb,zb,0,h[0],dq);
 CHECK(cudaGetLastError());CHECK(cudaDeviceSynchronize());output(f,nt,h[1]+h[0]);output(f,WtW,tri);
 candidate::compute_nt_scaling_kernel<<<(h[0]+127)/128,128>>>(W,WtW,nt,Winv,ds,dz,sb,zb,0,h[0],dq);
 CHECK(cudaGetLastError());CHECK(cudaDeviceSynchronize());output(f,nt,h[1]+h[0]);output(f,WtW,tri);fclose(f);
 for(auto p:{W,WtW,Winv,nt,sb,zb,ds,dz})CHECK(cudaFree(p));CHECK(cudaFree(dq));
 printf("%d paired SOC normalizations evaluated\n",h[0]);
}
