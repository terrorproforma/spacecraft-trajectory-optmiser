#include <cuda_runtime.h>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cfloat>
using QOCOFloat=double;using QOCOInt=int;
#define QOCOFloat_MAX DBL_MAX
#define qoco_min(a,b) (((a)<(b))?(a):(b))
#define qoco_sqrt(a) sqrt(a)
#include "arithmetic.cuh"
namespace legacy {
#include "old_step.cuh"
}
namespace candidate {
#include "new_step.cuh"
}
#define CHECK(call) do{auto e=(call);if(e!=cudaSuccess){fprintf(stderr,"%d: %s\n",__LINE__,cudaGetErrorString(e));exit(1);}}while(0)
template<class T> std::vector<T> read(FILE* f,int n){std::vector<T> a(n);if(fread(a.data(),sizeof(T),n,f)!=size_t(n))exit(2);return a;}
template<class T> T* upload(const std::vector<T>& a){T* d;CHECK(cudaMalloc(&d,a.size()*sizeof(T)));CHECK(cudaMemcpy(d,a.data(),a.size()*sizeof(T),cudaMemcpyHostToDevice));return d;}
__global__ void determinants(const double* values,const int* offsets,int count,double* output){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;
    const double* u=values+offsets[i];int n=(offsets[i+1]-offsets[i])/2;
    output[2*i]=legacy::soc_step_length_dev(u,u+n,n,1.0);
    output[2*i+1]=candidate::soc_step_length_dev(u,u+n,n,1.0);
}
int main(int argc,char** argv){
    if(argc!=3)return 2;FILE* f=fopen(argv[1],"rb");if(!f)return 2;
    auto h=read<int>(f,2);auto offsets=read<int>(f,h[0]+1);auto values=read<double>(f,h[1]);fclose(f);
    auto dv=upload(values);auto di=upload(offsets);double* out;CHECK(cudaMalloc(&out,h[0]*2*sizeof(double)));
    determinants<<<(h[0]+255)/256,256>>>(dv,di,h[0],out);CHECK(cudaGetLastError());CHECK(cudaDeviceSynchronize());
    std::vector<double> result(h[0]*2);CHECK(cudaMemcpy(result.data(),out,result.size()*sizeof(double),cudaMemcpyDeviceToHost));
    f=fopen(argv[2],"wb");if(!f)return 2;fwrite(result.data(),sizeof(double),result.size(),f);fclose(f);
    CHECK(cudaFree(dv));CHECK(cudaFree(di));CHECK(cudaFree(out));printf("%d SOC step bounds evaluated\n",h[0]);
}
