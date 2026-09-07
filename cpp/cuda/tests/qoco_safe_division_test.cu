// Compile with the prepared QOCO include directory. Exercises the actual vendor
// macro on the GPU, including divisions near an IPM cone boundary.
#include "definitions.h"
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <limits>

__global__ void divide(const double* a,const double* b,double* result,int n) {
    const int i=threadIdx.x;
    if(i<n) result[i]=safe_div(a[i],b[i]);
}
int main() {
    const double a[]{1e-12,1e-20,1e-20,-1e-20,0.0,1.0,1.0};
    const double b[]{1e-18,1e-20,-1e-20,1e-20,1e-300,1.0,0.0};
    const double expected[]{1e6,1.0,-1.0,-1.0,0.0,1.0,std::numeric_limits<double>::max()};
    constexpr int n=7;
    double *da{},*db{},*dr{},result[n]{};
    if(cudaMalloc(&da,sizeof(a))!=cudaSuccess || cudaMalloc(&db,sizeof(b))!=cudaSuccess
        || cudaMalloc(&dr,sizeof(result))!=cudaSuccess) return 2;
    if(cudaMemcpy(da,a,sizeof(a),cudaMemcpyHostToDevice)!=cudaSuccess
        || cudaMemcpy(db,b,sizeof(b),cudaMemcpyHostToDevice)!=cudaSuccess) return 2;
    divide<<<1,32>>>(da,db,dr,n);
    if(cudaGetLastError()!=cudaSuccess
        || cudaMemcpy(result,dr,sizeof(result),cudaMemcpyDeviceToHost)!=cudaSuccess) return 2;
    int failures=0;
    for(int i=0;i<n;++i) {
        const bool valid=std::isfinite(result[i])
            && std::abs(result[i]-expected[i])<=1e-14*std::max(1.0,std::abs(expected[i]));
        if(!valid) {std::fprintf(stderr,"division %d: %.17g expected %.17g\n",i,result[i],expected[i]);++failures;}
    }
    cudaFree(da);cudaFree(db);cudaFree(dr);
    std::printf("{\"cases\":%d,\"failures\":%d}\n",n,failures);
    return failures?1:0;
}
