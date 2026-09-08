// SPDX-License-Identifier: Apache-2.0
// Analytic boundary cases and the false-zero cone captured from an actual QP.
#include <cuda_runtime.h>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
using QOCOFloat=double;
using QOCOInt=int;
#define QOCOFloat_MAX DBL_MAX
#define qoco_min(a,b) (((a)<(b))?(a):(b))
#define qoco_sqrt(a) sqrt(a)
#ifndef SPACEPDHCG_TEST_SOC_STEP_HEADER
#define SPACEPDHCG_TEST_SOC_STEP_HEADER "qoco_soc_step.cuh"
#endif
#include SPACEPDHCG_TEST_SOC_STEP_HEADER
#define CHECK(call) do {auto e=(call);if(e!=cudaSuccess){fprintf(stderr,"line %d: %s\n",__LINE__,cudaGetErrorString(e));exit(1);}}while(0)
struct Case { int n; double limit,expected; double x[257],dx[257]; };
__global__ void evaluate(const Case* c,int n,double* out) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n) out[i+1]=soc_step_length_dev(c[i].x,c[i].dx,c[i].n,c[i].limit);
}
int main() {
    std::vector<Case> cases;
    auto add=[&](double x0,double x1,double d0,double d1,double expected,double limit=1.) {
        Case c{};c.n=3;c.limit=limit;c.expected=expected;c.x[0]=x0;c.x[1]=x1;c.dx[0]=d0;c.dx[1]=d1;cases.push_back(c);
    };
    add(2,0,-4,0,.5); // Scalar boundary.
    add(2,0,0,4,.5); // Quadratic boundary.
    add(2,0,-2,2,.5); // Linear polynomial.
    add(2,0,1,0,1); // Interior direction.
    add(1,1,0,1,0); // Outward from boundary.
    add(1,1,1,0,1); // Inward from boundary.
    add(1,1,0,-4,.5); // Enter, then exit the opposite side.
    add(1,1,1,1,1); // Along boundary.
    add(2,0,0,4,.25,.25); // Caller cap.
    Case tangent{};tangent.n=3;tangent.limit=1;tangent.x[0]=tangent.x[1]=1;tangent.dx[2]=1;tangent.expected=0;cases.push_back(tangent);
    for(int n:{4,33,257}) {
        Case c{};c.n=n;c.limit=1;c.expected=.5;c.x[0]=2;c.dx[n-1]=4;cases.push_back(c);
    }
    Case live{};live.n=4;live.limit=1;live.expected=0.8768290586498959;
    double x[]={16.179665645791484,-11.510165869819549,-11.367325792600852,.28559829081729216};
    double dx[]={5.603508152214337e-11,-1.2751664401219522e-10,-2.7825205138221965e-11,-3.0721304208742944e-9};
    for(int j=0;j<4;++j){live.x[j]=x[j];live.dx[j]=dx[j];}cases.push_back(live);
    Case* device;double* output;const int n=static_cast<int>(cases.size());
    std::vector<double> result(n+2,-777.25);
    CHECK(cudaMalloc(&device,n*sizeof(Case)));CHECK(cudaMalloc(&output,(n+2)*sizeof(double)));
    CHECK(cudaMemcpy(device,cases.data(),n*sizeof(Case),cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(output,result.data(),result.size()*sizeof(double),cudaMemcpyHostToDevice));
    evaluate<<<(n+63)/64,64>>>(device,n,output);CHECK(cudaGetLastError());CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(result.data(),output,result.size()*sizeof(double),cudaMemcpyDeviceToHost));
    CHECK(cudaFree(device));CHECK(cudaFree(output));
    if(result.front()!=-777.25 || result.back()!=-777.25)return 2;
    for(int i=0;i<n;++i) {
        double expected=cases[i].expected,got=result[i+1];
        if(!std::isfinite(got) || std::abs(got-expected)>2e-14) {
            fprintf(stderr,"case %d: got %.17g expected %.17g\n",i,got,expected);return 2;
        }
    }
    printf("%d SOC step cases and output canaries passed\n",n);
}
