// Tests actual device decision/copy kernels, without vendor factorization.
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "definitions.h"
#include "enums.h"
#define SPACEPDHCG_CONTROL_KERNEL_TEST
#include "qoco_device_control.cuh"
#define REQUIRE(x) do { if (!(x)) { std::fprintf(stderr,"line %d: %s\n",__LINE__,#x); std::exit(1); } } while (0)
#define CUDA(x) REQUIRE((x)==cudaSuccess)

int main() {
    using namespace qoco_device_control;
    State* device; CUDA(cudaMalloc(&device,sizeof(State)));
    const int n=257,p=513,m=1027,total=n+p+2*m;
    std::vector<double> input(total+2),got(total+2);
    input.front()=input.back()=-999;
    for (int i=0;i<total;++i) input[i+1]=i+.25;
    double *live,*best; CUDA(cudaMalloc(&live,input.size()*8)); CUDA(cudaMalloc(&best,input.size()*8));
    for (int test=0;test<13;++test) {
        State s{}; s.alpha=1; s.dynamic_reg=1e-13; s.status=QOCO_UNSOLVED;
        for (int i=0;i<3;++i) s.metrics[i]=.5e-9;
        s.metrics[6]=123; s.metrics[7]=.01;
        int expected=QOCO_SOLVED, stop=1,save=1;
        if (test<3) { s.metrics[test]=1e-9; expected=QOCO_UNSOLVED;stop=0; }
        if (test==3) { s.metrics[0]=1.5e-9; s.metrics[3]=1; } // relative tolerance
        if (test==4) { s.alpha=0; expected=QOCO_UNSOLVED;stop=0; }
        if (test==5) { s.alpha=0; s.dynamic_reg=1e-6; expected=QOCO_SOLVED_INACCURATE; }
        if (test==6) { s.alpha=0; s.dynamic_reg=1e-6; s.metrics[1]=1e-5; expected=QOCO_NUMERICAL_ERROR; }
        if (test==7) s.alpha=1e-8; // exact alpha boundary must use normal tolerance
        if (test==8) { s.best_valid=1; s.best_metric=.0005; s.best_obj=456; save=0; } // exact tie
        if (test==9) { s.best_valid=1; s.best_metric=.0001; s.best_obj=456; save=0; }
        if (test==10) { s.metrics[2]=NAN; expected=QOCO_UNSOLVED;stop=0;save=0; }
        if (test==11) { s.metrics[2]=INFINITY; expected=QOCO_UNSOLVED;stop=0;save=0; }
        if (test==12) { s.best_valid=1;s.best_metric=2;s.best_obj=456; }
        CUDA(cudaMemcpy(device,&s,sizeof(s),cudaMemcpyHostToDevice));
        CUDA(cudaMemcpy(live,input.data(),input.size()*8,cudaMemcpyHostToDevice));
        CUDA(cudaMemcpy(best,input.data(),input.size()*8,cudaMemcpyHostToDevice));
        CUDA(cudaMemset(best+1,0,total*8));
        decide<<<1,1>>>(device,1e-9,1e-9,1e-6,1e-6,17);
        best_vectors<false><<<17,256>>>(device,n,p,m,live+1,live+1+n,live+1+n+p,live+1+n+p+m,
            best+1,best+1+n,best+1+n+p,best+1+n+p+m);
        CUDA(cudaMemcpy(&s,device,sizeof(s),cudaMemcpyDeviceToHost));
        CUDA(cudaMemcpy(got.data(),best,got.size()*8,cudaMemcpyDeviceToHost));
        REQUIRE(s.stop==stop && s.status==expected && s.save==save);
        REQUIRE(got.front()==-999 && got.back()==-999);
        for (int i=0;i<total;++i) REQUIRE(got[i+1]==(save ? input[i+1] : 0));
        if (save) REQUIRE(s.best_iter==17 && s.best_obj==123);
        if (test==4) REQUIRE(s.dynamic_reg==1e-12);
        if (test==5 || test==6) REQUIRE(std::abs(s.dynamic_reg-1e-5)<1e-20);
    }
    for (int valid=0;valid<2;++valid) for (int inaccurate=0;inaccurate<2;++inaccurate) {
        State s{};s.best_valid=valid;s.best_metric=inaccurate ? 1.0 : 2.0;
        s.best_pres=1;s.best_dres=2;s.best_gap=3;s.best_obj=4;
        CUDA(cudaMemcpy(device,&s,sizeof(s),cudaMemcpyHostToDevice));
        CUDA(cudaMemcpy(best,input.data(),input.size()*8,cudaMemcpyHostToDevice));
        CUDA(cudaMemcpy(live,input.data(),input.size()*8,cudaMemcpyHostToDevice));
        CUDA(cudaMemset(live+1,0,total*8));
        restore_decision<<<1,1>>>(device,QOCO_MAX_ITER);
        best_vectors<true><<<17,256>>>(device,n,p,m,live+1,live+1+n,live+1+n+p,live+1+n+p+m,
            best+1,best+1+n,best+1+n+p,best+1+n+p+m);
        CUDA(cudaMemcpy(&s,device,sizeof(s),cudaMemcpyDeviceToHost));
        CUDA(cudaMemcpy(got.data(),live,got.size()*8,cudaMemcpyDeviceToHost));
        REQUIRE(s.restored==valid && s.status==(valid && inaccurate ? QOCO_SOLVED_INACCURATE : QOCO_MAX_ITER));
        REQUIRE(got.front()==-999 && got.back()==-999);
        for (int i=0;i<total;++i) REQUIRE(got[i+1]==(valid ? input[i+1] : 0));
        if (valid) REQUIRE(s.metrics[0]==1 && s.metrics[1]==2 && s.metrics[2]==3 && s.metrics[6]==4);
    }
    CUDA(cudaFree(device));CUDA(cudaFree(live));CUDA(cudaFree(best));
    std::puts("PASS: 13 stop/best/regularization cases and 4 best restoration cases with vector canaries");
}
