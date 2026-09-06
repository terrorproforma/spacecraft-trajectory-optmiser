// Exercise prepared terminal kernels under graph replay without vendor libraries.
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include "definitions.h"
#include "enums.h"
#define SPACEPDHCG_CONTROL_KERNEL_TEST
#include "qoco_device_control.cuh"
#include "qoco_ipm_parameters.cuh"
// Extracted verbatim from the frozen runtime by the evidence helper.
#include "qoco_ipm_terminal_kernels.cuh"
#define REQUIRE(x) do { if (!(x)) { std::fprintf(stderr,"line %d: %s\n",__LINE__,#x); std::exit(1); } } while (0)
#define CUDA(x) REQUIRE((x)==cudaSuccess)
int main() {
    using namespace qoco_device_control;
    int cases=0;
    for (auto dims : std::vector<std::vector<int>>{{257,513,1027}, {1,0,0}, {1,0,3}, {1,2,0}}) {
        const int n=dims[0],p=dims[1],m=dims[2],total=n+p+2*m;
        std::vector<double> input(total+2),saved(total+2),scale(total+2),expected(total+2),got(total+2);
        input.front()=input.back()=saved.front()=saved.back()=-999;
        for (int i=1;i<=total;++i) {
            input[i]=(i%31-15)*.137; saved[i]=(i%19-9)*.391; scale[i]=.731+(i%11)*.029;
        }
        double *live,*best,*scaling;
        State* state; QocoIpmParameters* parameters;
        CUDA(cudaMalloc(&live,input.size()*8)); CUDA(cudaMalloc(&best,input.size()*8));
        CUDA(cudaMalloc(&scaling,input.size()*8)); CUDA(cudaMalloc(&state,sizeof(State)));
        CUDA(cudaMalloc(&parameters,sizeof(QocoIpmParameters)));
        CUDA(cudaMemcpy(best,saved.data(),saved.size()*8,cudaMemcpyHostToDevice));
        CUDA(cudaMemcpy(scaling,scale.data(),scale.size()*8,cudaMemcpyHostToDevice));
        cudaGraph_t graph; cudaGraphExec_t executable;
        CUDA(cudaStreamBeginCapture(cudaStreamPerThread,cudaStreamCaptureModeThreadLocal));
        terminal_restore<<<1,1>>>(state);
        best_vectors<true><<<9,256>>>(state,n,p,m,live+1,live+1+n,live+1+n+p,live+1+n+p+m,
            best+1,best+1+n,best+1+n+p,best+1+n+p+m);
        terminal_unscale<<<9,256>>>(n,p,m,live+1,live+1+n,live+1+n+p,live+1+n+p+m,
            scaling+1,scaling+1+n,scaling+1+n+p+m,scaling+1+n+p,parameters);
        CUDA(cudaStreamEndCapture(cudaStreamPerThread,&graph));
        CUDA(cudaGraphInstantiate(&executable,graph,nullptr,nullptr,0));
        for (int status : {QOCO_SOLVED,QOCO_SOLVED_INACCURATE,QOCO_MAX_ITER,QOCO_NUMERICAL_ERROR,QOCO_UNSOLVED})
        for (int valid : {0,1}) for (double metric : {.5,1.,2.}) for (double kinv : {.13,7.31}) {
            State h{}; h.status=status; h.restored=1; h.best_valid=valid; h.best_metric=metric;
            h.best_pres=.01; h.best_dres=.02; h.best_gap=.03; h.best_obj=13;
            for (int j=0;j<8;++j) h.metrics[j]=j+1;
            State reference=h;
            const bool restore=valid && (status==QOCO_NUMERICAL_ERROR || status==QOCO_MAX_ITER);
            reference.restored=restore;
            if (restore) {
                reference.metrics[0]=.01; reference.metrics[1]=.02; reference.metrics[2]=.03;
                reference.metrics[6]=13;
                if (metric<=1.) reference.status=QOCO_SOLVED_INACCURATE;
            }
            expected=restore ? saved : input;
            for (int i=0;i<total;++i) {
                volatile double first=expected[i+1]*scale[i+1];
                expected[i+1]=(i>=n && i<n+p) || i>=n+p+m ? first*kinv : first;
            }
            QocoIpmParameters param{}; param.kinv=kinv;
            CUDA(cudaMemcpy(live,input.data(),input.size()*8,cudaMemcpyHostToDevice));
            CUDA(cudaMemcpy(state,&h,sizeof(h),cudaMemcpyHostToDevice));
            CUDA(cudaMemcpy(parameters,&param,sizeof(param),cudaMemcpyHostToDevice));
            CUDA(cudaGraphLaunch(executable,cudaStreamPerThread));
            CUDA(cudaStreamSynchronize(cudaStreamPerThread));
            CUDA(cudaMemcpy(&h,state,sizeof(h),cudaMemcpyDeviceToHost));
            CUDA(cudaMemcpy(got.data(),live,got.size()*8,cudaMemcpyDeviceToHost));
            REQUIRE(std::memcmp(&h,&reference,sizeof(h))==0);
            REQUIRE(std::memcmp(got.data(),expected.data(),got.size()*8)==0);
            ++cases;
        }
        CUDA(cudaGraphExecDestroy(executable)); CUDA(cudaGraphDestroy(graph));
        CUDA(cudaFree(parameters)); CUDA(cudaFree(state)); CUDA(cudaFree(scaling)); CUDA(cudaFree(best)); CUDA(cudaFree(live));
    }
    std::printf("PASS: %d terminal graph cases; exact state/vector parity and canaries\n",cases);
}
