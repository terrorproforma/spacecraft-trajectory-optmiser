#include "spacepdhcg/cuda/gtoc12_qoco_c_api.h"
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>

#define REQUIRE(x) do { if (!(x)) { std::fprintf(stderr,"line %d: %s\n",__LINE__,#x); std::exit(1); } } while (0)
#define CUDA(x) REQUIRE((x)==cudaSuccess)

struct Observed { int calls, qualified, status, iterations; };
__global__ void consume(const spacepdhcg_gtoc12_qoco_report* report, Observed* result) {
    ++result->calls;
    result->qualified=report->qualified; result->status=report->qoco_status;
    result->iterations=report->iterations;
}
int consumer(void* opaque, const spacepdhcg_gtoc12_qoco_report* report, const double*, void* stream) {
    consume<<<1,1,0,static_cast<cudaStream_t>(stream)>>>(report,static_cast<Observed*>(opaque));
    return cudaGetLastError()==cudaSuccess ? 0 : 1;
}

int main() {
    if (!std::getenv("SPACEPDHCG_QOCO_LIBRARY")) { std::puts("SKIP: prepared QOCO required"); return 0; }
    double times[4]{0,.003,.006,.009}, states[28]{}, controls[16]{}, boundary[12]{};
    double weights[4]{.003,.003,.003,0};
    const double omega=std::pow(2.7,-1.5);
    for (int k=0;k<4;++k) {
        states[7*k]=2.7*std::cos(omega*times[k]); states[7*k+1]=2.7*std::sin(omega*times[k]);
        states[7*k+3]=-2.7*omega*std::sin(omega*times[k]);
        states[7*k+4]=2.7*omega*std::cos(omega*times[k]); states[7*k+6]=1;
    }
    for (int j=0;j<6;++j) { boundary[j]=states[j]; boundary[6+j]=states[21+j]; }
    setenv("SPACEPDHCG_TEST_QOCO_IPM_GRAPH","1",1);
    setenv("SPACEPDHCG_TEST_QOCO_NATIVE_REPLAY","1",1);
    setenv("SPACEPDHCG_TEST_QOCO_DEVICE_VALIDATION","1",1);
    setenv("SPACEPDHCG_TEST_GTOC12_DEVICE_ASSEMBLY_VALIDATION","1",1);
    for (int mode=0;mode<3;++mode) {
        setenv("SPACEPDHCG_TEST_QOCO_NATIVE_NUMERIC_REPLAY",mode==1 ? "0" : "1",1);
        setenv("SPACEPDHCG_TEST_QOCO_GPU_CONVERSION_COMPARE",mode==2 ? "1" : "0",1);
        cudaStream_t stream{}; CUDA(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
        double *ds{},*du{}; spacepdhcg_gtoc12_conic_parameters* dp{}; Observed* observed{};
        CUDA(cudaMalloc(&ds,sizeof(states))); CUDA(cudaMalloc(&du,sizeof(controls)));
        CUDA(cudaMalloc(&dp,sizeof(*dp))); CUDA(cudaMalloc(&observed,sizeof(*observed)));
        spacepdhcg_gtoc12_qoco* w{};
        REQUIRE(spacepdhcg_gtoc12_qoco_create(3,0,1,1,.04047160536675379,.030730022327172587,
            times,boundary,weights,1e-9,0,&w)==0);
        const spacepdhcg_gtoc12_conic_parameters valid{.1,.3,13,.3,.05,.02,.001};
        auto params=valid;
        spacepdhcg_gtoc12_qoco_report report{}; Observed result{}; int consumed{};
        const auto solve=[&] {
            CUDA(cudaMemcpyAsync(ds,states,sizeof(states),cudaMemcpyHostToDevice,stream));
            CUDA(cudaMemcpyAsync(du,controls,sizeof(controls),cudaMemcpyHostToDevice,stream));
            CUDA(cudaMemcpyAsync(dp,&params,sizeof(params),cudaMemcpyHostToDevice,stream));
            CUDA(cudaMemsetAsync(observed,0,sizeof(*observed),stream));
            const int code=spacepdhcg_gtoc12_qoco_solve_device_with_consumer(w,ds,du,dp,8,stream,
                &report,consumer,observed,&consumed);
            CUDA(cudaMemcpyAsync(&result,observed,sizeof(result),cudaMemcpyDeviceToHost,stream));
            CUDA(cudaStreamSynchronize(stream));
            return code;
        };
        params.trust_state=-.1;
        REQUIRE(solve()==3 && !consumed && result.calls==0 && !report.qualified);
        params=valid;
        for (int bad=0;bad<4;++bad) {
            for (int prime=0;prime<3;++prime) {
                REQUIRE(solve()==0 && consumed==1 && result.calls==1 && result.qualified==1);
                REQUIRE(report.primal_residual<=1e-9 && report.dual_residual<=1e-9 && report.relative_gap<=1e-9);
            }
            const auto completed=report.solves;
            if (bad==0) params.trust_state=-.1;
            if (bad==1) params.minimum_mass=0;
            if (bad==2) states[6]=-1;
            if (bad==3) controls[4]=std::numeric_limits<double>::quiet_NaN();
            REQUIRE(solve()==3 && !report.qualified && report.iterations==0 && report.solves==completed);
            REQUIRE(std::isinf(report.primal_residual) && std::isinf(report.dual_residual));
            if (mode==0) REQUIRE(consumed==1 && result.calls==1 && result.qualified==0
                && result.status==3 && result.iterations==0);
            else REQUIRE(consumed==0 && result.calls==0);
            params=valid; states[6]=1; controls[4]=0;
        }
        REQUIRE(solve()==0 && result.qualified==1);
        spacepdhcg_gtoc12_qoco_destroy(w);
        CUDA(cudaFree(ds)); CUDA(cudaFree(du)); CUDA(cudaFree(dp)); CUDA(cudaFree(observed));
        CUDA(cudaStreamDestroy(stream));
    }
    std::puts("GTOC12 assembly guard: 4 queued GPU-consumer rejections, 8 fallback rejections, initial rejection and qualified recovery PASS");
}
