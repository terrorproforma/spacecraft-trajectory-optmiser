#include "../internal/gtoc12_qoco_qualification.cuh"
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <vector>

#define REQUIRE(x) do { if (!(x)) { std::fprintf(stderr,"line %d: %s\n",__LINE__,#x); std::exit(1); } } while (0)
#define CUDA(x) REQUIRE((x)==cudaSuccess)

// A graph consumer must observe the freshly computed gate on each replay.
__global__ void consume(const spacepdhcg_gtoc12_qoco_report* report,int* accepted) {
    *accepted=report->qualified ? report->iterations : -1;
}
int main() {
    cudaStream_t stream;
    CUDA(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    QocoReplayStatus* status; QocoAuditResult* audit; double* objective;
    spacepdhcg_gtoc12_qoco_report* report; int* accepted;
    CUDA(cudaMalloc(&status,sizeof(*status))); CUDA(cudaMalloc(&audit,sizeof(*audit)));
    CUDA(cudaMalloc(&objective,4*sizeof(double))); CUDA(cudaMalloc(&report,sizeof(*report)));
    CUDA(cudaMalloc(&accepted,sizeof(int)));
    constexpr double tolerance=1e-9;
    cudaGraph_t graph; cudaGraphExec_t executable;
    CUDA(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
    gtoc12_qoco::qualify<<<1,1,0,stream>>>(status,audit,objective,tolerance,report);
    consume<<<1,1,0,stream>>>(report,accepted);
    CUDA(cudaStreamEndCapture(stream,&graph));
    CUDA(cudaGraphInstantiate(&executable,graph,0));
    int cases=0;
    const double nan=std::numeric_limits<double>::quiet_NaN();
    const double inf=std::numeric_limits<double>::infinity();
    for (int raw_status:{-1,0,1,2,3,4,6}) for (int field=0;field<5;++field)
    for (double value:{0.0,tolerance,std::nextafter(tolerance,inf),nan,inf,-inf}) {
        QocoReplayStatus s{raw_status,37};
        QocoAuditResult a{0,0,3,4,0,0};
        double obj[4]{0,0,0,0};
        if (field==0) a.primal=value;
        if (field==1) a.dual=value;
        if (field==2) obj[0]=value;
        if (field==3) obj[1]=value;
        if (field==4) obj[3]=value;
        // Independent expectations from the case construction, not a duplicate
        // gate expression. Finite objective magnitudes have no tolerance bound.
        const bool expected=(raw_status==1 || raw_status==2) && std::isfinite(value)
            && ((field==2 || field==3) || value<=tolerance);
        CUDA(cudaMemcpyAsync(status,&s,sizeof(s),cudaMemcpyHostToDevice,stream));
        CUDA(cudaMemcpyAsync(audit,&a,sizeof(a),cudaMemcpyHostToDevice,stream));
        CUDA(cudaMemcpyAsync(objective,obj,sizeof(obj),cudaMemcpyHostToDevice,stream));
        CUDA(cudaGraphLaunch(executable,stream));
        spacepdhcg_gtoc12_qoco_report got{}; int consumed=0;
        CUDA(cudaMemcpyAsync(&got,report,sizeof(got),cudaMemcpyDeviceToHost,stream));
        CUDA(cudaMemcpyAsync(&consumed,accepted,sizeof(int),cudaMemcpyDeviceToHost,stream));
        CUDA(cudaStreamSynchronize(stream));
        REQUIRE(got.qualified==expected && consumed==(expected ? 37 : -1));
        REQUIRE(got.qoco_status==raw_status && got.iterations==37);
        REQUIRE(got.requested_tolerance==tolerance);
        REQUIRE(got.absolute_primal_residual==3 && got.absolute_dual_residual==4);
        REQUIRE(got.setup_seconds==0 && got.adapter_d2h_bytes==0);
        ++cases;
    }
    CUDA(cudaGraphExecDestroy(executable)); CUDA(cudaGraphDestroy(graph));
    CUDA(cudaFree(status)); CUDA(cudaFree(audit)); CUDA(cudaFree(objective));
    CUDA(cudaFree(report)); CUDA(cudaFree(accepted)); CUDA(cudaStreamDestroy(stream));
    std::printf("PASS: %d device qualification boundary/nonfinite/status graph-consumer cases\n",cases);
}
