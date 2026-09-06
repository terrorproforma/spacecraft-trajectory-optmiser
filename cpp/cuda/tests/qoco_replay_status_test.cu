#include "../internal/native_qoco_gpu.h"
#include <climits>
#include <cstdio>
#include <cstdlib>
#define REQUIRE(x) do { if (!(x)) { std::fprintf(stderr,"line %d: %s\n",__LINE__,#x); std::exit(1); } } while (0)
#define CUDA(x) REQUIRE((x)==cudaSuccess)
int main() {
    const int offsets[]{0,0}; const double cost=0;
    QocoAuditCsc p{1,1,0,offsets,nullptr,nullptr}, empty{0,1,0,offsets,nullptr,nullptr};
    QocoAuditCsc map{0,0,0,offsets,nullptr,nullptr};
    QocoAuditInput input{p,empty,empty,map,&cost,nullptr,nullptr,0,0,nullptr};
    cudaStream_t stream; CUDA(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    QocoGpuAudit* audit{}; CUDA(qoco_gpu_audit_create(input,false,stream,&audit));
    int* header; CUDA(cudaMalloc(&header,3*sizeof(int)));
    const auto transfers=qoco_gpu_audit_transfers(audit);
    const auto memory=qoco_gpu_audit_memory(audit);
    const QocoReplayStatus* output{};
    cudaGraph_t graph; cudaGraphExec_t executable;
    CUDA(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
    CUDA(qoco_gpu_audit_replay_status(audit,header,stream,&output));
    CUDA(cudaStreamEndCapture(stream,&graph));
    CUDA(cudaGraphInstantiate(&executable,graph,nullptr,nullptr,0));
    int cases=0;
    for (int abi : {-1,0,1,2}) for (int status : {1,2,4,6}) for (int iterations : {0,1,200,INT_MAX}) {
        const int values[]{abi,status,iterations};
        CUDA(cudaMemcpyAsync(header,values,sizeof(values),cudaMemcpyHostToDevice,stream));
        CUDA(cudaGraphLaunch(executable,stream));
        QocoReplayStatus got{};
        CUDA(cudaMemcpyAsync(&got,output,sizeof(got),cudaMemcpyDeviceToHost,stream));
        CUDA(cudaStreamSynchronize(stream));
        REQUIRE(got.status==(abi==1 ? status:-1) && got.iterations==(abi==1 ? iterations:0));
        ++cases;
    }
    const auto after=qoco_gpu_audit_transfers(audit);
    REQUIRE(after.d2h_count==transfers.d2h_count && after.d2h_bytes==transfers.d2h_bytes);
    REQUIRE(qoco_gpu_audit_memory(audit).allocations==memory.allocations);
    REQUIRE(qoco_gpu_audit_replay_status(nullptr,header,stream,&output)==cudaErrorInvalidValue && !output);
    CUDA(cudaGraphExecDestroy(executable)); CUDA(cudaGraphDestroy(graph)); CUDA(cudaFree(header));
    qoco_gpu_audit_destroy(audit); CUDA(cudaStreamDestroy(stream));
    std::printf("PASS: %d captured completion ABI/status cases, no internal downloads or allocations\n",cases);
}
