// Re-enter the complete IPM from a GPU outer WHILE. Check each result against
// independently solved analytic QPs, including skipped/failed inner solves.
#define main qoco_synchronous_probe_main
#include "qoco_ipm_cache_probe.cpp"
#undef main
#include <cuda_runtime.h>
#include <cstring>
#include <thread>
#include <vector>
#include "qoco_gpu_replay.h"
#define CUDA(x) REQUIRE((x)==cudaSuccess)
struct Snapshot { QocoGpuCompletion result; double x[2]; };
__global__ void prepare_update(double* update,const int* index,double k,double kinv) {
    for (int i=0;i<9;++i) update[i]=0;
    update[0]=k; update[1]=kinv;
    update[8]=*index%4==2; // invalid then valid must restart every inner condition
}
__global__ void consume_outer(QocoGpuOutput out,Snapshot* result,int* index,
    cudaGraphConditionalHandle loop) {
    auto& row=result[*index];
    row.result=*out.completion;
    row.x[0]=out.x[0]; row.x[1]=out.x[1];
    cudaGraphSetConditional(loop,++*index<8);
}
static std::vector<cudaGraphNode_t> end_capture(cudaGraph_t graph) {
    cudaStreamCaptureStatus status;
    const cudaGraphNode_t* deps{};size_t count{};cudaGraph_t found{};
    CUDA(cudaStreamGetCaptureInfo(cudaStreamPerThread,&status,nullptr,&found,&deps,&count));
    REQUIRE(found==graph);
    std::vector<cudaGraphNode_t> result;
    if(count) result.assign(deps,deps+count);
    CUDA(cudaStreamEndCapture(cudaStreamPerThread,&found));REQUIRE(found==graph);
    return result;
}
#ifndef QOCO_OUTER_GRAPH_NO_MAIN
int main() {
    REQUIRE(qoco_gpu_begin_reduction_scope()==0);
    setenv("QOCO_IPM_PROBE_RUIZ","1",1);
    cudaStream_t stream;CUDA(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    Snapshot* snapshots;int* index;double* update;
    CUDA(cudaMalloc(&snapshots,8*sizeof(Snapshot)));CUDA(cudaMalloc(&index,sizeof(int)));
    CUDA(cudaMalloc(&update,9*sizeof(double)));
    int solved=0,rejected=0;
    {
        Fixture fixture;
        for(int repeat:{0,1,2,8,14,15}) {
            REQUIRE(qoco_gpu_primal_start(fixture.solver,0)==0);
            fixture.run(0,repeat);
            const auto expected=*fixture.solver->sol;
            const double expected_x[2]{expected.x[0],expected.x[1]};
            cudaGraph_t root;CUDA(cudaGraphCreate(&root,0));
            cudaGraphConditionalHandle loop;
            CUDA(cudaGraphConditionalHandleCreate(&loop,root,1,cudaGraphCondAssignDefault));
            CUDA(cudaStreamBeginCaptureToGraph(cudaStreamPerThread,root,nullptr,nullptr,0,cudaStreamCaptureModeThreadLocal));
            CUDA(cudaMemsetAsync(index,0,sizeof(int),cudaStreamPerThread));
            auto deps=end_capture(root);
            cudaGraphNodeParams params{};params.type=cudaGraphNodeTypeConditional;
            params.conditional.handle=loop;params.conditional.type=cudaGraphCondTypeWhile;params.conditional.size=1;
            cudaGraphNode_t node;CUDA(cudaGraphAddNode(&node,root,deps.data(),deps.size(),&params));
            auto body=params.conditional.phGraph_out[0];
            CUDA(cudaStreamBeginCaptureToGraph(cudaStreamPerThread,body,nullptr,nullptr,0,cudaStreamCaptureModeThreadLocal));
            prepare_update<<<1,1>>>(update,index,fixture.solver->work->scaling->k,fixture.solver->work->scaling->kinv);
            deps=end_capture(body);
            QocoGpuOutput output{};cudaGraphNode_t completed{};
            const double static_g=fixture.solver->settings->kkt_static_reg_G;
            fixture.solver->settings->kkt_static_reg_G*=2;
            REQUIRE(qoco_gpu_ipm_emit_graph(fixture.solver,body,deps.data(),deps.size(),update,&output,&completed)==2);
            REQUIRE(!output.completion && !completed);
            fixture.solver->settings->kkt_static_reg_G=static_g;
            int thread_status{};
            std::thread worker([&]{thread_status=qoco_gpu_ipm_emit_graph(fixture.solver,body,deps.data(),deps.size(),update,&output,&completed);});
            worker.join();REQUIRE(thread_status==2 && !completed && !output.completion);
            REQUIRE(qoco_gpu_ipm_emit_graph(fixture.solver,body,deps.data(),deps.size(),update,&output,&completed)==0);
            REQUIRE(completed && output.completion && output.n==2);
            CUDA(cudaStreamBeginCaptureToGraph(cudaStreamPerThread,body,&completed,nullptr,1,cudaStreamCaptureModeThreadLocal));
            consume_outer<<<1,1>>>(output,snapshots,index,loop);end_capture(body);
            cudaGraphExec_t executable;CUDA(cudaGraphInstantiate(&executable,root,0));
            for(int launch=0;launch<2;++launch) {
                CUDA(cudaGraphLaunch(executable,stream));CUDA(cudaStreamSynchronize(stream));
                Snapshot got[8];int count{};
                CUDA(cudaMemcpy(got,snapshots,sizeof(got),cudaMemcpyDeviceToHost));
                CUDA(cudaMemcpy(&count,index,sizeof(count),cudaMemcpyDeviceToHost));REQUIRE(count==8);
                for(int i=0;i<8;++i) {
                    const auto& actual=got[i].result;
                    if(i%4==2) {
                        REQUIRE(actual.status==QOCO_NUMERICAL_ERROR && actual.iterations==0 && std::isnan(actual.objective));++rejected;
                    } else {
                        REQUIRE(actual.status==expected.status && actual.iterations==expected.iters);
                        REQUIRE(actual.ir_iterations==expected.ir_iters && actual.primal_residual==expected.pres);
                        REQUIRE(actual.dual_residual==expected.dres && actual.gap==expected.gap);
                        REQUIRE(actual.objective==expected.obj && std::memcmp(got[i].x,expected_x,sizeof(expected_x))==0);++solved;
                    }
                }
            }
            CUDA(cudaGraphExecDestroy(executable));CUDA(cudaGraphDestroy(root));
        }
    }
    CUDA(cudaFree(snapshots));CUDA(cudaFree(index));CUDA(cudaFree(update));CUDA(cudaStreamDestroy(stream));
    qoco_gpu_end_reduction_scope();
    std::printf("PASS: %d exact nested IPM results, %d guarded invalid updates; repeated outer loops, static/thread guards\n",solved,rejected);
}
#endif
