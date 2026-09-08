// Exercise the actual refinement guards inside a CUDA conditional graph.
// Synthetic corrections isolate backup/acceptance semantics from cuDSS.
#include "cudss_backend.h"
#include <dlfcn.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <limits>
#define CUDA_CHECK(call) do {auto e=(call);if(e!=cudaSuccess){std::fprintf(stderr,"%d: %s\n",__LINE__,cudaGetErrorString(e));std::exit(1);}}while(0)
#define CUDSS_CHECK(call) do {if((call)!=CUDSS_STATUS_SUCCESS)std::exit(1);}while(0)
static CudaLibFuncs g_cuda_funcs{};
static void* g_cudss_handle{};
#include "qoco_ir_runtime.cuh"
__global__ void next_correction(double* norm,double* x,const QocoIrState* state,const double* values){
    *norm=values[state->attempted];*x=state->attempted+1;
}
int main(){
    const double nan=std::numeric_limits<double>::quiet_NaN(),inf=std::numeric_limits<double>::infinity();
    struct Case {double initial,first,second;int accepted,attempted,restore;double solution;};
    const std::vector<Case> cases={
        {10,nan,3,0,1,1,0},{10,inf,3,0,1,1,0},
        {10,4,nan,1,2,1,1},{10,4,inf,1,2,1,1},
        {nan,4,3,0,0,0,0},{inf,4,3,0,0,0,0},
        {10,4,3,2,2,0,2},{10,4,4,1,2,1,1},
    };
    QocoIrState* state;double *norm,*x,*best,*values;
    CUDA_CHECK(cudaMalloc(&state,sizeof(QocoIrState)));
    for(auto p:{&norm,&x,&best})CUDA_CHECK(cudaMalloc(p,sizeof(double)));
    CUDA_CHECK(cudaMalloc(&values,2*sizeof(double)));
    cudaStream_t stream;CUDA_CHECK(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    for(const auto& test:cases){
        double seq[]={test.first,test.second};
        CUDA_CHECK(cudaMemcpy(norm,&test.initial,sizeof(double),cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(values,seq,sizeof(seq),cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(x,0,sizeof(double)));CUDA_CHECK(cudaMemset(best,0,sizeof(double)));
        CUDA_CHECK(cudaDeviceSynchronize());
        cudaGraph_t graph,ended;CUDA_CHECK(cudaGraphCreate(&graph,0));
        cudaGraphConditionalHandle handle;
        CUDA_CHECK(cudaGraphConditionalHandleCreate(&handle,graph,0,cudaGraphCondAssignDefault));
        CUDA_CHECK(cudaStreamBeginCaptureToGraph(stream,graph,nullptr,nullptr,0,cudaStreamCaptureModeThreadLocal));
        qoco_ir_initial<<<1,1,0,stream>>>(state,norm,handle,1.0,2);
        CUDA_CHECK(cudaStreamEndCapture(stream,&ended));
        size_t count=0;CUDA_CHECK(cudaGraphGetNodes(graph,nullptr,&count));
        std::vector<cudaGraphNode_t> deps(count);CUDA_CHECK(cudaGraphGetNodes(graph,deps.data(),&count));
        cudaGraphNodeParams params{};params.type=cudaGraphNodeTypeConditional;
        params.conditional.handle=handle;params.conditional.type=cudaGraphCondTypeWhile;params.conditional.size=1;
        cudaGraphNode_t loop;CUDA_CHECK(cudaGraphAddNode(&loop,graph,deps.data(),count,&params));
        CUDA_CHECK(cudaStreamBeginCaptureToGraph(stream,params.conditional.phGraph_out[0],nullptr,nullptr,0,cudaStreamCaptureModeThreadLocal));
        next_correction<<<1,1,0,stream>>>(norm,x,state,values);
        qoco_ir_decide<<<1,1,0,stream>>>(state,norm,handle,1.0,2);
        qoco_ir_save_or_restore<<<1,1,0,stream>>>(state,x,best,1);
        CUDA_CHECK(cudaStreamEndCapture(stream,&ended));
        cudaGraphExec_t exec;CUDA_CHECK(cudaGraphInstantiate(&exec,graph,0));
        CUDA_CHECK(cudaGraphLaunch(exec,stream));CUDA_CHECK(cudaStreamSynchronize(stream));
        QocoIrState observed;double solution,backup;
        CUDA_CHECK(cudaMemcpy(&observed,state,sizeof(observed),cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&solution,x,sizeof(double),cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&backup,best,sizeof(double),cudaMemcpyDeviceToHost));
        if(observed.accepted!=test.accepted||observed.attempted!=test.attempted||observed.restore!=test.restore||solution!=test.solution||backup!=test.solution)return 2;
        CUDA_CHECK(cudaGraphExecDestroy(exec));CUDA_CHECK(cudaGraphDestroy(graph));
    }
    CUDA_CHECK(cudaStreamDestroy(stream));CUDA_CHECK(cudaFree(state));
    for(auto p:{norm,x,best,values})CUDA_CHECK(cudaFree(p));
    std::puts("8 actual conditional-graph nonfinite/backup cases passed");
}
