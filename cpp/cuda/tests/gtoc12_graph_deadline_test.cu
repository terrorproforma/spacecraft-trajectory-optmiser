#include "../src/gtoc12_scvx.cu"
#include <vector>
#define REQUIRE(x) do {if(!(x)){std::fprintf(stderr,"line %d: %s\n",__LINE__,#x);std::exit(1);}}while(0)
#define CUDA(x) REQUIRE((x)==cudaSuccess)
__global__ void simulated_attempt(State* s,unsigned long long duration) {
    const auto begin=graph_nanoseconds();while(graph_nanoseconds()-begin<duration) {}
    ++s->result.iterations;
}
int main() {
    cudaStream_t stream;CUDA(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    State* state;QocoGraphProgress* progress;CUDA(cudaMalloc(&state,sizeof(State)));CUDA(cudaMalloc(&progress,sizeof(QocoGraphProgress)));
    for(int test=0;test<5;++test) {
        State initial{};QocoGraphProgress ledger{};
        if(test==3) initial.command.done=1;
        if(test==4) ledger.last_validation=16;
        CUDA(cudaMemcpyAsync(state,&initial,sizeof(initial),cudaMemcpyHostToDevice,stream));
        CUDA(cudaMemcpyAsync(progress,&ledger,sizeof(ledger),cudaMemcpyHostToDevice,stream));
        cudaGraph_t graph;CUDA(cudaGraphCreate(&graph,0));cudaGraphConditionalHandle loop;
        CUDA(cudaGraphConditionalHandleCreate(&loop,graph,0,cudaGraphCondAssignDefault));
        cudaGraphNode_t gate;
        CUDA(spacepdhcg_graph_append(stream,graph,nullptr,0,[&]{
            graph_gate<<<1,1,0,stream>>>(state,2,progress,loop);return cudaGetLastError();},&gate));
        cudaGraphNodeParams parameters{};parameters.type=cudaGraphNodeTypeConditional;
        parameters.conditional.handle=loop;parameters.conditional.type=cudaGraphCondTypeWhile;parameters.conditional.size=1;
        cudaGraphNode_t outer;CUDA(cudaGraphAddNode(&outer,graph,&gate,1,&parameters));auto body=parameters.conditional.phGraph_out[0];
        CUDA(spacepdhcg_graph_append(stream,body,nullptr,0,[&]{
            simulated_attempt<<<1,1,0,stream>>>(state,test==1 ? 3000000ULL : 0ULL);
            graph_gate<<<1,1,0,stream>>>(state,2,progress,loop);return cudaGetLastError();},&gate));
        cudaGraphExec_t executable;CUDA(cudaGraphInstantiate(&executable,graph,0));
        cudaEvent_t start,end;CUDA(cudaEventCreate(&start));CUDA(cudaEventCreate(&end));
        CUDA(cudaEventRecord(start,stream));start_graph_clock<<<1,1,0,stream>>>(state,test==0 || test==3 ? 0ULL : test==1 ? 1000000ULL : 1000000000ULL);
        CUDA(cudaGraphLaunch(executable,stream));CUDA(cudaEventRecord(end,stream));CUDA(cudaStreamSynchronize(stream));
        State got{};CUDA(cudaMemcpy(&got,state,sizeof(got),cudaMemcpyDeviceToHost));float ms;CUDA(cudaEventElapsedTime(&ms,start,end));
        REQUIRE(got.result.iterations==(test==1 ? 1 : test==2 ? 2 : 0));
        REQUIRE(got.graph_timeout==(test<2));REQUIRE(got.command.error==(test==4 ? 3 : 0));
        if(test==1) REQUIRE(ms>=2.5f && ms<100.0f);
        std::printf("DEADLINE case=%d attempts=%d timeout=%d error=%d elapsed_ms=%.4f\n",test,got.result.iterations,got.graph_timeout,got.command.error,ms);
        CUDA(cudaEventDestroy(start));CUDA(cudaEventDestroy(end));CUDA(cudaGraphExecDestroy(executable));CUDA(cudaGraphDestroy(graph));
    }
    CUDA(cudaFree(state));CUDA(cudaFree(progress));CUDA(cudaStreamDestroy(stream));
    std::puts("PASS: GPU zero budget, in-loop deadline, iteration cap, completed state and producer failure termination");
}
