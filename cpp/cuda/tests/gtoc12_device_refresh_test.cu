// Exercise the entire gated reference-refresh chain, including a graph replay
// that changes the step count and preserves stale invalid flags when disabled.
#include "../src/gtoc12_scvx.cu"
#include <cstring>
#include <vector>

#define REQUIRE(x) do { if (!(x)) { std::fprintf(stderr,"line %d: %s\n",__LINE__,#x); std::exit(1); } } while (0)
#define CUDA(x) REQUIRE((x)==cudaSuccess)

int main() {
    const int nodes=4,blocks=1;
    const double times[]={0,.01,.02,.03}, fuel[]={.01,.01,.01,0};
    std::vector<double> states(7*nodes),controls(4*nodes);
    for (int i=0;i<nodes;++i) { states[7*i]=2.7; states[7*i+4]=.6; states[7*i+6]=.9; controls[4*i]=.1; controls[4*i+3]=.2; }
    Settings p{}; p.virtual_weight=13; p.conic_tolerance=1e-9;
    Workspace w;
    REQUIRE(spacepdhcg_gtoc12_discretisation_create(nodes-1,0,.02,.003,times,&w.dynamics)==0);
    CUDA(cudaStreamCreateWithFlags(&w.stream,cudaStreamNonBlocking));
    REQUIRE(allocate(&w.states,7*nodes) && allocate(&w.controls,4*nodes) && allocate(&w.fuel,nodes)
        && allocate(&w.partial,blocks) && allocate(&w.metrics,1) && allocate(&w.state,1));
    CUDA(cudaMemcpyAsync(w.states,states.data(),states.size()*8,cudaMemcpyHostToDevice,w.stream));
    CUDA(cudaMemcpyAsync(w.controls,controls.data(),controls.size()*8,cudaMemcpyHostToDevice,w.stream));
    CUDA(cudaMemcpyAsync(w.fuel,fuel,sizeof(fuel),cudaMemcpyHostToDevice,w.stream));
    CUDA(cudaStreamSynchronize(w.stream));
    const double *a,*b,*c,*propagated; const int* invalid;
    REQUIRE(spacepdhcg_gtoc12_discretisation_outputs(w.dynamics,&a,&b,&c,&propagated,&invalid)==0);
    cudaGraph_t graph; cudaGraphExec_t executable;
    CUDA(cudaStreamBeginCapture(w.stream,cudaStreamCaptureModeThreadLocal));
    const int* enabled=&w.state->command.refresh;
    REQUIRE(spacepdhcg_gtoc12_discretisation_launch_controlled_device(w.dynamics,w.states,w.controls,
        &w.state->command.substeps,enabled,0,w.stream)==0);
    reduce_metrics<<<blocks,256,0,w.stream>>>(nodes,7*nodes,w.states,w.controls,nullptr,nullptr,nullptr,
        propagated,invalid,w.fuel,p.conic_tolerance,nodes,w.partial,enabled);
    finish_metrics<<<1,256,0,w.stream>>>(blocks,w.partial,w.metrics,enabled);
    set_reference<<<1,1,0,w.stream>>>(w.state,w.metrics,p,true);
    CUDA(cudaStreamEndCapture(w.stream,&graph));
    CUDA(cudaGraphInstantiate(&executable,graph,nullptr,nullptr,0));
    int count=0;
    for (int substeps:{8,16,0,-1}) for (int refresh:{0,1,-1}) {
        State initial{}; initial.command.substeps=substeps; initial.command.refresh=refresh;
        initial.merit=123; initial.result.max_defect=456;
        Metrics sentinel{91,92,93,94,95,96,97};
        int bad=1;
        CUDA(cudaMemcpyAsync(w.state,&initial,sizeof(initial),cudaMemcpyHostToDevice,w.stream));
        CUDA(cudaMemcpyAsync(w.metrics,&sentinel,sizeof(sentinel),cudaMemcpyHostToDevice,w.stream));
        CUDA(cudaMemcpyAsync(w.partial,&sentinel,sizeof(sentinel),cudaMemcpyHostToDevice,w.stream));
        CUDA(cudaMemcpyAsync(const_cast<int*>(invalid),&bad,sizeof(bad),cudaMemcpyHostToDevice,w.stream));
        CUDA(cudaGraphLaunch(executable,w.stream));
        State got{}; Metrics measured{},partial{};
        CUDA(cudaMemcpyAsync(&got,w.state,sizeof(got),cudaMemcpyDeviceToHost,w.stream));
        CUDA(cudaMemcpyAsync(&measured,w.metrics,sizeof(measured),cudaMemcpyDeviceToHost,w.stream));
        CUDA(cudaMemcpyAsync(&partial,w.partial,sizeof(partial),cudaMemcpyDeviceToHost,w.stream));
        CUDA(cudaMemcpyAsync(&bad,invalid,sizeof(bad),cudaMemcpyDeviceToHost,w.stream));
        CUDA(cudaStreamSynchronize(w.stream));
        if (!refresh) {
            REQUIRE(std::memcmp(&got,&initial,sizeof(got))==0);
            REQUIRE(std::memcmp(&measured,&sentinel,sizeof(measured))==0);
            REQUIRE(std::memcmp(&partial,&sentinel,sizeof(partial))==0 && bad==1);
        } else if (substeps<1) {
            REQUIRE(bad==1 && got.command.error==3 && got.command.done && !got.command.refresh);
        } else {
            REQUIRE(!bad && !got.command.error && !got.command.done && !got.command.refresh);
            // Compare complete reference state to the former fixed-step chain.
            CUDA(cudaMemcpyAsync(w.state,&initial,sizeof(initial),cudaMemcpyHostToDevice,w.stream));
            REQUIRE(spacepdhcg_gtoc12_discretisation_launch_device(w.dynamics,w.states,w.controls,substeps,0,w.stream)==0);
            reduce_metrics<<<blocks,256,0,w.stream>>>(nodes,7*nodes,w.states,w.controls,nullptr,nullptr,nullptr,
                propagated,invalid,w.fuel,p.conic_tolerance,nodes,w.partial);
            finish_metrics<<<1,256,0,w.stream>>>(blocks,w.partial,w.metrics);
            set_reference<<<1,1,0,w.stream>>>(w.state,w.metrics,p);
            State expected{};
            CUDA(cudaMemcpyAsync(&expected,w.state,sizeof(expected),cudaMemcpyDeviceToHost,w.stream));
            CUDA(cudaStreamSynchronize(w.stream));
            REQUIRE(std::memcmp(&expected,&got,sizeof(got))==0);
        }
        ++count;
    }
    CUDA(cudaGraphExecDestroy(executable)); CUDA(cudaGraphDestroy(graph));
    std::printf("PASS: %d captured reference refresh/skip/invalid cases; fixed-chain state parity\n",count);
}
