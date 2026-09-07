#include "../internal/gtoc12_qoco_graph.h"
#include "../internal/graph_append.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>

#define REQUIRE(x) do { if(!(x)){std::fprintf(stderr,"line %d: %s\n",__LINE__,#x);std::exit(1);} }while(0)
#define CUDA(x) REQUIRE((x)==cudaSuccess)
constexpr int cases=6,nodes=4;
__global__ void set_index(int* index,int i) { *index=i; }
__global__ void select_case(const double* initial,double* states,double* controls,
    spacepdhcg_gtoc12_conic_parameters* parameters,int* substeps,const int* index) {
    const int i=*index;
    for(int j=threadIdx.x;j<28;j+=blockDim.x) {
        states[j]=initial[j];
        if(j==7 || j==14) states[j]+=i*1e-5;
    }
    for(int j=threadIdx.x;j<16;j+=blockDim.x)
        controls[j]=j%4==0 ? i*.001 : j%4==3 ? i*.002 : 0;
    if(threadIdx.x==0) {
        *parameters={.1+i*.005,.3+i*.01,13.0+i,.3,.05,.02,.001*(1+i)};
        *substeps=i%2 ? 16 : 8;
        if(i==2) *substeps=0;
        if(i==4) parameters->trust_state=-.1;
    }
}
__global__ void snapshot_primal(const double* primal,double* snapshots,const int* index,int n) {
    for(int i=threadIdx.x;i<n;i+=blockDim.x) snapshots[*index*n+i]=primal[i];
}
__global__ void snapshot_report(const spacepdhcg_gtoc12_qoco_report* report,
    spacepdhcg_gtoc12_qoco_report* snapshots,int* index,cudaGraphConditionalHandle loop) {
    snapshots[*index]=*report;
    ++*index;
    if(loop) cudaGraphSetConditional(loop,*index<cases);
}
struct Consumer {
    spacepdhcg_gtoc12_qoco_report* reports;
    double* primal;
    int *index,variables;
    cudaGraphConditionalHandle loop;
};
int consume(void* opaque,const spacepdhcg_gtoc12_qoco_report* report,const double* primal,void* ptr) {
    const auto c=*static_cast<Consumer*>(opaque);auto stream=static_cast<cudaStream_t>(ptr);
    snapshot_primal<<<1,256,0,stream>>>(primal,c.primal,c.index,c.variables);
    snapshot_report<<<1,1,0,stream>>>(report,c.reports,c.index,c.loop);
    return cudaGetLastError()==cudaSuccess ? 0 : 2;
}
int main() {
    if(!std::getenv("SPACEPDHCG_QOCO_LIBRARY")) { std::puts("SKIP: prepared QOCO133 required");return 0; }
    for(auto flag:{"SPACEPDHCG_TEST_QOCO_IPM_GRAPH","SPACEPDHCG_TEST_QOCO_NATIVE_REPLAY",
        "SPACEPDHCG_TEST_QOCO_NATIVE_NUMERIC_REPLAY","SPACEPDHCG_TEST_QOCO_DEVICE_VALIDATION",
        "SPACEPDHCG_TEST_GTOC12_DEVICE_ASSEMBLY_VALIDATION"}) setenv(flag,"1",1);
    int passed=0,rejected=0;
    for(bool origin:{false,true}) for(int ruiz:{0,5}) {
        setenv("SPACEPDHCG_TEST_GTOC12_STATE_ORIGIN",origin ? "1" : "0",1);
        double times[4]{0,.003,.006,.009},initial[28]{},boundary[12]{},fuel[4]{.003,.003,.003,0};
        const double omega=std::pow(2.7,-1.5);
        for(int k=0;k<nodes;++k) {
            initial[7*k]=2.7*std::cos(omega*times[k]);initial[7*k+1]=2.7*std::sin(omega*times[k]);
            initial[7*k+3]=-2.7*omega*std::sin(omega*times[k]);initial[7*k+4]=2.7*omega*std::cos(omega*times[k]);initial[7*k+6]=1;
        }
        for(int i=0;i<6;++i) { boundary[i]=initial[i];boundary[6+i]=initial[21+i]; }
        spacepdhcg_gtoc12_qoco *reference{},*queued{};
        for(auto target:{&reference,&queued}) REQUIRE(spacepdhcg_gtoc12_qoco_create(3,0,1,1,
            .04047160536675379,.030730022327172587,times,boundary,fuel,1e-9,ruiz,target)==0);
        spacepdhcg_gtoc12_conic_dimensions dimensions{};
        REQUIRE(spacepdhcg_gtoc12_qoco_get_dimensions(queued,&dimensions)==0);
        cudaStream_t stream;CUDA(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
        double *initial_device,*states,*controls,*primal;
        int *index,*substeps;spacepdhcg_gtoc12_conic_parameters* parameters;
        spacepdhcg_gtoc12_qoco_report* reports;
        CUDA(cudaMalloc(&initial_device,sizeof(initial)));CUDA(cudaMalloc(&states,sizeof(initial)));
        CUDA(cudaMalloc(&controls,16*sizeof(double)));CUDA(cudaMalloc(&index,sizeof(int)));CUDA(cudaMalloc(&substeps,sizeof(int)));
        CUDA(cudaMalloc(&parameters,sizeof(*parameters)));CUDA(cudaMalloc(&reports,cases*sizeof(*reports)));
        CUDA(cudaMalloc(&primal,cases*dimensions.variables*sizeof(double)));
        CUDA(cudaMemcpyAsync(initial_device,initial,sizeof(initial),cudaMemcpyHostToDevice,stream));
        Consumer callback{reports,primal,index,dimensions.variables,0};
        const auto select=[&](int i) {
            set_index<<<1,1,0,stream>>>(index,i);
            select_case<<<1,32,0,stream>>>(initial_device,states,controls,parameters,substeps,index);
        };
        const QocoGraphProgress* progress{};
        REQUIRE(spacepdhcg_gtoc12_qoco_begin_graph(queued,stream,&progress)!=0 && !progress);
        for(auto w:{reference,queued}) for(int prime=0;prime<3;++prime) {
            select(0);spacepdhcg_gtoc12_qoco_report report{};int consumed=0;
            REQUIRE(spacepdhcg_gtoc12_qoco_solve_controlled_device_with_consumer(w,states,controls,parameters,
                substeps,stream,&report,consume,&callback,&consumed)==0 && consumed && report.qualified);
        }
        REQUIRE(spacepdhcg_gtoc12_qoco_begin_graph(queued,stream,&progress)==0 && progress);
        REQUIRE(!spacepdhcg_gtoc12_qoco_can_enqueue(queued));
        const QocoGraphProgress* extra{};
        REQUIRE(spacepdhcg_gtoc12_qoco_begin_graph(queued,stream,&extra)!=0 && !extra);
        int wrong=0;std::thread owner([&]{wrong=spacepdhcg_gtoc12_qoco_end_graph(queued,stream);});owner.join();REQUIRE(wrong!=0);
        spacepdhcg_gtoc12_qoco_report forbidden{};int consumed=0;
        REQUIRE(spacepdhcg_gtoc12_qoco_solve_controlled_device_with_consumer(queued,states,controls,parameters,
            substeps,stream,&forbidden,consume,&callback,&consumed)!=0 && !consumed);

        cudaGraph_t root;CUDA(cudaGraphCreate(&root,0));cudaGraphConditionalHandle loop;
        CUDA(cudaGraphConditionalHandleCreate(&loop,root,1,cudaGraphCondAssignDefault));
        cudaGraphNode_t prepared{};
        CUDA(spacepdhcg_graph_append(stream,root,nullptr,0,[&]{set_index<<<1,1,0,stream>>>(index,0);return cudaGetLastError();},&prepared));
        cudaGraphNodeParams params{};params.type=cudaGraphNodeTypeConditional;
        params.conditional.handle=loop;params.conditional.type=cudaGraphCondTypeWhile;params.conditional.size=1;
        cudaGraphNode_t outer;CUDA(cudaGraphAddNode(&outer,root,&prepared,1,&params));auto body=params.conditional.phGraph_out[0];
        CUDA(spacepdhcg_graph_append(stream,body,nullptr,0,[&]{
            select_case<<<1,32,0,stream>>>(initial_device,states,controls,parameters,substeps,index);return cudaGetLastError();},&prepared));
        callback.loop=loop;cudaGraphNode_t complete{};
        REQUIRE(spacepdhcg_gtoc12_qoco_emit_graph(queued,body,&prepared,1,stream,states,controls,parameters,
            substeps,consume,&callback,&complete)==0 && complete);
        callback.loop=0; // Graph retained kernel arguments, never this host context.
        if(const auto* prefix=std::getenv("SPACEPDHCG_TEST_GTOC12_OUTER_GRAPH_DOT")) {
            char path[4096];REQUIRE(std::snprintf(path,sizeof(path),"%s-origin%d-ruiz%d.dot",prefix,int(origin),ruiz)<int(sizeof(path)));
            CUDA(cudaGraphDebugDotPrint(root,path,cudaGraphDebugDotFlagsVerbose));
        }
        cudaGraphExec_t executable;CUDA(cudaGraphInstantiate(&executable,root,0));
        const bool abandon=origin && ruiz==5;
        if(abandon) spacepdhcg_gtoc12_qoco_destroy(queued); // Lease retains all producer/consumer buffers.
        for(int launch=0;launch<2;++launch) {
            std::vector<double> expected(cases*dimensions.variables);
            spacepdhcg_gtoc12_qoco_report expected_reports[cases]{};
            for(int i=0;i<cases;++i) {
                select(i);spacepdhcg_gtoc12_qoco_report report{};int consumed=0;
                const int code=spacepdhcg_gtoc12_qoco_solve_controlled_device_with_consumer(reference,states,controls,parameters,
                    substeps,stream,&report,consume,&callback,&consumed);
                if(i==2 || i==4) REQUIRE(code==3 && !report.qualified && report.iterations==0);
                else REQUIRE(code==0 && report.qualified && consumed);
                expected_reports[i]=report;
                if(code==0) CUDA(cudaMemcpyAsync(expected.data()+i*dimensions.variables,primal+i*dimensions.variables,
                    dimensions.variables*sizeof(double),cudaMemcpyDeviceToHost,stream));
                CUDA(cudaStreamSynchronize(stream));
            }
            CUDA(cudaGraphLaunch(executable,stream));CUDA(cudaStreamSynchronize(stream));
            std::vector<double> got(expected.size());spacepdhcg_gtoc12_qoco_report got_reports[cases];QocoGraphProgress ledger{};
            CUDA(cudaMemcpy(got.data(),primal,got.size()*sizeof(double),cudaMemcpyDeviceToHost));
            CUDA(cudaMemcpy(got_reports,reports,sizeof(got_reports),cudaMemcpyDeviceToHost));
            CUDA(cudaMemcpy(&ledger,progress,sizeof(ledger),cudaMemcpyDeviceToHost));
            REQUIRE(ledger.attempts==std::uint64_t((launch+1)*cases) && ledger.solver_runs==std::uint64_t((launch+1)*4));
            for(int i=0;i<cases;++i) {
                const auto& r=got_reports[i];const auto& ref=expected_reports[i];
                if(i==2 || i==4) { REQUIRE(!r.qualified && r.qoco_status==3 && r.iterations==0);++rejected;continue; }
                REQUIRE(r.qualified && r.primal_residual<=1e-9 && r.dual_residual<=1e-9 && r.relative_gap<=1e-9);
                double max_difference=0;
                for(int j=0;j<dimensions.variables;++j) max_difference=std::max(max_difference,std::abs(got[i*dimensions.variables+j]-expected[i*dimensions.variables+j]));
                std::printf("GRAPH_TRAJECTORY origin=%d ruiz=%d launch=%d case=%d ipm=%d/%d dx=%.3g objective=%.17g gap=%.3g\n",
                    int(origin),ruiz,launch,i,r.iterations,ref.iterations,max_difference,r.primal_objective,r.relative_gap);
                REQUIRE(max_difference<1e-7 && std::abs(r.primal_objective-ref.primal_objective)<1e-9);++passed;
            }
        }
        CUDA(cudaGraphExecDestroy(executable));CUDA(cudaGraphDestroy(root));
        REQUIRE(spacepdhcg_gtoc12_qoco_end_graph(queued,stream)==0);
        if(abandon) queued=nullptr;
        else {
            REQUIRE(spacepdhcg_gtoc12_qoco_end_graph(queued,stream)!=0);
            select(0);spacepdhcg_gtoc12_qoco_report report{};int consumed=0;
            REQUIRE(spacepdhcg_gtoc12_qoco_solve_controlled_device_with_consumer(queued,states,controls,parameters,
                substeps,stream,&report,consume,&callback,&consumed)==0 && report.workspace_creations==2);
        }
        spacepdhcg_gtoc12_qoco_destroy(queued);spacepdhcg_gtoc12_qoco_destroy(reference);
        CUDA(cudaFree(initial_device));CUDA(cudaFree(states));CUDA(cudaFree(controls));CUDA(cudaFree(index));CUDA(cudaFree(substeps));
        CUDA(cudaFree(parameters));CUDA(cudaFree(reports));CUDA(cudaFree(primal));CUDA(cudaStreamDestroy(stream));
    }
    std::printf("PASS: %d qualified changing trajectory QPs, %d rejected producer inputs; GPU outer dispatch, original-coordinate audits, lease ownership and post-graph recovery\n",passed,rejected);
}
