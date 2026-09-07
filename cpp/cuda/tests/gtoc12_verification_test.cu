#include "spacepdhcg/cuda/gtoc12_verification_c_api.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>

void require(bool ok, const char* message) {
    if(!ok) { std::fprintf(stderr,"%s\n",message); std::exit(1); }
}
void check(cudaError_t status) { require(status==cudaSuccess,cudaGetErrorString(status)); }
int main() {
    static_assert(sizeof(spacepdhcg_verify_leg)==72);
    static_assert(sizeof(spacepdhcg_verify_arc)==8);
    static_assert(sizeof(spacepdhcg_verify_sample)==32);
    static_assert(sizeof(spacepdhcg_verify_result)==80);
    constexpr int n=37;
    std::vector<spacepdhcg_verify_leg> legs(n);
    for(auto& leg:legs) {
        leg.initial[0]=1.49597870691e8;
        leg.initial[4]=sqrt(1.32712440018e11/leg.initial[0]);
        leg.initial[6]=2500.0; leg.duration_s=86400.0;
    }
    spacepdhcg_verify_leg* dlegs=nullptr;
    spacepdhcg_verify_result* dout=nullptr;
    cudaStream_t stream; check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    check(cudaMalloc(&dlegs,n*sizeof(*dlegs))); check(cudaMalloc(&dout,n*sizeof(*dout)));
    check(cudaMemcpyAsync(dlegs,legs.data(),n*sizeof(*dlegs),cudaMemcpyHostToDevice,stream));
    check(cudaStreamSynchronize(stream));
    cudaGraph_t graph; cudaGraphExec_t executable;
    check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
    check(spacepdhcg_gtoc12_verify_launch(dlegs,n,nullptr,0,nullptr,0,1000,dout,stream));
    check(cudaStreamEndCapture(stream,&graph)); check(cudaGraphInstantiate(&executable,graph,nullptr,nullptr,0));
    std::vector<spacepdhcg_verify_result> out(n);
    for(int iteration=0;iteration<3;++iteration) {
        legs[3].initial[6]=iteration==1 ? -1 : 2500;
        check(cudaMemcpyAsync(dlegs,legs.data(),n*sizeof(*dlegs),cudaMemcpyHostToDevice,stream));
        check(cudaGraphLaunch(executable,stream));
        check(cudaMemcpyAsync(out.data(),dout,n*sizeof(*dout),cudaMemcpyDeviceToHost,stream));
        check(cudaStreamSynchronize(stream));
        for(int i=0;i<n;++i) {
            if(iteration==1 && i==3) require(out[i].status==1 && std::isnan(out[i].final_state[0]),"graph failed-input isolation");
            else require(out[i].status==0 && out[i].final_state[6]==2500 && out[i].final_state[1]>0,"graph propagation");
        }
    }
    spacepdhcg_verify_workspace* workspace=nullptr;
    check(spacepdhcg_gtoc12_verify_create(n,0,0,&workspace));
    cudaError_t foreign=cudaSuccess;
    std::thread other([&] { foreign=spacepdhcg_gtoc12_verify_host(workspace,legs.data(),n,nullptr,0,nullptr,0,1000,out.data()); });
    other.join(); require(foreign==cudaErrorInvalidResourceHandle,"thread ownership");
    check(spacepdhcg_gtoc12_verify_host(workspace,legs.data(),n,nullptr,0,nullptr,0,1000,out.data()));
    check(spacepdhcg_gtoc12_verify_destroy(&workspace)); require(!workspace,"destroy clears pointer");
    check(spacepdhcg_gtoc12_verify_destroy(&workspace));
    check(cudaGraphExecDestroy(executable)); check(cudaGraphDestroy(graph));
    check(cudaFree(dout)); check(cudaFree(dlegs)); check(cudaStreamDestroy(stream));
    std::puts("PASS: device graph replay, invalid-leg isolation, retained host workspace and thread ownership");
}
