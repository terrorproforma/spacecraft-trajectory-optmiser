#include "spacepdhcg/cuda/gtoc12_neighbours_c_api.h"
#include <cuda_runtime_api.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

using Body=spacepdhcg_gtoc12_neighbour_body;
using Query=spacepdhcg_gtoc12_neighbour_query;
static_assert(sizeof(Body)==64 && sizeof(Query)==56);
static void require(bool ok){if(!ok)std::abort();}
static void check(cudaError_t s){require(s==cudaSuccess);}
static void pass(spacepdhcg_cuda_status s){require(s==SPACEPDHCG_CUDA_SUCCESS);}

int main(){
    constexpr int n=137;
    std::vector<Body> bodies(n);std::vector<int> pool(n);
    for(int i=0;i<n;++i){bodies[i]={i+1,64328,2.8*149597870.7,0,0,0,0,0};pool[i]=i;}
    const double tofs[]={90,180,360};
    spacepdhcg_gtoc12_neighbours* w=nullptr;
    pass(spacepdhcg_gtoc12_neighbours_create(bodies.data(),n,pool.data(),n,tofs,3,1.32712440018e11,149597870.7,0,&w));
    Query q{0,8,64328,0.04,0.06,4.5,3.3,1.5};
    std::vector<int64_t> host(n),device(n);int host_count=0,device_count=0;
    Query* dq=nullptr;int64_t* dout=nullptr;int* dc=nullptr;
    check(cudaMalloc(&dq,sizeof(Query)));check(cudaMalloc(&dout,n*sizeof(int64_t)));check(cudaMalloc(&dc,sizeof(int)));
    cudaStream_t stream{};check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    check(cudaMemcpyAsync(dq,&q,sizeof(Query),cudaMemcpyHostToDevice,stream));check(cudaStreamSynchronize(stream));
    check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
    pass(spacepdhcg_gtoc12_neighbours_launch_device(w,dq,dout,dc,{{SPACEPDHCG_DEVICE_CUDA,0},reinterpret_cast<uintptr_t>(stream)}));
    cudaGraph_t graph{};cudaGraphExec_t executable{};
    check(cudaStreamEndCapture(stream,&graph));check(cudaGraphInstantiate(&executable,graph,nullptr,nullptr,0));
    for(int replay=0;replay<4;++replay){
        q.source_index=replay==2 ? n : replay;q.epoch+=100;q.filter_scale=replay%2 ? 0 : 1000;
        pass(spacepdhcg_gtoc12_neighbours_host(w,&q,host.data(),n,&host_count));
        check(cudaMemcpyAsync(dq,&q,sizeof(Query),cudaMemcpyHostToDevice,stream));
        check(cudaGraphLaunch(executable,stream));
        check(cudaMemcpyAsync(device.data(),dout,n*sizeof(int64_t),cudaMemcpyDeviceToHost,stream));
        check(cudaMemcpyAsync(&device_count,dc,sizeof(int),cudaMemcpyDeviceToHost,stream));check(cudaStreamSynchronize(stream));
        require(host_count==device_count);
        if(replay==2){require(device_count==-1);continue;}
        require(device_count==8);
        int expected=1;
        for(int k=0;k<device_count;++k){if(expected==q.source_index+1)++expected;require(device[k]==expected++ && device[k]==host[k]);}
    }
    require(spacepdhcg_gtoc12_neighbours_host(w,&q,host.data(),1,&host_count)==SPACEPDHCG_CUDA_INVALID_ARGUMENT);
    check(cudaGraphExecDestroy(executable));check(cudaGraphDestroy(graph));
    check(cudaFree(dq));check(cudaFree(dout));check(cudaFree(dc));check(cudaStreamDestroy(stream));
    pass(spacepdhcg_gtoc12_neighbours_destroy(&w));
    pool[1]=pool[0];require(spacepdhcg_gtoc12_neighbours_create(bodies.data(),n,pool.data(),n,tofs,3,1.32712440018e11,149597870.7,0,&w)==SPACEPDHCG_CUDA_INVALID_ARGUMENT);
    std::puts("GPU neighbours: graph replay, stable ties, invalid query recovery and guards passed");
}
