#include "gtoc12_fleet.cuh"
extern "C" int profile_fleet_stages(void* handle,const uint8_t* warm,int rounds,uint64_t cap,
                                    double* milliseconds,uint8_t* selected,gtoc12_fleet::Report* report) {
    using namespace gtoc12_fleet;
    if(!handle||rounds<0||rounds>100)return 1;
    auto& w=*static_cast<Workspace*>(handle);auto p=w.problem(100);auto s=w.m.stream;
    std::vector<cudaEvent_t> events(size_t(5+2*rounds));
    auto release=[&](){for(auto event:events)if(event)cudaEventDestroy(event);};
    for(auto& event:events)if(cudaEventCreate(&event)!=cudaSuccess){release();return 2;}
    cudaMemcpyAsync(w.dw,warm,w.n,cudaMemcpyHostToDevice,s);
    cudaEventRecord(events[0],s);
    seed<<<3,128,0,s>>>(p,w.order,w.dw,w.masks,w.rows);
    cudaEventRecord(events[1],s);
    if(rounds)start_exchange<<<1,32,0,s>>>(p,w.masks,w.rows,w.exchange);
    cudaEventRecord(events[2],s);
    for(int r=0;r<rounds;++r) {
        evaluate_exchanges<<<((w.n+1)*101+127)/128,128,0,s>>>(p,w.masks,w.rows,w.exchange,w.proposal_values);
        cudaEventRecord(events[3+2*r],s);
        accept_exchange<<<1,256,0,s>>>(p,w.masks,w.rows,w.exchange,w.proposal_values);
        cudaEventRecord(events[4+2*r],s);
    }
    search<<<w.tasks,128,0,s>>>(p,w.bits,w.tasks,cap,w.masks,w.active,w.phase,w.exclusions,w.rows);
    cudaEventRecord(events[3+2*rounds],s);
    finish<<<1,32,0,s>>>(p,w.tasks,w.masks,w.rows,w.out,w.result);
    cudaEventRecord(events[4+2*rounds],s);
    if(cudaGetLastError()!=cudaSuccess||cudaStreamSynchronize(s)!=cudaSuccess){release();return 2;}
    for(size_t i=1;i<events.size();++i) {
        float value=0;if(cudaEventElapsedTime(&value,events[i-1],events[i])!=cudaSuccess){release();return 2;}
        milliseconds[i-1]=value;
    }
    cudaMemcpyAsync(selected,w.out,w.n,cudaMemcpyDeviceToHost,s);
    cudaMemcpyAsync(report,w.result,sizeof(Report),cudaMemcpyDeviceToHost,s);
    const int status=cudaStreamSynchronize(s)==cudaSuccess?0:2;release();return status;
}
