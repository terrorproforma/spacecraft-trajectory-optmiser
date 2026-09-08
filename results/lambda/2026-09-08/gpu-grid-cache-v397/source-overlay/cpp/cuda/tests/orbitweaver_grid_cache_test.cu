#include "spacepdhcg/cuda/orbitweaver_gpu_c_api.h"
#include <cuda_runtime.h>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

int main(int argc,char**) {
    spacepdhcg_accelerator_stream stream{};
    stream.device.type=SPACEPDHCG_DEVICE_CUDA;stream.device.id=0;
    spacepdhcg_orbitweaver_lambert_config config{};
    config.abi_version=SPACEPDHCG_ORBITWEAVER_GPU_ABI_VERSION;
    config.device_id=0;config.maximum_batch_size=16384;config.scan_samples_per_band=16;
    spacepdhcg_orbitweaver_lambert_workspace* w=nullptr;
    assert(spacepdhcg_orbitweaver_lambert_workspace_create(&config,stream,&w)==0);
    spacepdhcg_orbitweaver_hop_elements elements{};
    elements.departure={64328,149597870.7,0.0167,0,0,1.8,0.1};
    elements.arrival={64328,2.2*149597870.7,0.15,0.04,0.3,0.6,1.2};
    elements.gravitational_parameter=1.32712440018e11;
    elements.departure_allowance=6;
    const size_t n=argc>1?256:17,nt=argc>1?256:13,cells=n*nt;
    std::vector<double> epochs(n),tofs(nt),reference(cells),actual(cells);
    std::vector<uint8_t> flags(cells),actual_flags(cells);
    for(size_t i=0;i<n;++i)epochs[i]=64328+double(i)/8;
    for(size_t i=0;i<nt;++i)tofs[i]=180+double(i)/8;
    double *de{},*dt{},*dv{};uint8_t* ok{};
    assert(cudaMalloc(&de,n*sizeof(double))==0&&cudaMalloc(&dt,nt*sizeof(double))==0);
    assert(cudaMalloc(&dv,cells*sizeof(double))==0&&cudaMalloc(&ok,cells)==0);
    const auto evaluate=[&](bool cached) {
        if(cached)return int(spacepdhcg_orbitweaver_hop_grid_cached_host(w,&elements,epochs.data(),n,tofs.data(),nt,dv,ok));
        assert(cudaMemcpy(de,epochs.data(),n*sizeof(double),cudaMemcpyHostToDevice)==0);
        assert(cudaMemcpy(dt,tofs.data(),nt*sizeof(double),cudaMemcpyHostToDevice)==0);
        return int(spacepdhcg_orbitweaver_hop_grid_device(w,&elements,de,n,dt,nt,dv,ok));
    };
    const auto read=[&](std::vector<double>& values,std::vector<uint8_t>& feasible) {
        assert(cudaMemcpy(values.data(),dv,cells*sizeof(double),cudaMemcpyDeviceToHost)==0);
        assert(cudaMemcpy(feasible.data(),ok,cells,cudaMemcpyDeviceToHost)==0);
    };
    uint64_t hits{},misses{},evictions{},bytes{};
    const auto stats=[&] {assert(spacepdhcg_orbitweaver_hop_grid_cache_stats(w,&hits,&misses,&evictions,&bytes)==0);};
    for(int change=0;change<6;++change) {
        if(change==1)epochs[0]+=0.25;
        if(change==2)tofs[0]+=0.5;
        if(change==3)elements.arrival_allowance=1;
        if(change==4)elements.arrival.mean+=0.01;
        if(change==5)elements.gravitational_parameter*=1.00001;
        assert(evaluate(false)==0);read(reference,flags);
        assert(evaluate(true)==0);read(actual,actual_flags);
        assert(std::memcmp(reference.data(),actual.data(),cells*sizeof(double))==0&&flags==actual_flags);
        assert(cudaMemset(dv,0,cells*sizeof(double))==0&&cudaMemset(ok,0,cells)==0);
        assert(evaluate(true)==0);read(actual,actual_flags);
        assert(std::memcmp(reference.data(),actual.data(),cells*sizeof(double))==0&&flags==actual_flags);
    }
    stats();assert(hits==6&&misses==6&&evictions==0&&bytes==6*cells*9);
    const auto good_epoch=epochs[0];epochs[0]=NAN;assert(evaluate(true)!=0);epochs[0]=good_epoch;
    const auto good_tof=tofs[0];tofs[0]=0;assert(evaluate(true)!=0);tofs[0]=good_tof;
    stats();assert(hits==6&&misses==6);
    if(argc>1) {
        const auto allowance=elements.arrival_allowance;
        for(int i=0;i<120;++i){elements.arrival_allowance=2+i*.001;assert(evaluate(true)==0);}
        stats();assert(evictions>0&&bytes<=64U*1024U*1024U);
        const auto old_misses=misses;elements.arrival_allowance=allowance;
        assert(evaluate(true)==0);read(actual,actual_flags);stats();assert(misses==old_misses+1);
        assert(std::memcmp(reference.data(),actual.data(),cells*sizeof(double))==0&&flags==actual_flags);
    }
    assert(spacepdhcg_orbitweaver_lambert_workspace_destroy(&w)==0&&w==nullptr);
    assert(spacepdhcg_orbitweaver_lambert_workspace_create(&config,stream,&w)==0);
    stats();assert(hits==0&&misses==0&&bytes==0);
    assert(evaluate(true)==0);read(actual,actual_flags);
    assert(std::memcmp(reference.data(),actual.data(),cells*sizeof(double))==0&&flags==actual_flags);
    assert(spacepdhcg_orbitweaver_lambert_workspace_destroy(&w)==0);
    cudaFree(de);cudaFree(dt);cudaFree(dv);cudaFree(ok);
    std::puts("grid cache: exact cold/warm/changed-input parity, invalid inputs, lifecycle and bounded eviction passed");
}
