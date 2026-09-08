// Exercise the production kernels, including equal-key ordering across blocks.
#include "../src/orbitweaver_gpu.cu"
#include <cstdio>
#include <cstdlib>

#define REQUIRE(x) do { if(!(x)){std::fprintf(stderr,"line %d: %s\n",__LINE__,#x);std::exit(1);} } while(0)
#define CUDA(x) REQUIRE((x)==cudaSuccess)
using Option=spacepdhcg_orbitweaver_hop_option;
using Hop=spacepdhcg_orbitweaver_hop_result;
static_assert(sizeof(Option)==24);

std::vector<Option> expected(const std::vector<Hop>& hops,const std::vector<double>& times,bool sorted) {
    std::vector<Option> out;
    for(size_t i=0;i<hops.size();++i){
        const double dv=hops[i].departure_delta_v+hops[i].arrival_delta_v;
        if(hops[i].feasible&&std::isfinite(dv))out.push_back({dv,times[2*i],times[2*i+1]});
    }
    if(sorted)std::stable_sort(out.begin(),out.end(),[](const Option& a,const Option& b){
        return a.delta_v<b.delta_v||(a.delta_v==b.delta_v&&a.departure>b.departure);
    });
    return out;
}

void synthetic(int n,bool all_invalid,bool sorted) {
    std::vector<Hop> hops(n);std::vector<double> times(2*n);
    for(int i=0;i<n;++i){
        hops[i].feasible=!all_invalid&&i%7!=0;
        hops[i].departure_delta_v=double(i%9);hops[i].arrival_delta_v=.5;
        if(i%11==0)hops[i].arrival_delta_v=INFINITY;
        if(i%17==0)hops[i].departure_delta_v=NAN;
        times[2*i]=64328+double(i%13)/8;times[2*i+1]=100+i; // unique row identity through ties
    }
    auto wanted=expected(hops,times,sorted);
    Hop *dh{};double* dt{};HopOptionScratch scratch;scratch.capacity=n;
    CUDA(cudaMalloc(&dh,n*sizeof(Hop)));CUDA(cudaMalloc(&dt,2*n*sizeof(double)));
    CUDA(cudaMalloc(&scratch.rows,n*sizeof(RankedHopOption)));CUDA(cudaMalloc(&scratch.compact,n*sizeof(RankedHopOption)));
    CUDA(cudaMalloc(&scratch.packed,n*sizeof(Option)));CUDA(cudaMalloc(&scratch.selected,sizeof(int)));
    size_t sorting=0,filtering=0;
    CUDA(cub::DeviceMergeSort::SortKeys(nullptr,sorting,scratch.rows,n,HopOptionLess{}));
    CUDA(cub::DeviceSelect::If(nullptr,filtering,scratch.rows,scratch.compact,scratch.selected,n,ValidHopOption{}));
    scratch.temporary_bytes=std::max(sorting,filtering);CUDA(cudaMalloc(&scratch.temporary,scratch.temporary_bytes));
    CUDA(cudaMemcpy(dh,hops.data(),n*sizeof(Hop),cudaMemcpyHostToDevice));
    CUDA(cudaMemcpy(dt,times.data(),2*n*sizeof(double),cudaMemcpyHostToDevice));
    build_hop_options<<<(n+127)/128,128>>>(dh,dt,n,0,scratch.rows);
    auto bytes=scratch.temporary_bytes;
    if(sorted)CUDA(cub::DeviceMergeSort::SortKeys(scratch.temporary,bytes,scratch.rows,n,HopOptionLess{}));
    bytes=scratch.temporary_bytes;
    CUDA(cub::DeviceSelect::If(scratch.temporary,bytes,scratch.rows,scratch.compact,scratch.selected,n,ValidHopOption{}));
    pack_hop_options<<<(n+127)/128,128>>>(scratch.compact,scratch.selected,scratch.packed);
    CUDA(cudaGetLastError());int selected=-1;CUDA(cudaMemcpy(&selected,scratch.selected,sizeof(int),cudaMemcpyDeviceToHost));
    REQUIRE(size_t(selected)==wanted.size());std::vector<Option> got(selected);
    if(selected){CUDA(cudaMemcpy(got.data(),scratch.packed,selected*sizeof(Option),cudaMemcpyDeviceToHost));REQUIRE(std::memcmp(got.data(),wanted.data(),selected*sizeof(Option))==0);}
    CUDA(cudaFree(dh));CUDA(cudaFree(dt));
}

int main() {
    for(int n:{1,31,533,1025,4097})for(bool invalid:{false,true})for(bool sorted:{false,true})synthetic(n,invalid,sorted);
    spacepdhcg_accelerator_stream stream{};stream.device.type=SPACEPDHCG_DEVICE_CUDA;stream.device.id=0;
    spacepdhcg_orbitweaver_lambert_config config{};config.abi_version=SPACEPDHCG_ORBITWEAVER_GPU_ABI_VERSION;
    config.maximum_batch_size=31;config.scan_samples_per_band=256;
    spacepdhcg_orbitweaver_lambert_workspace* w{};
    REQUIRE(spacepdhcg_orbitweaver_lambert_workspace_create(&config,stream,&w)==0);
    spacepdhcg_orbitweaver_hop_elements elements{};
    elements.departure={64328,149597870.7,.0167,0,0,1.8,.1};
    elements.arrival={64328,2.2*149597870.7,.15,.04,.3,.6,1.2};
    elements.gravitational_parameter=1.32712440018e11;elements.departure_allowance=6;
    for(size_t n:{1,7,533,1025,3}) {
        std::vector<double> times(2*n);std::vector<Hop> hops(n);std::vector<Option> got(n);
        for(size_t i=0;i<n;++i){times[2*i]=64328+double(i%17)/8;times[2*i+1]=180+double(i%23)/8;if(i%19==0)times[2*i+1]=0;}
        for(size_t i=0;i<n;i+=31){const size_t size=std::min(size_t(31),n-i);REQUIRE(spacepdhcg_orbitweaver_hop_elements_host(w,&elements,times.data()+2*i,size,hops.data()+i,size)==0);}
        for(int sorted:{0,1}) {
            auto wanted=expected(hops,times,sorted);size_t selected=n+1;
            REQUIRE(spacepdhcg_orbitweaver_hop_options_host(w,&elements,times.data(),n,sorted,got.data(),n,&selected)==0);
            REQUIRE(selected==wanted.size());if(selected)REQUIRE(std::memcmp(wanted.data(),got.data(),selected*sizeof(Option))==0);
        }
        size_t selected=0;
        REQUIRE(spacepdhcg_orbitweaver_hop_options_host(w,&elements,times.data(),n,2,got.data(),n,&selected)!=0);
        REQUIRE(spacepdhcg_orbitweaver_hop_options_host(w,&elements,times.data(),n,0,got.data(),n-1,&selected)!=0);
    }
    REQUIRE(spacepdhcg_orbitweaver_lambert_workspace_destroy(&w)==0&&w==nullptr);
    std::puts("PASS: 20 synthetic sort/filter cases, exact public bridge parity, chunking, scratch growth/reuse, invalid inputs and lifecycle");
}
