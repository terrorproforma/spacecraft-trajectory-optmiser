#pragma once
#include <cub/cub.cuh>
#include <climits>

namespace gtoc12_fleet {
using Event=spacepdhcg_gtoc12_fleet_event;
using Mass=spacepdhcg_gtoc12_fleet_mass;
static_assert(sizeof(Event)==16&&sizeof(Mass)==16);
struct Routes {
    int n;const Column* metadata;
    const int *d,*c,*f,*m;const Event *deploy,*foreign;
    const int64_t* collect;const Mass* mass;
};
struct Sum {
    double high=0,low=0;
    __device__ void add(double x) {
        const double next=__dadd_rn(high,x);
        const double correction=fabs(high)>=fabs(x)?__dadd_rn(__dsub_rn(high,next),x):
                                                       __dadd_rn(__dsub_rn(x,next),high);
        low=__dadd_rn(low,correction);high=next;
    }
    __device__ double value() const {return low&&isfinite(low)?__dadd_rn(high,low):high;}
};
__global__ void score_routes(Routes r,Column* values) {
    const int i=int(blockIdx.x*blockDim.x+threadIdx.x);if(i>=r.n)return;
    Column c=r.metadata[i];c.value=c.mass=0;
    // Neumaier compensated sums match the Python 3.12 reference's float sums.
    // Keep products separately rounded; never fuse weighting with accumulation.
    Sum value,mass;
    if(c.reserved)for(int k=r.m[i];k<r.m[i+1];++k) {
        value.add(__dmul_rn(r.mass[k].mass,r.mass[k].weight));mass.add(r.mass[k].mass);
    }
    c.value=value.value();c.mass=mass.value();
    values[i]=c;
}
__global__ void sort_route_keys(Routes r,Event* deploy,int64_t* collect) {
    const int i=int(blockIdx.x*blockDim.x+threadIdx.x);if(i>=r.n)return;
    for(int a=r.d[i]+1;a<r.d[i+1];++a) {
        const Event key=deploy[a];int b=a;
        while(b>r.d[i]&&deploy[b-1].asteroid>key.asteroid){deploy[b]=deploy[b-1];--b;}
        deploy[b]=key;
    }
    for(int a=r.c[i]+1;a<r.c[i+1];++a) {
        const int64_t key=collect[a];int b=a;
        while(b>r.c[i]&&collect[b-1]>key){collect[b]=collect[b-1];--b;}
        collect[b]=key;
    }
}
__global__ void provider_availability(Routes r,int* available) {
    const int k=int(blockIdx.x),lane=int(threadIdx.x);int found=0;
    const Event need=r.foreign[k];
    for(int j=lane;j<r.n&&!found;j+=128)if(r.metadata[j].reserved==1)
        for(int d=r.d[j];d<r.d[j+1];++d) {
            if(r.deploy[d].asteroid>need.asteroid)break;
            if(r.deploy[d].asteroid==need.asteroid&&fabs(r.deploy[d].epoch-need.epoch)<=1e-6){found=1;break;}
        }
    __shared__ cub::BlockReduce<int,128>::TempStorage temp;
    found=cub::BlockReduce<int,128>(temp).Reduce(found,cub::Max());
    if(!lane)available[k]=found;
}
__global__ void eligible_routes(Routes r,const int* available,uint8_t* usable) {
    const int i=int(blockIdx.x*blockDim.x+threadIdx.x);if(i>=r.n)return;
    bool valid=r.metadata[i].reserved==1;
    for(int k=r.f[i];valid&&k<r.f[i+1];++k)valid=available[k];
    usable[i]=valid;
}
__global__ void canonical_routes(Routes r,const Column* values,const uint8_t* usable,
                                 Column* sorted,int* permutation,int* info,
                                 double* magnitude,double* mass) {
    const int i=int(blockIdx.x*blockDim.x+threadIdx.x);if(i>=r.n)return;
    magnitude[i]=mass[i]=0;
    if(r.metadata[i].reserved!=0&&r.metadata[i].reserved!=1)atomicExch(info+1,1);
    if(!usable[i])return;
    const Column c=values[i];
    if(c.ships<=0||c.ships>100||!isfinite(c.value)||!isfinite(c.mass)||c.mass<0) {
        atomicExch(info+1,1);return;
    }
    int rank=0;
    for(int j=0;j<r.n;++j)if(usable[j]) {
        if(i!=j&&values[j].identifier==c.identifier)atomicExch(info+1,1);
        if(values[j].value>c.value||(values[j].value==c.value&&
           (values[j].identifier<c.identifier||(values[j].identifier==c.identifier&&j<i))))++rank;
    }
    sorted[rank]=c;permutation[rank]=i;atomicAdd(info,1);
    magnitude[i]=fabs(c.value);mass[i]=c.mass;
}
__global__ void validate_totals(const double* totals,int* info) {
    if(!isfinite(totals[0])||!isfinite(totals[1]))info[1]=1;
}
__device__ bool route_conflict(Routes r,int i,int j) {
    int a=r.d[i],b=r.d[j];
    while(a<r.d[i+1]&&b<r.d[j+1]) {
        if(r.deploy[a].asteroid==r.deploy[b].asteroid)return true;
        if(r.deploy[a].asteroid<r.deploy[b].asteroid)++a;else ++b;
    }
    a=r.c[i];b=r.c[j];
    while(a<r.c[i+1]&&b<r.c[j+1]) {
        if(r.collect[a]==r.collect[b])return true;
        if(r.collect[a]<r.collect[b])++a;else ++b;
    }
    return false;
}
__global__ void route_conflicts(Routes r,int n,const int* map,uint8_t* matrix,int* counts) {
    const int i=int(blockIdx.x),lane=int(threadIdx.x);int count=0;
    for(int j=lane;j<n;j+=128) {
        const bool conflict=i!=j&&route_conflict(r,map[i],map[j]);
        matrix[size_t(i)*n+j]=conflict;count+=conflict;
    }
    __shared__ cub::BlockReduce<int,128>::TempStorage temp;
    count=cub::BlockReduce<int,128>(temp).Sum(count);
    if(!lane)counts[i+1]=count;
}
__global__ void requirement_counts(Routes r,int n,const int* map,int* counts) {
    const int i=int(blockIdx.x*blockDim.x+threadIdx.x);if(i>=n)return;
    counts[i+1]=r.f[map[i]+1]-r.f[map[i]];
}
__global__ void map_requirements(Routes r,int n,const int* map,const int* offsets,int* groups) {
    const int i=int(blockIdx.x*blockDim.x+threadIdx.x);if(i>=n)return;
    for(int g=r.f[map[i]];g<r.f[map[i]+1];++g)groups[offsets[i]+g-r.f[map[i]]]=g;
}
__global__ void route_providers(Routes r,int n,const int* map,const int* offsets,
                                const int* groups,uint8_t* matrix,int* counts) {
    const int g=int(blockIdx.x),lane=int(threadIdx.x);
    if(g>=offsets[n]){if(!lane)counts[g+1]=0;return;}
    const Event need=r.foreign[groups[g]];int count=0;
    for(int j=lane;j<n;j+=128) {
        bool found=false;const int raw=map[j];
        for(int d=r.d[raw];d<r.d[raw+1];++d) {
            if(r.deploy[d].asteroid>need.asteroid)break;
            if(r.deploy[d].asteroid==need.asteroid&&fabs(r.deploy[d].epoch-need.epoch)<=1e-6){found=true;break;}
        }
        matrix[size_t(g)*n+j]=found;count+=found;
    }
    __shared__ cub::BlockReduce<int,128>::TempStorage temp;
    count=cub::BlockReduce<int,128>(temp).Sum(count);
    if(!lane)counts[g+1]=count;
}
__global__ void compact_topology(int n,const int* ro,const uint8_t* conflicts,
                                const uint8_t* providers,const int* co,int* ci,const int* po,int* pi) {
    const int i=int(blockIdx.x*blockDim.x+threadIdx.x);
    if(i<n) {
        int k=co[i];for(int j=0;j<n;++j)if(conflicts[size_t(i)*n+j])ci[k++]=j;
    }
    if(i<ro[n]) {
        int k=po[i];for(int j=0;j<n;++j)if(providers[size_t(i)*n+j])pi[k++]=j;
    }
}
int init_routes(Workspace& w,int n,int bits,const Column* metadata,
                const int* d,const Event* deploy,const int* c,const int64_t* collect,
                const int* f,const Event* foreign,const int* m,const Mass* mass,
                int* permutation,int* usable_count) {
    if(!w.open())return 2;
    Memory raw;raw.pointers.reserve(24);raw.stream=w.m.stream;raw.device=w.m.device;raw.owns_stream=false;
    auto s=w.m.stream;Column *input{},*values{};int *dd{},*dc{},*df{},*dm{},*map{},*info{},*groups{},*available{};
    Event *deps{},*foreigns{};int64_t* collects{};Mass* masses{};
    uint8_t *usable{},*conflicts{},*providers{},*temp{};double *magnitudes{},*raw_mass{},*totals{};
    if(!raw.input(input,metadata,n)||!raw.input(dd,d,n+1)||!raw.input(deps,deploy,d[n])||
       !raw.input(dc,c,n+1)||!raw.input(collects,collect,c[n])||!raw.input(df,f,n+1)||
       !raw.input(foreigns,foreign,f[n])||!raw.input(dm,m,n+1)||!raw.input(masses,mass,m[n])||
       !raw.alloc(values,n)||!raw.alloc(usable,n)||!raw.alloc(map,n)||!raw.alloc(info,2)||
       !raw.alloc(magnitudes,n)||!raw.alloc(raw_mass,n)||!raw.alloc(totals,2)||
       !raw.alloc(available,f[n])||!w.m.alloc(w.c,n))return 2;
    size_t bytes=0,b=0;
    if(cub::DeviceReduce::Sum(nullptr,bytes,magnitudes,totals,n,s)!=cudaSuccess||
       cub::DeviceScan::InclusiveSum(nullptr,b,dd,dd,n+1,s)!=cudaSuccess)return 2;
    bytes=std::max(bytes,b);
    if(cub::DeviceScan::InclusiveSum(nullptr,b,df,df,f[n]+1,s)!=cudaSuccess)return 2;
    bytes=std::max(bytes,b);if(!raw.alloc(temp,bytes))return 2;
    auto sum=[&](const double* from,double* to) {
        size_t available=bytes;return cub::DeviceReduce::Sum(temp,available,from,to,n,s);
    };
    auto scan=[&](int* data,int count) {
        size_t available=bytes;return cub::DeviceScan::InclusiveSum(temp,available,data,data,count,s);
    };
    if(cudaMemsetAsync(info,0,2*sizeof(int),s)!=cudaSuccess)return 2;
    const Routes r{n,input,dd,dc,df,dm,deps,foreigns,collects,masses};
    if(n) {
        score_routes<<<(n+127)/128,128,0,s>>>(r,values);
        sort_route_keys<<<(n+127)/128,128,0,s>>>(r,deps,collects);
    }
    if(f[n])provider_availability<<<f[n],128,0,s>>>(r,available);
    if(n) {
        eligible_routes<<<(n+127)/128,128,0,s>>>(r,available,usable);
        canonical_routes<<<(n+127)/128,128,0,s>>>(r,values,usable,w.c,map,info,magnitudes,raw_mass);
    }
    if(sum(magnitudes,totals)!=cudaSuccess||sum(raw_mass,totals+1)!=cudaSuccess)return 2;
    validate_totals<<<1,1,0,s>>>(totals,info);
    int metadata_out[2]{};
    if(cudaGetLastError()!=cudaSuccess||cudaMemcpyAsync(metadata_out,info,sizeof(metadata_out),cudaMemcpyDeviceToHost,s)!=cudaSuccess||
       cudaStreamSynchronize(s)!=cudaSuccess)return 2;
    if(metadata_out[1])return 1;
    const int usable_n=metadata_out[0],nf=f[n];
    if(!w.buffers(usable_n,bits)||!w.m.alloc(w.dco,usable_n+1)||!w.m.alloc(w.dro,usable_n+1)||
       !w.m.alloc(w.dpo,nf+1)||!raw.alloc(conflicts,size_t(usable_n)*usable_n)||
       !raw.alloc(providers,size_t(usable_n)*nf)||!raw.alloc(groups,nf))return 2;
    if(cudaMemsetAsync(w.dco,0,sizeof(int),s)!=cudaSuccess||cudaMemsetAsync(w.dro,0,sizeof(int),s)!=cudaSuccess||
       cudaMemsetAsync(w.dpo,0,sizeof(int),s)!=cudaSuccess)return 2;
    if(usable_n) {
        route_conflicts<<<usable_n,128,0,s>>>(r,usable_n,map,conflicts,w.dco);
        requirement_counts<<<(usable_n+127)/128,128,0,s>>>(r,usable_n,map,w.dro);
    }
    if(scan(w.dco,usable_n+1)!=cudaSuccess||scan(w.dro,usable_n+1)!=cudaSuccess)return 2;
    if(usable_n)map_requirements<<<(usable_n+127)/128,128,0,s>>>(r,usable_n,map,w.dro,groups);
    if(nf)route_providers<<<nf,128,0,s>>>(r,usable_n,map,w.dro,groups,providers,w.dpo);
    if(scan(w.dpo,nf+1)!=cudaSuccess)return 2;
    int conflict_count=0,provider_count=0;
    if(cudaGetLastError()!=cudaSuccess||cudaMemcpyAsync(&conflict_count,w.dco+usable_n,sizeof(int),cudaMemcpyDeviceToHost,s)!=cudaSuccess||
       cudaMemcpyAsync(&provider_count,w.dpo+nf,sizeof(int),cudaMemcpyDeviceToHost,s)!=cudaSuccess||cudaStreamSynchronize(s)!=cudaSuccess)return 2;
    if(!w.m.alloc(w.dci,conflict_count)||!w.m.alloc(w.dpi,provider_count))return 2;
    if(usable_n||nf)compact_topology<<<(std::max(usable_n,nf)+127)/128,128,0,s>>>(usable_n,w.dro,conflicts,providers,w.dco,w.dci,w.dpo,w.dpi);
    if(usable_n)rank_columns<<<(usable_n+127)/128,128,0,s>>>(w.problem(100),w.order);
    if(cudaGetLastError()!=cudaSuccess||(usable_n&&cudaMemcpyAsync(permutation,map,size_t(usable_n)*sizeof(int),cudaMemcpyDeviceToHost,s)!=cudaSuccess)||
       cudaStreamSynchronize(s)!=cudaSuccess)return 2;
    *usable_count=usable_n;return 0;
}
}
extern "C" int spacepdhcg_gtoc12_fleet_workspace_create_routes_host(int32_t n,int32_t bits,
    const spacepdhcg_gtoc12_fleet_column* metadata,
    const int32_t* d,const spacepdhcg_gtoc12_fleet_event* deploy,
    const int32_t* c,const int64_t* collect,const int32_t* f,const spacepdhcg_gtoc12_fleet_event* foreign,
    const int32_t* m,const spacepdhcg_gtoc12_fleet_mass* mass,void** workspace,int32_t* permutation,int32_t* usable_count) {
    using namespace gtoc12_fleet;
    if(!workspace||!usable_count)return 1;*workspace=nullptr;*usable_count=0;
    if(n<0||n>4096||bits<0||bits>10||(n&&(!metadata||!permutation))||
       !offsets(d,n)||!offsets(c,n)||!offsets(f,n)||!offsets(m,n)||
       (d[n]&&!deploy)||(c[n]&&!collect)||(f[n]&&!foreign)||(m[n]&&!mass)||
       f[n]>=INT_MAX||uint64_t(f[n])*uint64_t(n)>INT_MAX)return 1;
    try {
        auto w=std::make_unique<Workspace>();
        const int status=init_routes(*w,n,bits,metadata,d,deploy,c,collect,f,foreign,m,mass,permutation,usable_count);
        if(status)return status;*workspace=w.release();return 0;
    }catch(...){return 2;}
}
