#include "spacepdhcg/cuda/gtoc12_expansion_c_api.h"
#include <cuda_runtime.h>
#include <cub/device/device_merge_sort.cuh>
#include <cmath>
#include <mutex>
#include <new>

namespace {
using Policy=spacepdhcg_gtoc12_expansion_policy;
using Parent=spacepdhcg_gtoc12_expansion_parent;
using Deploy=spacepdhcg_gtoc12_expansion_deploy;
using Option=spacepdhcg_gtoc12_expansion_option;
using Result=spacepdhcg_gtoc12_expansion_result;
static_assert(sizeof(Policy)==144 && sizeof(Parent)==48 && sizeof(Deploy)==32);
static_assert(sizeof(Option)==72 && sizeof(Result)==88);
struct Ranked { Result row; int ordinal; };
struct Less {
    const Parent* parents; const Deploy* deploys;
    __device__ bool operator()(const Ranked& a,const Ranked& b) const {
        const auto& x=a.row; const auto& y=b.row;
        if(x.valid!=y.valid)return x.valid>y.valid;
        if(!x.valid)return a.ordinal<b.ordinal;
        if(x.score!=y.score)return x.score>y.score;
        if(x.arrival!=y.arrival)return x.arrival<y.arrival;
        const auto px=parents[x.parent],py=parents[y.parent];
        const int nx=px.deploy_count+1,ny=py.deploy_count+1;
        for(int i=0;i<nx && i<ny;++i) {
            const int64_t ax=i==px.deploy_count?x.target:deploys[px.deploy_begin+i].body;
            const int64_t ay=i==py.deploy_count?y.target:deploys[py.deploy_begin+i].body;
            if(ax!=ay)return ax<ay;
        }
        return nx!=ny?nx<ny:a.ordinal<b.ordinal;
    }
};
struct Sum {
    double hi=0,lo=0;
    __device__ void add(double x,bool compensated) {
        const double t=hi+x;
        if(compensated)lo+=fabs(hi)>=fabs(x)?(hi-t)+x:(x-t)+hi;
        hi=t;
    }
    __device__ double value() const { return lo && isfinite(lo)?hi+lo:hi; }
};
__global__ void price(Policy p,int np,const Parent* parents,const Deploy* deploys,
    int n,const Option* options,Ranked* output,int* counts) {
    const int i=int(blockIdx.x*blockDim.x+threadIdx.x);if(i>=n)return;
    Ranked r{};r.ordinal=i;r.row.score=-INFINITY;output[i]=r;
    const Option o=options[i];
    if(o.parent<0||o.parent>=np||(o.allowed!=0&&o.allowed!=1)||o.target<=0) {
        atomicExch(counts+1,1);return;
    }
    if(!o.allowed || !isfinite(o.lookahead))return;
    const Parent parent=parents[o.parent];
    for(int j=0;j<parent.deploy_count;++j)if(deploys[parent.deploy_begin+j].body==o.target)return;
    if(!isfinite(o.departure)||!isfinite(o.tof)||o.tof<=0||!isfinite(o.weight)||
       !isfinite(o.price)||!isfinite(o.cluster_bonus)) {atomicExch(counts+1,1);return;}
    const double authority=(p.thrust/parent.mass*1e-3)*o.tof*p.day_seconds;
    if(!(o.delta_v<=p.authority_ratio*authority))return;
    const double inflation=p.ratio_inflation?p.floor+p.slope*(o.delta_v/fmax(authority,1e-12)):p.inflation;
    const double fuel=parent.mass*(1.0-exp(-(o.delta_v*inflation)/p.exhaust_velocity));
    const double arrival=o.departure+o.tof;
    if(arrival>p.arrival_horizon)return;
    const double mass=parent.mass-fuel-p.miner_mass;
    Sum mined,prices;
    for(int j=0;j<parent.deploy_count;++j) {
        const Deploy d=deploys[parent.deploy_begin+j];
        mined.add(d.weight*(p.mining_rate*fmax(p.mining_horizon-d.epoch,0.0)/p.year_days),p.compensated_sum);
        prices.add(d.price,p.compensated_sum);
    }
    mined.add(o.weight*(p.mining_rate*fmax(p.mining_horizon-arrival,0.0)/p.year_days),p.compensated_sum);
    prices.add(o.price,p.compensated_sum);
    const double lookahead=parent.lookahead+o.lookahead;
    const double score=mined.value()-p.propellant_weight*(p.initial_mass-mass)
        -p.time_weight*(arrival-parent.launch)-p.lookahead_weight*lookahead+o.cluster_bonus-prices.value();
    if(!isfinite(score)||!isfinite(fuel)||!isfinite(mass)) {atomicExch(counts+1,1);return;}
    r.row={o.parent,1,o.target,o.departure,arrival,o.delta_v,inflation,fuel,mass,score,
        lookahead,parent.hop_propellant+fuel};
    output[i]=r;atomicAdd(counts,1);
}
__global__ void unpack(const Ranked* rows,int offset,int count,Result* out) {
    const int i=int(blockIdx.x*blockDim.x+threadIdx.x);if(i<count)out[i]=rows[offset+i].row;
}
struct Workspace {
    int device=-1,np=0,nd=0,no=0,valid=-1;
    Parent* parents{};Deploy* deploys{};Option* options{};Ranked* rows{};Result* readback{};
    int* counts{};void* scratch{};size_t scratch_bytes=0;cudaStream_t stream{};std::mutex mutex;
};
bool correct(Workspace* w) {int d=-1;return w&&cudaGetDevice(&d)==cudaSuccess&&d==w->device;}
bool release(Workspace* w) {
    bool ok=true;
    for(void* p:{static_cast<void*>(w->parents),static_cast<void*>(w->deploys),static_cast<void*>(w->options),
        static_cast<void*>(w->rows),static_cast<void*>(w->readback),static_cast<void*>(w->counts),w->scratch})
        if(p&&cudaFree(p)!=cudaSuccess)ok=false;
    if(w->stream&&cudaStreamDestroy(w->stream)!=cudaSuccess)ok=false;
    return ok;
}
template<class T> bool allocate(T*& p,size_t n) {return cudaMalloc(&p,n*sizeof(T))==cudaSuccess;}
}
extern "C" int spacepdhcg_gtoc12_expansion_create(int device,int np,int nd,int no,void** out) {
    if(!out)return 1;*out=nullptr;int current=-1;
    if(device<0||np<1||nd<1||no<1||cudaGetDevice(&current)!=cudaSuccess||device!=current)return 1;
    auto* w=new(std::nothrow) Workspace;if(!w)return 2;
    w->device=device;w->np=np;w->nd=nd;w->no=no;
    bool ok=cudaStreamCreateWithFlags(&w->stream,cudaStreamNonBlocking)==cudaSuccess &&
        allocate(w->parents,np)&&allocate(w->deploys,nd)&&allocate(w->options,no)&&allocate(w->rows,no)&&
        allocate(w->readback,no)&&allocate(w->counts,2);
    if(ok)ok=cub::DeviceMergeSort::SortKeys(nullptr,w->scratch_bytes,w->rows,no,Less{w->parents,w->deploys},w->stream)==cudaSuccess;
    if(ok)ok=cudaMalloc(&w->scratch,w->scratch_bytes)==cudaSuccess;
    if(!ok){release(w);delete w;return 2;}*out=w;return 0;
}
extern "C" int spacepdhcg_gtoc12_expansion_destroy(void** handle) {
    if(!handle)return 1;auto* w=static_cast<Workspace*>(*handle);if(!w)return 0;
    if(!correct(w))return 1;std::unique_lock<std::mutex> lock(w->mutex,std::try_to_lock);if(!lock.owns_lock())return 3;
    const bool done=cudaStreamSynchronize(w->stream)==cudaSuccess;const bool freed=release(w);
    lock.unlock();delete w;*handle=nullptr;return done&&freed?0:2;
}
extern "C" int spacepdhcg_gtoc12_expansion_rank(void* opaque,const Policy* p,
    int np,const Parent* parents,int nd,const Deploy* deploys,int no,const Option* options,int* valid) {
    auto* w=static_cast<Workspace*>(opaque);if(!correct(w))return 1;
    std::unique_lock<std::mutex> lock(w->mutex,std::try_to_lock);if(!lock.owns_lock())return 3;w->valid=-1;
    if(!valid||!p||np<1||np>w->np||nd<1||nd>w->nd||no<0||no>w->no||!parents||!deploys||(no&&!options))return 1;
    if(p->abi_version!=1||p->reserved||(p->ratio_inflation!=0&&p->ratio_inflation!=1)||
       (p->compensated_sum!=0&&p->compensated_sum!=1))return 1;
    const double numbers[]={p->thrust,p->day_seconds,p->exhaust_velocity,p->authority_ratio,p->inflation,
        p->floor,p->slope,p->miner_mass,p->initial_mass,p->arrival_horizon,p->mining_horizon,p->mining_rate,
        p->year_days,p->propellant_weight,p->time_weight,p->lookahead_weight};
    for(double x:numbers)if(!std::isfinite(x))return 1;
    if(p->thrust<=0||p->day_seconds<=0||p->exhaust_velocity<=0||p->year_days<=0)return 1;
    int64_t end=0;
    for(int i=0;i<np;++i) {
        const Parent a=parents[i];
        if(a.deploy_begin!=end||a.deploy_count<1)return 1;end+=a.deploy_count;if(end>nd)return 1;
        if(!std::isfinite(a.mass)||!std::isfinite(a.epoch)||!std::isfinite(a.launch)||
           !std::isfinite(a.hop_propellant)||!std::isfinite(a.lookahead))return 1;
    }
    if(end!=nd)return 1;
    for(int i=0;i<nd;++i)if(deploys[i].body<=0||!std::isfinite(deploys[i].epoch)||
        !std::isfinite(deploys[i].weight)||!std::isfinite(deploys[i].price))return 1;
    if(!no){w->valid=0;*valid=0;return 0;}
    auto status=cudaMemcpyAsync(w->parents,parents,size_t(np)*sizeof(Parent),cudaMemcpyHostToDevice,w->stream);
    if(status==cudaSuccess)status=cudaMemcpyAsync(w->deploys,deploys,size_t(nd)*sizeof(Deploy),cudaMemcpyHostToDevice,w->stream);
    if(status==cudaSuccess)status=cudaMemcpyAsync(w->options,options,size_t(no)*sizeof(Option),cudaMemcpyHostToDevice,w->stream);
    if(status==cudaSuccess)status=cudaMemsetAsync(w->counts,0,2*sizeof(int),w->stream);
    if(status==cudaSuccess){price<<<unsigned((int64_t(no)+127)/128),128,0,w->stream>>>(*p,np,w->parents,w->deploys,no,w->options,w->rows,w->counts);status=cudaGetLastError();}
    auto bytes=w->scratch_bytes;
    if(status==cudaSuccess)status=cub::DeviceMergeSort::SortKeys(w->scratch,bytes,w->rows,no,Less{w->parents,w->deploys},w->stream);
    int counts[2]{};if(status==cudaSuccess)status=cudaMemcpyAsync(counts,w->counts,sizeof(counts),cudaMemcpyDeviceToHost,w->stream);
    const auto done=cudaStreamSynchronize(w->stream);if(status!=cudaSuccess||done!=cudaSuccess)return 2;
    if(counts[1])return 1;w->valid=counts[0];*valid=counts[0];return 0;
}
extern "C" int spacepdhcg_gtoc12_expansion_read(void* opaque,int offset,int n,Result* results) {
    auto* w=static_cast<Workspace*>(opaque);if(!correct(w))return 1;
    std::unique_lock<std::mutex> lock(w->mutex,std::try_to_lock);if(!lock.owns_lock())return 3;
    if(w->valid<0||offset<0||n<0||int64_t(offset)+n>w->valid||(n&&!results))return 1;if(!n)return 0;
    unpack<<<unsigned((int64_t(n)+127)/128),128,0,w->stream>>>(w->rows,offset,n,w->readback);
    auto status=cudaGetLastError();if(status==cudaSuccess)status=cudaMemcpyAsync(results,w->readback,size_t(n)*sizeof(Result),cudaMemcpyDeviceToHost,w->stream);
    const auto done=cudaStreamSynchronize(w->stream);return status==cudaSuccess&&done==cudaSuccess?0:2;
}
