#include "spacepdhcg/cuda/gtoc12_retime_c_api.h"
#include <cuda_runtime.h>
#include <cmath>
#include <mutex>
#include <new>
#include <utility>

namespace {
using Stage=spacepdhcg_gtoc12_retime_stage;
static_assert(sizeof(Stage)==104);
struct Result { double objective; int32_t feasible; int32_t reserved; };
struct Workspace {
    int device,n,stages,cells,nt;
    cudaStream_t stream{};
    double *epochs{},*dv{},*swept{},*tofs{},*value{},*next{},*departure{},*path_dv{};
    uint8_t *ok{},*swept_ok{};
    int32_t *shifts{},*back_camp{},*back_leg{},*arrivals{},*departures{};
    Stage* params{};Result* result{};
    std::mutex mutex;
    ~Workspace() {
        cudaStreamSynchronize(stream);
        cudaFree(epochs);cudaFree(dv);cudaFree(swept);cudaFree(tofs);
        cudaFree(value);cudaFree(next);cudaFree(departure);cudaFree(ok);cudaFree(swept_ok);
        cudaFree(shifts);cudaFree(back_camp);cudaFree(back_leg);cudaFree(arrivals);
        cudaFree(departures);cudaFree(params);cudaFree(result);cudaFree(path_dv);
        if(stream)cudaStreamDestroy(stream);
    }
};
template<class T> bool allocate(T*& p,size_t n) {return cudaMalloc(&p,n*sizeof(T))==cudaSuccess;}
template<class T> bool upload(T* dst,const T* src,size_t n,cudaStream_t stream) {
    return cudaMemcpyAsync(dst,src,n*sizeof(T),cudaMemcpyHostToDevice,stream)==cudaSuccess;
}
__global__ void camp(const Stage* params,int stage,const double* epochs,int n,
    const double* value,double* departure,int32_t* back) {
    const size_t index=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(index>=size_t(n))return;const int d=int(index);
    const auto s=params[stage];double best=-INFINITY;int winner=0;
    // Preserve the reference's strict > comparison and shortest-camp tie order.
    for(int c=s.camp_min;c<=s.camp_max&&c<=d;++c) {
        const int a=d-c;const double candidate=value[a]-s.arrival_rate*epochs[a];
        if(candidate>best){best=candidate;winner=a;}
    }
    best+=s.departure_rate*epochs[d];
    if(epochs[d]<s.earliest_collect-1e-9)best=-INFINITY;
    departure[d]=best;back[stage*n+d]=winner;
}
__device__ double return_base(double tof) {
    constexpr double days[]={352,420,450,480,510,540,578,630,690,810};
    constexpr double base[]={1.323,1.383,1.295,1.195,1.099,.977,.885,.930,.932,1.014};
    if(tof<=days[0])return base[0];
    for(int i=1;i<10;++i)if(tof<=days[i])
        return base[i-1]+(base[i]-base[i-1])/(days[i]-days[i-1])*(tof-days[i-1]);
    return base[9];
}
__global__ void leg(const Stage* params,int stage,int n,const double* dv,
    const uint8_t* feasible,const double* swept,const uint8_t* swept_ok,
    const double* tofs,const int32_t* shifts,const double* departure,
    double price,double thrust,double exhaust,double* next,int32_t* back) {
    const size_t index=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(index>=size_t(n))return;const int a=int(index);
    const auto s=params[stage];double best=-INFINITY;int winner=0;
    if(s.pinned_next<0||a==s.pinned_next)for(int k=0;k<s.tofs;++k) {
        const int shift=shifts[s.tof_offset+k];if(shift>=n)break;
        const int d=a-shift;if(d<0)continue;
        const int cell=s.cell_offset+d*s.tofs+k;
        if(!feasible[cell]||!swept_ok[cell])continue;
        const double tof=tofs[s.tof_offset+k],delta=dv[cell];
        const double authority=thrust/s.mass*1e-3*tof*86400.;
        const bool measured=!isnan(swept[cell]);
        if(!measured&&!(delta<=s.ratio_limit*authority))continue;
        const double ratio=(isfinite(delta)?delta:0.0)/fmax(authority,1e-12);
        double inflation=s.flat;
        if(s.model==1)inflation=(s.floor+s.slope*ratio)*s.calibration;
        if(s.model==2)inflation=fmax(return_base(tof)*fmin(fmax(1+.6*(ratio-.33),.85),1.2),.85)*s.calibration;
        if(measured)inflation=swept[cell];
        const double propellant=s.mass*(1-exp(-(delta*inflation)/exhaust));
        const double candidate=departure[d]-price*propellant;
        if(candidate>best){best=candidate;winner=d;}
    }
    next[a]=best;back[stage*n+a]=winner;
}
__global__ void finish(const double* value,int n,int stages,const int32_t* camp_back,
    const int32_t* leg_back,int32_t* arrivals,int32_t* departures,Result* result,
    const Stage* params,const double* dv,const int32_t* shifts,double* path_dv) {
    // Only the final scalar choice and O(stages) path reconstruction are serial.
    double best=-INFINITY;int winner=0;
    for(int i=0;i<n;++i)if(value[i]>best){best=value[i];winner=i;}
    *result={best,isfinite(best)?1:0,0};
    arrivals[stages]=departures[stages]=winner;
    for(int j=stages-1;j>=0;--j) {
        departures[j]=leg_back[j*n+arrivals[j+1]];
        arrivals[j]=camp_back[j*n+departures[j]];
        path_dv[j]=INFINITY;
        if(isfinite(best)) {
            const auto s=params[j];
            const int shift=arrivals[j+1]-departures[j];
            for(int k=0;k<s.tofs;++k)if(shifts[s.tof_offset+k]==shift) {
                path_dv[j]=dv[s.cell_offset+departures[j]*s.tofs+k];break;
            }
        }
    }
}
}
static int create_tables(int32_t device,int32_t n,int32_t stages,
    int32_t cells,int32_t nt,const double* epochs,const double* dv,const uint8_t* ok,
    const double* swept,const uint8_t* swept_ok,const double* tofs,const int32_t* shifts,void** output) {
    if(!output||*output||device<0||n<=0||stages<=0||stages==INT32_MAX||cells<=0||nt<=0||!epochs
        ||!tofs||!shifts||int64_t(n)*stages>INT32_MAX)return 1;
    for(int k=0;k<nt;++k)if(shifts[k]<0||!std::isfinite(tofs[k])||tofs[k]<=0)return 1;
    for(int i=0;i<n;++i)if(!std::isfinite(epochs[i]))return 1;
    if(cudaSetDevice(device)!=cudaSuccess)return 2;
    auto* w=new(std::nothrow) Workspace{};if(!w)return 2;
    w->device=device;w->n=n;w->stages=stages;w->cells=cells;w->nt=nt;
    bool good=cudaStreamCreateWithFlags(&w->stream,cudaStreamNonBlocking)==cudaSuccess;
    good=good&&allocate(w->epochs,n)&&allocate(w->dv,cells)&&allocate(w->ok,cells)
        &&allocate(w->swept,cells)&&allocate(w->swept_ok,cells)&&allocate(w->tofs,nt)
        &&allocate(w->shifts,nt)&&allocate(w->value,n)&&allocate(w->next,n)
        &&allocate(w->departure,n)&&allocate(w->back_camp,size_t(n)*stages)
        &&allocate(w->back_leg,size_t(n)*stages)&&allocate(w->arrivals,stages+1)
        &&allocate(w->departures,stages+1)&&allocate(w->params,stages)&&allocate(w->result,1)
        &&allocate(w->path_dv,stages);
    good=good&&upload(w->epochs,epochs,n,w->stream)
        &&(!dv||upload(w->dv,dv,cells,w->stream))&&(!ok||upload(w->ok,ok,cells,w->stream))
        &&(swept?upload(w->swept,swept,cells,w->stream):cudaMemsetAsync(w->swept,255,cells*sizeof(double),w->stream)==cudaSuccess)
        &&(swept_ok?upload(w->swept_ok,swept_ok,cells,w->stream):cudaMemsetAsync(w->swept_ok,1,cells,w->stream)==cudaSuccess)
        &&upload(w->tofs,tofs,nt,w->stream)
        &&upload(w->shifts,shifts,nt,w->stream);
    const auto done=cudaStreamSynchronize(w->stream);
    if(!good||done!=cudaSuccess){delete w;return 2;}
    *output=w;return 0;
}
extern "C" int spacepdhcg_gtoc12_retime_create(int32_t device,int32_t n,int32_t stages,
    int32_t cells,int32_t nt,const double* epochs,const double* dv,const uint8_t* ok,
    const double* swept,const uint8_t* swept_ok,const double* tofs,const int32_t* shifts,void** output) {
    if(!dv||!ok||!swept||!swept_ok)return 1;
    return create_tables(device,n,stages,cells,nt,epochs,dv,ok,swept,swept_ok,tofs,shifts,output);
}
extern "C" int spacepdhcg_gtoc12_retime_create_elements(int32_t device,int32_t n,int32_t stages,
    int32_t cells,int32_t nt,const double* epochs,const double* tofs,const int32_t* shifts,
    const Stage* params,const spacepdhcg_orbitweaver_hop_elements* elements,
    spacepdhcg_orbitweaver_lambert_workspace* lambert,void** output) {
    if(!params||!elements||!lambert||n<=0||stages<=0)return 1;
    int64_t cell=0,tof=0;
    for(int j=0;j<stages;++j) {
        if(params[j].cell_offset!=cell||params[j].tof_offset!=tof||params[j].tofs<=0)return 1;
        cell+=int64_t(n)*params[j].tofs;tof+=params[j].tofs;
    }
    if(cell!=cells||tof!=nt)return 1;
    int status=create_tables(device,n,stages,cells,nt,epochs,nullptr,nullptr,nullptr,nullptr,tofs,shifts,output);
    if(status)return status;
    auto* w=static_cast<Workspace*>(*output);
    for(int j=0;j<stages;++j) {
        const auto code=spacepdhcg_orbitweaver_hop_grid_device(lambert,elements+j,w->epochs,n,
            w->tofs+params[j].tof_offset,params[j].tofs,w->dv+params[j].cell_offset,w->ok+params[j].cell_offset);
        if(code!=SPACEPDHCG_CUDA_SUCCESS){delete w;*output=nullptr;return 2;}
    }
    return 0;
}
static int evaluate_path(void* workspace,const Stage* params,
    double price,double thrust,double exhaust,int32_t* arrivals,int32_t* departures,
    double* objective,int32_t* feasible,double* path_dv) {
    auto* w=static_cast<Workspace*>(workspace);
    if(!w||!params||!arrivals||!departures||!objective||!feasible||!std::isfinite(price)
        ||price<0||!std::isfinite(thrust)||thrust<=0||!std::isfinite(exhaust)||exhaust<=0)return 1;
    std::unique_lock<std::mutex> lock(w->mutex,std::try_to_lock);if(!lock.owns_lock())return 3;
    int device=-1;if(cudaGetDevice(&device)!=cudaSuccess)return 2;if(device!=w->device)return 1;
    for(int j=0;j<w->stages;++j) {
        const auto s=params[j];
        if(s.cell_offset<0||s.tof_offset<0||s.tofs<=0||int64_t(s.cell_offset)+int64_t(w->n)*s.tofs>w->cells
            ||int64_t(s.tof_offset)+s.tofs>w->nt||s.camp_min<0||s.camp_max<s.camp_min
            ||s.pinned_next<-1||s.pinned_next>=w->n||s.model<0||s.model>2
            ||!std::isfinite(s.mass)||s.mass<=0||!std::isfinite(s.arrival_rate)
            ||!std::isfinite(s.departure_rate)||std::isnan(s.earliest_collect)
            ||std::isnan(s.ratio_limit)||s.ratio_limit<0||!std::isfinite(s.flat)
            ||!std::isfinite(s.floor)||!std::isfinite(s.slope)||!std::isfinite(s.calibration))return 1;
    }
    bool good=upload(w->params,params,w->stages,w->stream)
        &&cudaMemsetAsync(w->value,0,w->n*sizeof(double),w->stream)==cudaSuccess;
    auto* value=w->value;auto* next=w->next;
    for(int j=0;good&&j<w->stages;++j) {
        camp<<<(unsigned(w->n)+127)/128,128,0,w->stream>>>(w->params,j,w->epochs,w->n,value,w->departure,w->back_camp);
        good=cudaGetLastError()==cudaSuccess;if(!good)break;
        leg<<<(unsigned(w->n)+127)/128,128,0,w->stream>>>(w->params,j,w->n,w->dv,w->ok,w->swept,w->swept_ok,
            w->tofs,w->shifts,w->departure,price,thrust,exhaust,next,w->back_leg);
        good=cudaGetLastError()==cudaSuccess;std::swap(value,next);
    }
    Result result{};
    if(good) {
        finish<<<1,1,0,w->stream>>>(value,w->n,w->stages,w->back_camp,w->back_leg,w->arrivals,w->departures,w->result,w->params,w->dv,w->shifts,w->path_dv);
        good=cudaGetLastError()==cudaSuccess;
    }
    good=good&&cudaMemcpyAsync(&result,w->result,sizeof(result),cudaMemcpyDeviceToHost,w->stream)==cudaSuccess
        &&cudaMemcpyAsync(arrivals,w->arrivals,(w->stages+1)*sizeof(int32_t),cudaMemcpyDeviceToHost,w->stream)==cudaSuccess
        &&cudaMemcpyAsync(departures,w->departures,(w->stages+1)*sizeof(int32_t),cudaMemcpyDeviceToHost,w->stream)==cudaSuccess;
    good=good&&(!path_dv||cudaMemcpyAsync(path_dv,w->path_dv,w->stages*sizeof(double),cudaMemcpyDeviceToHost,w->stream)==cudaSuccess);
    const auto done=cudaStreamSynchronize(w->stream);
    if(!good||done!=cudaSuccess)return 2;
    *objective=result.objective;*feasible=result.feasible;return 0;
}
extern "C" int spacepdhcg_gtoc12_retime_host(void* w,const Stage* p,double price,double thrust,
    double exhaust,int32_t* a,int32_t* d,double* objective,int32_t* feasible) {
    return evaluate_path(w,p,price,thrust,exhaust,a,d,objective,feasible,nullptr);
}
extern "C" int spacepdhcg_gtoc12_retime_path_host(void* w,const Stage* p,double price,double thrust,
    double exhaust,int32_t* a,int32_t* d,double* objective,int32_t* feasible,double* dv) {
    if(!dv)return 1;
    return evaluate_path(w,p,price,thrust,exhaust,a,d,objective,feasible,dv);
}
extern "C" int spacepdhcg_gtoc12_retime_destroy(void** workspace) {
    if(!workspace)return 1;auto* w=static_cast<Workspace*>(*workspace);if(!w)return 0;
    if(cudaSetDevice(w->device)!=cudaSuccess)return 2;
    {std::unique_lock<std::mutex> lock(w->mutex,std::try_to_lock);if(!lock.owns_lock())return 3;}
    delete w;*workspace=nullptr;return 0;
}
