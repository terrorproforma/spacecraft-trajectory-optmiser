#include "spacepdhcg/cuda/gtoc12_retime_c_api.h"
#include <cuda_runtime.h>
#include <cmath>
#include <cstring>
#include <mutex>
#include <new>
#include <utility>

namespace {
using Stage=spacepdhcg_gtoc12_retime_stage;
using SweepCell=spacepdhcg_gtoc12_sweep_cell;
using ForwardVisit=spacepdhcg_gtoc12_forward_visit;
using ForwardPolicy=spacepdhcg_gtoc12_forward_policy;
using ForwardResult=spacepdhcg_gtoc12_forward_result;
using DriverPolicy=spacepdhcg_gtoc12_driver_policy;
using DriverWeight=spacepdhcg_gtoc12_driver_weight;
using DriverResult=spacepdhcg_gtoc12_driver_result;
struct DriverState {
    DriverResult result;
    double lo,hi,best_price;
    int32_t round,bisections;
};
static_assert(sizeof(Stage)==104);
static_assert(sizeof(SweepCell)==24);
struct Result { double objective; int32_t feasible; int32_t reserved; };
struct Controls { double price,thrust,exhaust; int32_t forward,reserved; };
struct PathLayout {
    size_t arrivals,departures,dv,swept,ok,forward,masses,inflations,collected,bytes;
    explicit PathLayout(int stages) {
        arrivals=sizeof(Result);departures=arrivals+(size_t(stages)+1)*sizeof(int32_t);
        // The two int32 arrays together occupy a multiple of eight bytes.
        dv=departures+(size_t(stages)+1)*sizeof(int32_t);
        swept=dv+size_t(stages)*sizeof(double);ok=swept+size_t(stages)*sizeof(double);
        forward=(ok+size_t(stages)+7)&~size_t(7);masses=forward+sizeof(ForwardResult);
        inflations=masses+size_t(stages)*sizeof(double);
        collected=inflations+size_t(stages)*sizeof(double);
        bytes=collected+(size_t(stages)+1)*sizeof(double);
    }
};
struct Workspace {
    int device,n,stages,cells,nt;
    cudaStream_t stream{};
    double *epochs{},*dv{},*swept{},*tofs{},*value{},*next{},*departure{},*path_dv{},*path_swept{};
    uint8_t *ok{},*swept_ok{},*path_swept_ok{};
    SweepCell* samples{};int sample_capacity{};
    int32_t *shifts{},*back_camp{},*back_leg{},*arrivals{},*departures{};
    Stage* params{};Result* result{};
    Controls* controls{};
    ForwardVisit* visits{};ForwardPolicy* policy{};
    DriverPolicy* driver_policy{};DriverWeight* weights{};DriverState* driver_state{};
    uint8_t *best_output{},*host_best{};
    cudaGraph_t driver_graph{};cudaGraphExec_t driver_exec{};
    uint8_t *output{},*host_output{};
    cudaGraph_t graph{};cudaGraphExec_t graph_exec{};
    bool use_graph=true;
    uint64_t graph_builds{},graph_launches{};
    std::mutex mutex;
    ~Workspace() {
        cudaStreamSynchronize(stream);
        if(graph_exec)cudaGraphExecDestroy(graph_exec);
        if(graph)cudaGraphDestroy(graph);
        if(driver_exec)cudaGraphExecDestroy(driver_exec);
        if(driver_graph)cudaGraphDestroy(driver_graph);
        cudaFree(driver_policy);cudaFree(weights);cudaFree(driver_state);
        cudaFree(best_output);delete[] host_best;
        cudaFree(epochs);cudaFree(dv);cudaFree(swept);cudaFree(tofs);
        cudaFree(value);cudaFree(next);cudaFree(departure);cudaFree(ok);cudaFree(swept_ok);
        cudaFree(shifts);cudaFree(back_camp);cudaFree(back_leg);cudaFree(params);
        cudaFree(output);delete[] host_output;cudaFree(samples);
        cudaFree(controls);
        cudaFree(visits);cudaFree(policy);
        if(stream)cudaStreamDestroy(stream);
    }
};
template<class T> bool allocate(T*& p,size_t n) {return cudaMalloc(&p,n*sizeof(T))==cudaSuccess;}
bool allocate_output(Workspace* w) {
    const PathLayout layout(w->stages);
    if(!allocate(w->output,layout.bytes))return false;
    w->host_output=new(std::nothrow) uint8_t[layout.bytes];
    if(!w->host_output)return false;
    w->result=reinterpret_cast<Result*>(w->output);
    w->arrivals=reinterpret_cast<int32_t*>(w->output+layout.arrivals);
    w->departures=reinterpret_cast<int32_t*>(w->output+layout.departures);
    w->path_dv=reinterpret_cast<double*>(w->output+layout.dv);
    w->path_swept=reinterpret_cast<double*>(w->output+layout.swept);
    w->path_swept_ok=w->output+layout.ok;
    return true;
}
template<class T> bool upload(T* dst,const T* src,size_t n,cudaStream_t stream) {
    return cudaMemcpyAsync(dst,src,n*sizeof(T),cudaMemcpyHostToDevice,stream)==cudaSuccess;
}
__global__ void sweep_grid(int n,int nt,const double* dv,const SweepCell* samples,int count,
    int reach,double* inflation,uint8_t* ok) {
    const size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i>=size_t(n)*nt)return;
    if(!count){inflation[i]=NAN;ok[i]=1;return;}
    const int departure=int(i/nt),tof=int(i%nt);
    int nearest=0;int64_t best=INT64_MAX;
    for(int j=0;j<count;++j) {
        const int64_t d=int64_t(departure)-samples[j].departure,t=int64_t(tof)-samples[j].tof;
        const int64_t distance=d*d+t*t;
        if(distance<best){best=distance;nearest=j;}
    }
    const auto cell=samples[nearest];
    const double lambert=dv[size_t(cell.departure)*nt+cell.tof];
    const double ratio=cell.certified&&lambert>1e-9?cell.delta_v/lambert:NAN;
    const bool accepted=best<=int64_t(reach)*reach&&cell.certified&&isfinite(ratio);
    inflation[i]=accepted?ratio:NAN;ok[i]=accepted;
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
__device__ void forward_mass(int stages,const Stage* params,const ForwardVisit* visits,
    const ForwardPolicy* policy,const Controls* controls,const double* epochs,
    const int32_t* arrivals,const int32_t* departures,const double* tofs,
    const double* dv,const double* swept,const uint8_t* ok,ForwardResult* result,
    double* masses,double* inflations,double* collected) {
    const auto p=*policy;const auto c=*controls;
    *result={0,0,0.0,p.initial_mass};
    for(int j=0;j<=stages;++j)collected[j]=0;
    for(int j=0;j<stages;++j){masses[j]=0;inflations[j]=0;}
    // Match CPU failure ordering: missing deployers are checked before stays.
    for(int j=0;j<=stages;++j)if(visits[j].collect&&visits[j].donor==-2){result->failure=1;return;}
    for(int j=0;j<=stages;++j)if(visits[j].collect) {
        const auto v=visits[j];
        const double deployed=v.donor<0?v.foreign_epoch:epochs[arrivals[v.donor]];
        const double stay=epochs[departures[j]]-deployed;
        if(stay<p.minimum_stay-1e-6){result->failure=2;return;}
    }
    double payload=0;
    for(int j=0;j<=stages;++j)if(visits[j].collect) {
        const auto v=visits[j];
        const double stay=epochs[departures[j]]-(v.donor<0?v.foreign_epoch:epochs[arrivals[v.donor]]);
        if(!isfinite(stay)||stay<0){result->failure=7;return;}
        collected[j]=p.mining_rate*stay/p.year_days;payload+=collected[j];
    }
    double mass=p.initial_mass,total=0;
    for(int j=0;j<stages;++j) {
        const auto s=params[j];
        if(visits[j].collect)mass+=collected[j];
        const double tof=epochs[arrivals[j+1]]-epochs[departures[j]];
        const double k=nearbyint(tof/p.step)-nearbyint(tofs[s.tof_offset]/p.step);
        if(k<0||k>=s.tofs){result->failure=3;result->mass_count=0;return;}
        if(!isfinite(dv[j])||!ok[j]){result->failure=4;result->mass_count=0;return;}
        const bool measured=!isnan(swept[j]);
        const double authority=c.thrust/mass*1e-3*tof*86400.;
        if(!measured&&dv[j]/authority>s.ratio_limit) {
            masses[j]=mass;result->mass_count=j+1;result->failure=5;return;
        }
        masses[j]=mass;result->mass_count=j+1;
        const double ratio=dv[j]/fmax(authority,1e-12);
        double inflation=s.flat;
        if(s.model==1)inflation=(s.floor+s.slope*ratio)*s.calibration;
        if(s.model==2)inflation=fmax(return_base(tof)*fmin(fmax(1+.6*(ratio-.33),.85),1.2),.85)*s.calibration;
        if(measured)inflation=swept[j];
        inflations[j]=inflation;
        const double propellant=mass*(1-exp(-(dv[j]*inflation)/c.exhaust));
        total+=propellant;mass-=propellant;
        if(visits[j+1].deploy)mass-=p.miner_mass;
    }
    result->propellant=total;result->final_mass=mass;
    if(mass<p.dry_mass+payload-1e-9)result->failure=6;
}
__global__ void leg(const Stage* params,int stage,int n,const double* dv,
    const uint8_t* feasible,const double* swept,const uint8_t* swept_ok,
    const double* tofs,const int32_t* shifts,const double* departure,
    const Controls* controls,double* next,int32_t* back) {
    const size_t index=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    // A full warp owns one arrival. Candidates are independent; only the
    // winning value/index is reduced, so no floating-point sum is reordered.
    const int lane=threadIdx.x&31;
    const size_t arrival=index/32;
    if(arrival>=size_t(n))return;const int a=int(arrival);
    const auto s=params[stage];const auto control=*controls;
    const double price=control.price,thrust=control.thrust,exhaust=control.exhaust;
    // Preserve the old prefix break even for a caller's unsorted shift array.
    int cutoff=s.tofs;
    for(int64_t k=lane;k<s.tofs;k+=32)if(shifts[s.tof_offset+k]>=n)cutoff=min(cutoff,int(k));
    for(int offset=16;offset;offset/=2)cutoff=min(cutoff,__shfl_down_sync(0xffffffff,cutoff,offset));
    cutoff=__shfl_sync(0xffffffff,cutoff,0);
    double best=-INFINITY;int winner=INT32_MAX;
    if(s.pinned_next<0||a==s.pinned_next)for(int64_t k=lane;k<cutoff;k+=32) {
        const int shift=shifts[s.tof_offset+k];
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
        if(candidate>best){best=candidate;winner=int(k);}
    }
    for(int offset=16;offset;offset/=2) {
        const double other=__shfl_down_sync(0xffffffff,best,offset);
        const int key=__shfl_down_sync(0xffffffff,winner,offset);
        if(other>best||(other==best&&key<winner)){best=other;winner=key;}
    }
    if(lane==0) {
        next[a]=best;
        back[stage*n+a]=winner==INT32_MAX?0:a-shifts[s.tof_offset+winner];
    }
}
__global__ void finish(const double* value,int n,int stages,const int32_t* camp_back,
    const int32_t* leg_back,int32_t* arrivals,int32_t* departures,Result* result,
    const Stage* params,const double* dv,const int32_t* shifts,double* path_dv,
    const double* swept,const uint8_t* swept_ok,double* path_swept,uint8_t* path_swept_ok,
    const Controls* controls,const ForwardVisit* visits,const ForwardPolicy* policy,
    const double* epochs,const double* tofs,ForwardResult* forward,double* masses,
    double* inflations,double* collected) {
    // Only the final scalar choice and O(stages) path reconstruction are serial.
    double best=-INFINITY;int winner=0;
    for(int i=0;i<n;++i)if(value[i]>best){best=value[i];winner=i;}
    *result={best,isfinite(best)?1:0,0};
    arrivals[stages]=departures[stages]=winner;
    for(int j=stages-1;j>=0;--j) {
        departures[j]=leg_back[j*n+arrivals[j+1]];
        arrivals[j]=camp_back[j*n+departures[j]];
        path_dv[j]=INFINITY;
        path_swept[j]=NAN;path_swept_ok[j]=0;
        if(isfinite(best)) {
            const auto s=params[j];
            const int shift=arrivals[j+1]-departures[j];
            for(int k=0;k<s.tofs;++k)if(shifts[s.tof_offset+k]==shift) {
                const int cell=s.cell_offset+departures[j]*s.tofs+k;
                path_dv[j]=dv[cell];path_swept[j]=swept[cell];path_swept_ok[j]=swept_ok[cell];break;
            }
        }
    }
    forward->failure=-1;
    if(controls->forward) {
        if(!isfinite(best)){*forward={8,0,0,0};return;}
        forward_mass(stages,params,visits,policy,controls,epochs,arrivals,departures,tofs,
            path_dv,path_swept,path_swept_ok,forward,masses,inflations,collected);
    }
}

__global__ void init_driver(DriverState* state,const Controls* controls) {
    *state={};state->result.objective=-INFINITY;state->best_price=controls->price;
    state->lo=state->hi=NAN;state->result.price_rounds=1;
}
__device__ bool valid_next_driver(int stages,const Stage* params,const Controls* controls) {
    if(!isfinite(controls->price)||controls->price<0)return false;
    for(int j=0;j<stages;++j)if(!isfinite(params[j].mass)||params[j].mass<=0)return false;
    return true;
}
__global__ void advance_driver(int stages,Stage* params,Controls* controls,
    const ForwardVisit* visits,const ForwardPolicy* forward_policy,const DriverPolicy* policy,
    const DriverWeight* weights,DriverState* state,const double* epochs,
    const int32_t* arrivals,const ForwardResult* forward,const double* masses,
    const double* collected,const uint8_t* output,uint8_t* best,size_t bytes,
    cudaGraphConditionalHandle condition) {
    auto& r=state->result;const auto p=*policy;
    ++r.evaluations;++state->round;r.mass_rounds=max(r.mass_rounds,state->round);
    const int failure=forward->failure;r.failure=failure;
    if(failure==5&&forward->mass_count>0) {
        const int count=forward->mass_count;
        const double scale=masses[count-1]/fmax(params[count-1].mass,1e-9);
        for(int j=0;j<stages;++j)
            params[j].mass=j<count?masses[j]:fmax(params[j].mass,params[j].mass*scale);
        if(state->round<p.max_masses) {
            if(!valid_next_driver(stages,params,controls)){r.failure=9;cudaGraphSetConditional(condition,0);}
            return;
        }
    }
    bool stop=false,geometric=false;
    if(!failure) {
        double weighted=0,orphan=0;
        // Preserve collection insertion order and the separately accumulated
        // deploy-order orphan credit used by plan_value on the CPU.
        for(int j=0;j<=stages;++j)if(visits[j].collect)weighted+=weights[j].weight*collected[j];
        if(p.orphan_credit>0)for(int j=0;j<=stages;++j)if(weights[j].orphan) {
            const double stay=p.mission_end-p.orphan_margin-epochs[arrivals[j]];
            if(stay>=forward_policy->minimum_stay)
                orphan+=weights[j].weight*(forward_policy->mining_rate*stay/forward_policy->year_days);
        }
        const double objective=weighted+p.orphan_credit*orphan;
        if(objective>r.objective) {
            r.feasible=1;r.objective=objective;state->best_price=controls->price;
            for(size_t j=0;j<bytes;++j)best[j]=output[j];
        }
        state->hi=isnan(state->hi)?controls->price:fmin(state->hi,controls->price);
        if(isnan(state->lo)){controls->price/=p.price_growth;geometric=true;}
    } else if(failure==5||failure==6) {
        state->lo=controls->price;
        if(isnan(state->hi)){controls->price*=p.price_growth;geometric=true;}
    } else stop=true;
    if(!stop&&!geometric) {
        if(state->bisections>=2)stop=true;
        else {++state->bisections;controls->price=.5*(state->lo+state->hi);}
    }
    if(r.price_rounds>=p.max_prices)stop=true;
    if(!stop&&!valid_next_driver(stages,params,controls)){r.failure=9;stop=true;}
    if(stop)cudaGraphSetConditional(condition,0);
    else {++r.price_rounds;state->round=0;}
}
__global__ void finish_driver(int stages,const Stage* params,const Controls* controls,
    const DriverState* state,uint8_t* best,size_t bytes) {
    auto* summary=reinterpret_cast<DriverResult*>(best+bytes);
    *summary=state->result;
    summary->price=summary->feasible?state->best_price:controls->price;
    if(summary->feasible&&summary->failure!=7&&summary->failure!=9)summary->failure=0;
    else {*reinterpret_cast<Result*>(best)={-INFINITY,0,0};}
    auto* profile=reinterpret_cast<double*>(best+bytes+sizeof(DriverResult));
    for(int j=0;j<stages;++j)profile[j]=params[j].mass;
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
        &&allocate(w->back_leg,size_t(n)*stages)&&allocate(w->params,stages)&&allocate_output(w)
        &&allocate(w->controls,1)&&allocate(w->visits,size_t(stages)+1)&&allocate(w->policy,1);
    // Initialize padding/unused optional output fields once before any download.
    good=good&&cudaMemsetAsync(w->output,0,PathLayout(stages).bytes,w->stream)==cudaSuccess;
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
extern "C" int spacepdhcg_gtoc12_retime_set_sweep(void* workspace,int32_t offset,int32_t nt,
    int32_t count,const SweepCell* samples,int32_t reach) {
    auto* w=static_cast<Workspace*>(workspace);
    if(!w||offset<0||nt<=0||int64_t(offset)+int64_t(w->n)*nt>w->cells||count<0||reach<0
        ||(count&&!samples))return 1;
    std::unique_lock<std::mutex> lock(w->mutex,std::try_to_lock);if(!lock.owns_lock())return 3;
    int device=-1;if(cudaGetDevice(&device)!=cudaSuccess)return 2;if(device!=w->device)return 1;
    for(int j=0;j<count;++j)if(samples[j].departure<0||samples[j].departure>=w->n
        ||samples[j].tof<0||samples[j].tof>=nt||(samples[j].certified!=0&&samples[j].certified!=1))return 1;
    if(count>w->sample_capacity) {
        SweepCell* fresh=nullptr;if(!allocate(fresh,count))return 2;
        cudaFree(w->samples);w->samples=fresh;w->sample_capacity=count;
    }
    bool good=!count||upload(w->samples,samples,count,w->stream);
    if(good) {
        sweep_grid<<<(unsigned(w->n*nt)+127)/128,128,0,w->stream>>>(w->n,nt,w->dv+offset,w->samples,count,reach,w->swept+offset,w->swept_ok+offset);
        good=cudaGetLastError()==cudaSuccess;
    }
    const auto done=cudaStreamSynchronize(w->stream);return good&&done==cudaSuccess?0:2;
}
extern "C" int spacepdhcg_gtoc12_retime_read_sweep(void* workspace,int32_t offset,int32_t nt,
    double* inflation,uint8_t* feasible) {
    auto* w=static_cast<Workspace*>(workspace);
    if(!w||offset<0||nt<=0||int64_t(offset)+int64_t(w->n)*nt>w->cells||!inflation||!feasible)return 1;
    std::unique_lock<std::mutex> lock(w->mutex,std::try_to_lock);if(!lock.owns_lock())return 3;
    int device=-1;if(cudaGetDevice(&device)!=cudaSuccess)return 2;if(device!=w->device)return 1;
    const size_t cells=size_t(w->n)*nt;
    const bool good=cudaMemcpyAsync(inflation,w->swept+offset,cells*sizeof(double),cudaMemcpyDeviceToHost,w->stream)==cudaSuccess
        &&cudaMemcpyAsync(feasible,w->swept_ok+offset,cells,cudaMemcpyDeviceToHost,w->stream)==cudaSuccess;
    const auto done=cudaStreamSynchronize(w->stream);return good&&done==cudaSuccess?0:2;
}
// This sequence uses only retained device addresses. Policies, masses and
// controls are uploaded before replay, so a graph never captures stale values.
static bool enqueue_dp(Workspace* w) {
    bool good=cudaMemsetAsync(w->value,0,w->n*sizeof(double),w->stream)==cudaSuccess;
    auto* value=w->value;auto* next=w->next;
    for(int j=0;good&&j<w->stages;++j) {
        camp<<<(unsigned(w->n)+127)/128,128,0,w->stream>>>(w->params,j,w->epochs,w->n,value,w->departure,w->back_camp);
        good=cudaGetLastError()==cudaSuccess;if(!good)break;
        leg<<<(unsigned(w->n)+3)/4,128,0,w->stream>>>(w->params,j,w->n,w->dv,w->ok,w->swept,w->swept_ok,
            w->tofs,w->shifts,w->departure,w->controls,next,w->back_leg);
        good=cudaGetLastError()==cudaSuccess;std::swap(value,next);
    }
    if(good) {
        const PathLayout layout(w->stages);
        finish<<<1,1,0,w->stream>>>(value,w->n,w->stages,w->back_camp,w->back_leg,w->arrivals,w->departures,w->result,w->params,w->dv,w->shifts,w->path_dv,w->swept,w->swept_ok,w->path_swept,w->path_swept_ok,
            w->controls,w->visits,w->policy,w->epochs,w->tofs,
            reinterpret_cast<ForwardResult*>(w->output+layout.forward),
            reinterpret_cast<double*>(w->output+layout.masses),
            reinterpret_cast<double*>(w->output+layout.inflations),
            reinterpret_cast<double*>(w->output+layout.collected));
        good=cudaGetLastError()==cudaSuccess;
    }
    return good;
}
static bool enqueue_graph(Workspace* w) {
    if(!w->graph_exec) {
        if(cudaStreamBeginCapture(w->stream,cudaStreamCaptureModeThreadLocal)!=cudaSuccess)return false;
        const bool recorded=enqueue_dp(w);
        cudaGraph_t graph=nullptr;
        const auto ended=cudaStreamEndCapture(w->stream,&graph);
        if(!recorded||ended!=cudaSuccess){if(graph)cudaGraphDestroy(graph);return false;}
        cudaGraphExec_t executable=nullptr;
        if(cudaGraphInstantiate(&executable,graph,0)!=cudaSuccess){cudaGraphDestroy(graph);return false;}
        w->graph=graph;w->graph_exec=executable;++w->graph_builds;
    }
    if(cudaGraphLaunch(w->graph_exec,w->stream)!=cudaSuccess)return false;
    ++w->graph_launches;return true;
}
static bool prepare_driver(Workspace* w) {
    const PathLayout layout(w->stages);
    const size_t bytes=layout.bytes+sizeof(DriverResult)+size_t(w->stages)*sizeof(double);
    if(!w->driver_exec) {
        if(!w->driver_policy&&!allocate(w->driver_policy,1))return false;
        if(!w->weights&&!allocate(w->weights,size_t(w->stages)+1))return false;
        if(!w->driver_state&&!allocate(w->driver_state,1))return false;
        if(!w->best_output) {
            if(!allocate(w->best_output,bytes))return false;
            if(cudaMemsetAsync(w->best_output,0,bytes,w->stream)!=cudaSuccess)return false;
        }
        if(!w->host_best)w->host_best=new(std::nothrow) uint8_t[bytes];
        if(!w->host_best)return false;
        cudaGraph_t graph=nullptr;
        if(cudaGraphCreate(&graph,0)!=cudaSuccess)return false;
        cudaGraphConditionalHandle handle;
        if(cudaGraphConditionalHandleCreate(&handle,graph,1,cudaGraphCondAssignDefault)!=cudaSuccess){cudaGraphDestroy(graph);return false;}
        if(cudaStreamBeginCaptureToGraph(w->stream,graph,nullptr,nullptr,0,cudaStreamCaptureModeThreadLocal)!=cudaSuccess){cudaGraphDestroy(graph);return false;}
        init_driver<<<1,1,0,w->stream>>>(w->driver_state,w->controls);
        bool good=cudaGetLastError()==cudaSuccess;
        good=(cudaStreamEndCapture(w->stream,nullptr)==cudaSuccess)&&good;
        cudaGraphNode_t init=nullptr;size_t count=1;
        good=good&&cudaGraphGetNodes(graph,&init,&count)==cudaSuccess&&count==1;
        cudaGraphNodeParams node{};node.type=cudaGraphNodeTypeConditional;
        node.conditional.handle=handle;node.conditional.type=cudaGraphCondTypeWhile;node.conditional.size=1;
        cudaGraphNode_t loop=nullptr;
        good=good&&cudaGraphAddNode(&loop,graph,&init,1,&node)==cudaSuccess;
        if(!good){cudaGraphDestroy(graph);return false;}
        if(cudaStreamBeginCaptureToGraph(w->stream,node.conditional.phGraph_out[0],nullptr,nullptr,0,cudaStreamCaptureModeThreadLocal)!=cudaSuccess){cudaGraphDestroy(graph);return false;}
        good=enqueue_dp(w);
        if(good) {
            advance_driver<<<1,1,0,w->stream>>>(w->stages,w->params,w->controls,w->visits,w->policy,
                w->driver_policy,w->weights,w->driver_state,w->epochs,w->arrivals,
                reinterpret_cast<ForwardResult*>(w->output+layout.forward),
                reinterpret_cast<double*>(w->output+layout.masses),
                reinterpret_cast<double*>(w->output+layout.collected),w->output,w->best_output,layout.bytes,handle);
            good=cudaGetLastError()==cudaSuccess;
        }
        good=(cudaStreamEndCapture(w->stream,nullptr)==cudaSuccess)&&good;
        if(!good){cudaGraphDestroy(graph);return false;}
        if(cudaStreamBeginCaptureToGraph(w->stream,graph,&loop,nullptr,1,cudaStreamCaptureModeThreadLocal)!=cudaSuccess){cudaGraphDestroy(graph);return false;}
        finish_driver<<<1,1,0,w->stream>>>(w->stages,w->params,w->controls,w->driver_state,w->best_output,layout.bytes);
        good=cudaGetLastError()==cudaSuccess;
        good=(cudaStreamEndCapture(w->stream,nullptr)==cudaSuccess)&&good;
        cudaGraphExec_t exec=nullptr;
        good=good&&cudaGraphInstantiate(&exec,graph,0)==cudaSuccess;
        if(!good){cudaGraphDestroy(graph);return false;}
        w->driver_graph=graph;w->driver_exec=exec;
    }
    return true;
}
extern "C" int spacepdhcg_gtoc12_retime_set_graph(void* workspace,int32_t enabled) {
    auto* w=static_cast<Workspace*>(workspace);if(!w||(enabled!=0&&enabled!=1))return 1;
    std::unique_lock<std::mutex> lock(w->mutex,std::try_to_lock);if(!lock.owns_lock())return 3;
    w->use_graph=enabled;return 0;
}
extern "C" int spacepdhcg_gtoc12_retime_graph_stats(void* workspace,uint64_t* builds,uint64_t* launches) {
    auto* w=static_cast<Workspace*>(workspace);if(!w||!builds||!launches)return 1;
    std::unique_lock<std::mutex> lock(w->mutex,std::try_to_lock);if(!lock.owns_lock())return 3;
    *builds=w->graph_builds;*launches=w->graph_launches;return 0;
}
static int evaluate_path(void* workspace,const Stage* params,
    double price,double thrust,double exhaust,int32_t* arrivals,int32_t* departures,
    double* objective,int32_t* feasible,double* path_dv,double* path_swept,uint8_t* path_swept_ok,
    const ForwardVisit* visits=nullptr,const ForwardPolicy* policy=nullptr,
    ForwardResult* forward=nullptr,double* masses=nullptr,double* inflations=nullptr,double* collected=nullptr,
    const DriverPolicy* driver=nullptr,const DriverWeight* weights=nullptr,
    DriverResult* summary=nullptr,double* final_profile=nullptr) {
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
    if(visits) {
        if(!policy||!forward||!masses||!inflations||!collected)return 1;
        const double values[]={policy->initial_mass,policy->minimum_stay,policy->mining_rate,
            policy->year_days,policy->miner_mass,policy->dry_mass,policy->step};
        for(double v:values)if(!std::isfinite(v)||v<=0)return 1;
        for(int j=0;j<=w->stages;++j)if((visits[j].deploy!=0&&visits[j].deploy!=1)
            ||(visits[j].collect!=0&&visits[j].collect!=1)||visits[j].donor<-2||visits[j].donor>j)return 1;
    }
    if(driver) {
        if(!visits||!weights||!summary||!final_profile||driver->max_prices<=0||driver->max_masses<=0
            ||int64_t(driver->max_prices)*driver->max_masses>INT32_MAX
            ||!std::isfinite(driver->price_growth)||driver->price_growth<=0
            ||!std::isfinite(driver->orphan_credit)||!std::isfinite(driver->orphan_margin)
            ||!std::isfinite(driver->mission_end))return 1;
        for(int j=0;j<=w->stages;++j)if(!std::isfinite(weights[j].weight)
            ||(weights[j].orphan!=0&&weights[j].orphan!=1))return 1;
        if(!prepare_driver(w))return 2;
    }
    const Controls controls{price,thrust,exhaust,visits?1:0,0};
    bool good=upload(w->params,params,w->stages,w->stream)
        &&upload(w->controls,&controls,1,w->stream);
    good=good&&(!visits||(upload(w->visits,visits,size_t(w->stages)+1,w->stream)
        &&upload(w->policy,policy,1,w->stream)));
    if(driver) {
        good=good&&upload(w->driver_policy,driver,1,w->stream)
            &&upload(w->weights,weights,size_t(w->stages)+1,w->stream)
            &&cudaGraphLaunch(w->driver_exec,w->stream)==cudaSuccess;
    } else good=good&&(w->use_graph?enqueue_graph(w):enqueue_dp(w));
    Result result{};
    const PathLayout layout(w->stages);
    auto* host=driver?w->host_best:w->host_output;
    const size_t bytes=layout.bytes+(driver?sizeof(DriverResult)+size_t(w->stages)*sizeof(double):0);
    good=good&&cudaMemcpyAsync(host,driver?w->best_output:w->output,bytes,cudaMemcpyDeviceToHost,w->stream)==cudaSuccess;
    const auto done=cudaStreamSynchronize(w->stream);
    if(!good||done!=cudaSuccess)return 2;
    // One contiguous transfer; caller buffers are touched only after it
    // completes. The legacy APIs can omit path fields without changing layout.
    std::memcpy(&result,host,sizeof(result));
    std::memcpy(arrivals,host+layout.arrivals,(size_t(w->stages)+1)*sizeof(int32_t));
    std::memcpy(departures,host+layout.departures,(size_t(w->stages)+1)*sizeof(int32_t));
    if(path_dv)std::memcpy(path_dv,host+layout.dv,size_t(w->stages)*sizeof(double));
    if(path_swept)std::memcpy(path_swept,host+layout.swept,size_t(w->stages)*sizeof(double));
    if(path_swept_ok)std::memcpy(path_swept_ok,host+layout.ok,w->stages);
    if(visits) {
        std::memcpy(forward,host+layout.forward,sizeof(ForwardResult));
        std::memcpy(masses,host+layout.masses,size_t(w->stages)*sizeof(double));
        std::memcpy(inflations,host+layout.inflations,size_t(w->stages)*sizeof(double));
        std::memcpy(collected,host+layout.collected,(size_t(w->stages)+1)*sizeof(double));
    }
    if(driver) {
        std::memcpy(summary,host+layout.bytes,sizeof(DriverResult));
        std::memcpy(final_profile,host+layout.bytes+sizeof(DriverResult),size_t(w->stages)*sizeof(double));
        if(summary->failure==9)return 1;
    }
    *objective=result.objective;*feasible=result.feasible;return 0;
}
extern "C" int spacepdhcg_gtoc12_retime_host(void* w,const Stage* p,double price,double thrust,
    double exhaust,int32_t* a,int32_t* d,double* objective,int32_t* feasible) {
    return evaluate_path(w,p,price,thrust,exhaust,a,d,objective,feasible,nullptr,nullptr,nullptr);
}
extern "C" int spacepdhcg_gtoc12_retime_path_host(void* w,const Stage* p,double price,double thrust,
    double exhaust,int32_t* a,int32_t* d,double* objective,int32_t* feasible,double* dv) {
    if(!dv)return 1;
    return evaluate_path(w,p,price,thrust,exhaust,a,d,objective,feasible,dv,nullptr,nullptr);
}
extern "C" int spacepdhcg_gtoc12_retime_swept_path_host(void* w,const Stage* p,double price,double thrust,
    double exhaust,int32_t* a,int32_t* d,double* objective,int32_t* feasible,double* dv,double* swept,uint8_t* ok) {
    if(!dv||!swept||!ok)return 1;
    return evaluate_path(w,p,price,thrust,exhaust,a,d,objective,feasible,dv,swept,ok);
}
extern "C" int spacepdhcg_gtoc12_retime_forward_host(void* w,const Stage* p,double price,double thrust,
    double exhaust,int32_t* a,int32_t* d,double* objective,int32_t* feasible,double* dv,double* swept,uint8_t* ok,
    const ForwardVisit* visits,const ForwardPolicy* policy,ForwardResult* result,double* masses,double* inflations,double* collected) {
    if(!dv||!swept||!ok||!visits||!policy||!result||!masses||!inflations||!collected)return 1;
    return evaluate_path(w,p,price,thrust,exhaust,a,d,objective,feasible,dv,swept,ok,visits,policy,result,masses,inflations,collected);
}
extern "C" int spacepdhcg_gtoc12_retime_order_host(void* w,const Stage* p,double price,double thrust,
    double exhaust,int32_t* a,int32_t* d,double* objective,int32_t* feasible,double* dv,double* swept,uint8_t* ok,
    const ForwardVisit* visits,const ForwardPolicy* policy,ForwardResult* result,double* masses,double* inflations,double* collected,
    const DriverPolicy* driver,const DriverWeight* weights,DriverResult* summary,double* final_profile) {
    if(!dv||!swept||!ok||!visits||!policy||!result||!masses||!inflations||!collected||!driver||!weights||!summary||!final_profile)return 1;
    return evaluate_path(w,p,price,thrust,exhaust,a,d,objective,feasible,dv,swept,ok,visits,policy,result,masses,inflations,collected,driver,weights,summary,final_profile);
}
extern "C" int spacepdhcg_gtoc12_retime_destroy(void** workspace) {
    if(!workspace)return 1;auto* w=static_cast<Workspace*>(*workspace);if(!w)return 0;
    if(cudaSetDevice(w->device)!=cudaSuccess)return 2;
    {std::unique_lock<std::mutex> lock(w->mutex,std::try_to_lock);if(!lock.owns_lock())return 3;}
    delete w;*workspace=nullptr;return 0;
}
