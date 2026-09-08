#pragma once
#include "spacepdhcg/cuda/gtoc12_fleet_c_api.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <vector>

namespace gtoc12_fleet {
using Column=spacepdhcg_gtoc12_fleet_column;
using Report=spacepdhcg_gtoc12_fleet_report;
using ExchangeReport=spacepdhcg_gtoc12_fleet_exchange_report;
static_assert(sizeof(Column)==32&&sizeof(Report)==40);
static_assert(sizeof(ExchangeReport)==16);
struct Row {double value;uint64_t nodes;int exhaustive;};
struct Problem {int n,max_ships;const Column* c;const int *co,*ci,*ro,*po,*pi;};
__device__ bool conflicts(const Problem& p,int i,const uint8_t* selected) {
    for(int k=p.co[i];k<p.co[i+1];++k)if(selected[p.ci[k]])return true;
    return false;
}
__device__ bool supplied(const Problem& p,int i,const uint8_t* selected) {
    for(int g=p.ro[i];g<p.ro[i+1];++g) {
        bool found=false;
        for(int k=p.po[g];k<p.po[g+1];++k)if(selected[p.pi[k]]){found=true;break;}
        if(!found)return false;
    }
    return true;
}
__device__ bool feasible(const Problem& p,const uint8_t* selected,int& ships,double& mass,double& value) {
    ships=0;mass=value=0;
    bool valid=true;
    for(int i=0;i<p.n;++i)if(selected[i]) {
        ships+=p.c[i].ships;mass+=p.c[i].mass;value+=p.c[i].value;
        if(conflicts(p,i,selected)||!supplied(p,i,selected))valid=false;
    }
    return valid&&ships<=p.max_ships&&(!ships||ships<=fmin(100.0,2.0*exp(.004*mass/ships))+1e-9);
}
__device__ void copy(int n,uint8_t* dst,const uint8_t* src) {for(int i=0;i<n;++i)dst[i]=src[i];}
// A selected fleet contains at most 100 columns. Counts therefore fit in a byte.
// Updating adjacent columns on each push/pop replaces repeated CSR scans at every node.
__device__ void mark(const Problem& p,int i,uint8_t* blocked,int delta) {
    for(int k=p.co[i];k<p.co[i+1];++k)blocked[p.ci[k]]+=delta;
}
__global__ void rank_columns(Problem p,int* order) {
    const int i=int(blockIdx.x*blockDim.x+threadIdx.x);if(i>=p.n)return;
    for(int mode=0;mode<2;++mode) {
        const double a=mode?p.c[i].value/p.c[i].ships:p.c[i].value;int rank=0;
        for(int j=0;j<p.n;++j) {
            const double b=mode?p.c[j].value/p.c[j].ships:p.c[j].value;
            if(b>a||(b==a&&(p.c[j].value>p.c[i].value||(p.c[j].value==p.c[i].value&&
                (p.c[j].identifier<p.c[i].identifier||(p.c[j].identifier==p.c[i].identifier&&j<i))))))++rank;
        }
        order[mode*p.n+rank]=i;
    }
}
__global__ void seed(Problem p,const int* order,const uint8_t* warm,uint8_t* masks,Row* rows) {
    const int mode=int(threadIdx.x);if(mode>=3)return;
    auto* chosen=masks+size_t(mode)*p.n;
    for(int i=0;i<p.n;++i)chosen[i]=mode==2?warm[i]:0;
    if(mode<2) {
        int ships=0;
        for(int k=0;k<p.n;++k) {
            const int i=order[mode*p.n+k];
            if(ships+p.c[i].ships<=p.max_ships&&!conflicts(p,i,chosen)){chosen[i]=1;ships+=p.c[i].ships;}
        }
        for(;;) {
            double mass,value;int count;
            if(feasible(p,chosen,count,mass,value))break;
            bool stranded=false;
            for(int i=0;i<p.n;++i)if(chosen[i]&&!supplied(p,i,chosen))stranded=true;
            int victim=-1;
            for(int i=0;i<p.n;++i)if(chosen[i]&&(!stranded||!supplied(p,i,chosen))) {
                if(victim<0||p.c[i].mass/p.c[i].ships<p.c[victim].mass/p.c[victim].ships||
                   (p.c[i].mass/p.c[i].ships==p.c[victim].mass/p.c[victim].ships&&p.c[i].identifier>p.c[victim].identifier))victim=i;
            }
            if(victim<0)break;
            chosen[victim]=0;
        }
    }
    double mass,value;int ships;
    if(!feasible(p,chosen,ships,mass,value)||value<0){for(int i=0;i<p.n;++i)chosen[i]=0;value=0;}
    rows[mode]={value,0,1};
}
struct Exchange {
    int ids[100],count,active;
    ExchangeReport report;
};
__global__ void start_exchange(Problem p,uint8_t* masks,Row* rows,Exchange* state) {
    if(threadIdx.x)return;
    int best=2;
    for(int i=0;i<2;++i)if(rows[i].value>rows[best].value+1e-9)best=i;
    auto* chosen=masks+size_t(2)*p.n;
    if(best!=2){copy(p.n,chosen,masks+size_t(best)*p.n);rows[2]=rows[best];}
    state->count=0;state->active=1;state->report={0,0,0};
    for(int i=0;i<p.n;++i)if(chosen[i])state->ids[state->count++]=i;
}
__device__ bool exchange_contains(int i,const uint8_t* chosen,int add,int remove) {
    return i==add||(chosen[i]&&i!=remove);
}
__device__ bool exchange_supplied(Problem p,int i,const uint8_t* chosen,int add,int remove) {
    for(int g=p.ro[i];g<p.ro[i+1];++g) {
        bool found=false;
        for(int k=p.po[g];k<p.po[g+1];++k)
            if(exchange_contains(p.pi[k],chosen,add,remove)){found=true;break;}
        if(!found)return false;
    }
    return true;
}
__global__ void evaluate_exchanges(Problem p,const uint8_t* masks,const Row* rows,
                                   const Exchange* state,double* values) {
    if(!state->active)return;
    const int t=int(blockIdx.x*blockDim.x+threadIdx.x);
    if(t>=(p.n+1)*101)return;
    values[t]=-INFINITY;
    const int add=t/101-1,slot=t%101-1;
    if(slot>=state->count||(slot<0&&add<0))return;
    const int remove=slot<0?-1:state->ids[slot];
    const auto* chosen=masks+size_t(2)*p.n;
    if(add>=0&&chosen[add])return;
    if((add<0?0:p.c[add].value)-(remove<0?0:p.c[remove].value)<=0)return;
    if(add>=0)for(int k=p.co[add];k<p.co[add+1];++k)
        if(chosen[p.ci[k]]&&p.ci[k]!=remove)return;
    int ships=0;double mass=0,value=0;bool inserted=add<0;
    // Sum in input order, just as the independent packing check does.
    for(int k=0;k<state->count;++k) {
        const int i=state->ids[k];
        if(!inserted&&add<i){ships+=p.c[add].ships;mass+=p.c[add].mass;value+=p.c[add].value;inserted=true;}
        if(i==remove)continue;
        ships+=p.c[i].ships;mass+=p.c[i].mass;value+=p.c[i].value;
        if(!exchange_supplied(p,i,chosen,add,remove))return;
    }
    if(!inserted){ships+=p.c[add].ships;mass+=p.c[add].mass;value+=p.c[add].value;}
    if(add>=0&&!exchange_supplied(p,add,chosen,add,remove))return;
    if(ships>p.max_ships||(ships&&ships>fmin(100.0,2.0*exp(.004*mass/ships))+1e-9))return;
    if(value>rows[2].value+1e-9)values[t]=value;
}
__global__ void accept_exchange(Problem p,uint8_t* masks,Row* rows,Exchange* state,
                                const double* values) {
    if(!state->active)return;
    __shared__ double best_values[256];
    __shared__ int best_indices[256];
    const int lane=int(threadIdx.x);double value=-INFINITY;int best=-1;
    for(int i=lane;i<(p.n+1)*101;i+=256)
        if(values[i]>value||(values[i]==value&&best>=0&&i<best)){value=values[i];best=i;}
    best_values[lane]=value;best_indices[lane]=best;__syncthreads();
    if(lane)return;
    for(int i=1;i<256;++i)if(best_values[i]>value||
        (best_values[i]==value&&best_indices[i]>=0&&(best<0||best_indices[i]<best))) {
        value=best_values[i];best=best_indices[i];
    }
    state->report.proposals+=uint64_t(p.n+1)*uint64_t(state->count+1)-1;
    ++state->report.rounds;
    if(best<0){state->active=0;return;}
    const int add=best/101-1,slot=best%101-1;
    auto* chosen=masks+size_t(2)*p.n;
    const int remove=slot<0?-1:state->ids[slot];
    if(remove>=0)chosen[remove]=0;
    if(add>=0)chosen[add]=1;
    int ships;double mass,checked;
    if(!feasible(p,chosen,ships,mass,checked)||checked<=rows[2].value+1e-9) {
        if(add>=0)chosen[add]=0;
        if(remove>=0)chosen[remove]=1;
        state->active=0;return;
    }
    rows[2].value=checked;state->count=0;++state->report.moves;
    for(int i=0;i<p.n;++i)if(chosen[i])state->ids[state->count++]=i;
}
__global__ void search(Problem p,int bits,int tasks,uint64_t cap,uint8_t* best_masks,
                      uint8_t* active,uint8_t* phase,uint8_t* exclusions,Row* rows) {
    const int t=int(blockIdx.x*blockDim.x+threadIdx.x);if(t>=tasks)return;
    const uint64_t budget=cap/uint64_t(tasks)+(uint64_t(t)<cap%uint64_t(tasks));
    uint8_t* chosen=active+size_t(t)*p.n;
    uint8_t* best=best_masks+size_t(t+3)*p.n;
    uint8_t* state=phase+size_t(t)*(p.n+1);
    uint8_t* blocked=exclusions+size_t(t)*p.n;
    int initial=0;for(int i=1;i<3;++i)if(rows[i].value>rows[initial].value+1e-9)initial=i;
    copy(p.n,best,best_masks+size_t(initial)*p.n);double best_value=rows[initial].value;
    for(int i=0;i<p.n;++i){chosen[i]=0;blocked[i]=0;}
    uint64_t nodes=0;bool exhaustive=true;
    int prefix_ships=0;
    for(int i=0;i<bits;++i)if(((t>>(bits-1-i))&1)==0) {
        if(prefix_ships+p.c[i].ships>p.max_ships||blocked[i]) {
            rows[t+3]={best_value,0,1};return;
        }
        chosen[i]=1;prefix_ships+=p.c[i].ships;mark(p,i,blocked,1);
    }
    int index=bits;bool enter=true;
    for(;;) {
        if(enter) {
            if(nodes==budget){exhaustive=false;break;}++nodes;
            int ships=0;double mass=0,value=0;bool supplied_all=true;
            for(int i=0;i<p.n;++i)if(chosen[i]) {
                ships+=p.c[i].ships;mass+=p.c[i].mass;value+=p.c[i].value;
                if(!supplied(p,i,chosen))supplied_all=false;
            }
            const bool valid=supplied_all&&(!ships||ships<=fmin(100.0,2.0*exp(.004*mass/ships))+1e-9);
            if(valid&&value>best_value+1e-9) {
                best_value=value;copy(p.n,best,chosen);
            }
            double bound=value;
            for(int j=index;j<p.n;++j)if(p.c[j].value>0&&!blocked[j])
                bound=__dadd_ru(bound,p.c[j].value);
            if(index==p.n||ships>=p.max_ships||bound<=best_value+1e-9)enter=false;
            else {
                const bool include=ships+p.c[index].ships<=p.max_ships&&!blocked[index];
                state[index]=include?1:2;chosen[index]=include;
                if(include)mark(p,index,blocked,1);
                ++index;continue;
            }
        }
        --index;if(index<bits)break;
        if(state[index]==1){chosen[index]=0;mark(p,index,blocked,-1);state[index]=2;++index;enter=true;}
        else enter=false;
    }
    rows[t+3]={best_value,nodes,int(exhaustive)};
}
__global__ void finish(Problem p,int tasks,const uint8_t* masks,const Row* rows,
                       uint8_t* output,Report* report) {
    if(threadIdx.x)return;
    int best=0;uint64_t nodes=0;bool exhaustive=true;
    for(int i=0;i<tasks+3;++i) {
        if(rows[i].value>rows[best].value+1e-9)best=i;
        if(i>=3){nodes+=rows[i].nodes;exhaustive=exhaustive&&rows[i].exhaustive;}
    }
    double bound=0;for(int i=0;i<p.n;++i)if(p.c[i].value>0)bound=__dadd_ru(bound,p.c[i].value);
    if(p.n==0)exhaustive=true;
    *report={rows[best].value,exhaustive?rows[best].value:fmax(bound,rows[best].value),
             fmax(rows[0].value,rows[1].value),nodes,int(exhaustive),tasks};
    copy(p.n,output,masks+size_t(best)*p.n);
}
struct Memory {
    std::vector<void*> pointers;cudaStream_t stream{};int device=-1;
    ~Memory(){
        int previous=-1;cudaGetDevice(&previous);
        if(device>=0&&previous!=device)cudaSetDevice(device);
        if(stream)cudaStreamSynchronize(stream);
        for(void* p:pointers)cudaFree(p);
        if(stream)cudaStreamDestroy(stream);
        if(previous>=0&&previous!=device)cudaSetDevice(previous);
    }
    template<class T> bool alloc(T*& out,size_t n) {
        if(cudaMalloc(&out,std::max(size_t(1),n)*sizeof(T))!=cudaSuccess)return false;
        pointers.push_back(out);return true;
    }
    template<class T> bool input(T*& out,const T* in,size_t n) {
        return alloc(out,n)&&(!n||cudaMemcpyAsync(out,in,n*sizeof(T),cudaMemcpyHostToDevice,stream)==cudaSuccess);
    }
};
inline bool offsets(const int32_t* p,int n) {
    if(!p||p[0]!=0)return false;for(int i=0;i<n;++i)if(p[i]<0||p[i+1]<p[i])return false;return true;
}
struct Workspace {
    Memory m;int n{},bits{},tasks{};
    Column* c{};int *dco{},*dci{},*dro{},*dpo{},*dpi{},*order{};
    uint8_t *dw{},*masks{},*active{},*phase{},*exclusions{},*out{};
    Row* rows{};Report* result{};Exchange* exchange{};double* proposal_values{};
    Problem problem(int max_ships) const {return {n,max_ships,c,dco,dci,dro,dpo,dpi};}
    bool init(int count,int prefix,const Column* columns,const int32_t* co,const int32_t* ci,
              const int32_t* ro,const int32_t* po,const int32_t* pi) {
        n=count;bits=std::min(prefix,n);tasks=1<<bits;m.pointers.reserve(18);
        if(cudaGetDevice(&m.device)!=cudaSuccess||
           cudaStreamCreateWithFlags(&m.stream,cudaStreamNonBlocking)!=cudaSuccess)return false;
        if(!m.input(c,columns,n)||!m.input(dco,co,n+1)||!m.input(dci,ci,co[n])||
           !m.input(dro,ro,n+1)||!m.input(dpo,po,ro[n]+1)||!m.input(dpi,pi,po[ro[n]])||
           !m.alloc(dw,n)||!m.alloc(order,2*size_t(n))||!m.alloc(masks,size_t(tasks+3)*n)||
           !m.alloc(active,size_t(tasks)*n)||!m.alloc(phase,size_t(tasks)*(n+1))||
           !m.alloc(exclusions,size_t(tasks)*n)||!m.alloc(out,n)||!m.alloc(rows,tasks+3)||
           !m.alloc(result,1)||!m.alloc(exchange,1)||!m.alloc(proposal_values,size_t(n+1)*101))return false;
        if(n)rank_columns<<<(n+127)/128,128,0,m.stream>>>(problem(100),order);
        return cudaGetLastError()==cudaSuccess&&cudaStreamSynchronize(m.stream)==cudaSuccess;
    }
    int solve(int max_ships,uint64_t cap,const uint8_t* warm,uint8_t* selected,
              Report* report,int rounds,ExchangeReport* exchange_report) {
        int device=-1;
        if(max_ships<0||max_ships>100||rounds<0||rounds>100||!report||!exchange_report||
           (n&&(!warm||!selected)))return 1;
        if(cudaGetDevice(&device)!=cudaSuccess)return 2;
        if(device!=m.device)return 1;
        for(int i=0;i<n;++i)if(warm[i]>1)return 1;
        if(n&&cudaMemcpyAsync(dw,warm,n,cudaMemcpyHostToDevice,m.stream)!=cudaSuccess)return 2;
        const auto p=problem(max_ships);
        seed<<<1,32,0,m.stream>>>(p,order,dw,masks,rows);
        if(rounds) {
            start_exchange<<<1,32,0,m.stream>>>(p,masks,rows,exchange);
            // Device state stops evaluation after a locally optimal sweep.
            for(int round=0;round<rounds;++round) {
                evaluate_exchanges<<<((n+1)*101+127)/128,128,0,m.stream>>>(p,masks,rows,exchange,proposal_values);
                accept_exchange<<<1,256,0,m.stream>>>(p,masks,rows,exchange,proposal_values);
            }
        }
        search<<<tasks,1,0,m.stream>>>(p,bits,tasks,cap,masks,active,phase,exclusions,rows);
        finish<<<1,32,0,m.stream>>>(p,tasks,masks,rows,out,result);
        // Submit all computation before collecting any host diagnostics.
        if(rounds) {
            if(cudaMemcpyAsync(exchange_report,&exchange->report,sizeof(ExchangeReport),cudaMemcpyDeviceToHost,m.stream)!=cudaSuccess)return 2;
        } else *exchange_report={0,0,0};
        if(cudaGetLastError()!=cudaSuccess||cudaMemcpyAsync(report,result,sizeof(Report),cudaMemcpyDeviceToHost,m.stream)!=cudaSuccess||
           (n&&cudaMemcpyAsync(selected,out,n,cudaMemcpyDeviceToHost,m.stream)!=cudaSuccess)||cudaStreamSynchronize(m.stream)!=cudaSuccess)return 2;
        return 0;
    }
};
}
extern "C" int spacepdhcg_gtoc12_fleet_workspace_create_host(int32_t n,int32_t bits,
    const spacepdhcg_gtoc12_fleet_column* columns,const int32_t* co,const int32_t* ci,
    const int32_t* ro,const int32_t* po,const int32_t* pi,void** workspace) {
    using namespace gtoc12_fleet;
    if(!workspace)return 1;*workspace=nullptr;
    if(n<0||n>4096||bits<0||bits>10||(n&&!columns)||
       !offsets(co,n)||!offsets(ro,n)||!offsets(po,ro[n]))return 1;
    if((co[n]&&!ci)||(po[ro[n]]&&!pi))return 1;
    double magnitude=0,total_mass=0;
    for(int i=0;i<n;++i) {
        if(columns[i].ships<=0||columns[i].ships>100||
           !std::isfinite(columns[i].value)||!std::isfinite(columns[i].mass)||columns[i].mass<0)return 1;
        magnitude+=std::abs(columns[i].value);total_mass+=columns[i].mass;
        if(!std::isfinite(magnitude)||!std::isfinite(total_mass))return 1;
    }
    for(int i=0;i<co[n];++i)if(ci[i]<0||ci[i]>=n)return 1;
    for(int i=0;i<po[ro[n]];++i)if(pi[i]<0||pi[i]>=n)return 1;
    try {
        auto w=std::make_unique<Workspace>();
        if(!w->init(n,bits,columns,co,ci,ro,po,pi))return 2;
        *workspace=w.release();return 0;
    } catch(...){return 2;}
}
extern "C" int spacepdhcg_gtoc12_fleet_workspace_solve_host(void* workspace,int32_t max_ships,
    uint64_t cap,const uint8_t* warm,uint8_t* selected,spacepdhcg_gtoc12_fleet_report* report,
    int32_t rounds,spacepdhcg_gtoc12_fleet_exchange_report* exchange) {
    if(!workspace)return 1;
    return static_cast<gtoc12_fleet::Workspace*>(workspace)->solve(max_ships,cap,warm,selected,report,rounds,exchange);
}
extern "C" void spacepdhcg_gtoc12_fleet_workspace_destroy_host(void* workspace) {
    delete static_cast<gtoc12_fleet::Workspace*>(workspace);
}
static int run_gtoc12_fleet(int32_t n,int32_t max_ships,int32_t bits,uint64_t cap,
    const spacepdhcg_gtoc12_fleet_column* c,const int32_t* co,const int32_t* ci,
    const int32_t* ro,const int32_t* po,const int32_t* pi,const uint8_t* warm,
    uint8_t* selected,spacepdhcg_gtoc12_fleet_report* report,
    int rounds,spacepdhcg_gtoc12_fleet_exchange_report* exchange) {
    // Keep the original one-shot entry points and output semantics.
    if(max_ships<0||max_ships>100||rounds<0||rounds>100||!report||
       (n>0&&(!warm||!selected)))return 1;
    void* handle=nullptr;
    int status=spacepdhcg_gtoc12_fleet_workspace_create_host(n,bits,c,co,ci,ro,po,pi,&handle);
    if(status)return status;
    std::unique_ptr<gtoc12_fleet::Workspace> w(static_cast<gtoc12_fleet::Workspace*>(handle));
    gtoc12_fleet::ExchangeReport unused{};
    return w->solve(max_ships,cap,warm,selected,report,rounds,exchange?exchange:&unused);
}
extern "C" int spacepdhcg_gtoc12_fleet_search_host(int32_t n,int32_t max_ships,int32_t bits,uint64_t cap,
    const spacepdhcg_gtoc12_fleet_column* c,const int32_t* co,const int32_t* ci,
    const int32_t* ro,const int32_t* po,const int32_t* pi,const uint8_t* warm,
    uint8_t* selected,spacepdhcg_gtoc12_fleet_report* report) {
    return run_gtoc12_fleet(n,max_ships,bits,cap,c,co,ci,ro,po,pi,warm,selected,report,0,nullptr);
}
extern "C" int spacepdhcg_gtoc12_fleet_search_v2_host(int32_t n,int32_t max_ships,int32_t bits,uint64_t cap,
    const spacepdhcg_gtoc12_fleet_column* c,const int32_t* co,const int32_t* ci,
    const int32_t* ro,const int32_t* po,const int32_t* pi,const uint8_t* warm,
    uint8_t* selected,spacepdhcg_gtoc12_fleet_report* report,int32_t rounds,
    spacepdhcg_gtoc12_fleet_exchange_report* exchange) {
    if(!exchange)return 1;
    return run_gtoc12_fleet(n,max_ships,bits,cap,c,co,ci,ro,po,pi,warm,selected,report,rounds,exchange);
}
