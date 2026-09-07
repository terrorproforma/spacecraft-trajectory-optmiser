#include "spacepdhcg/cuda/gtoc12_collect_dp_c_api.h"
#include <cuda_runtime.h>
#include <cmath>
#include <climits>
#include <initializer_list>
#include <mutex>
#include <new>
#include <vector>
#include <algorithm>
#include <cstring>

namespace {
using Policy=spacepdhcg_collect_policy;
using Inputs=spacepdhcg_collect_inputs;
using Result=spacepdhcg_collect_result_v2;
static_assert(sizeof(Policy)==144);
static_assert(sizeof(Inputs)==88);
static_assert(sizeof(Result)==632);
static_assert(sizeof(spacepdhcg_collect_result)==496);
struct View {
    Policy p; Inputs in;
    double *arrive{},*ready{},*mass{},*terminal{};
    int32_t *prefix{},*previous{},*departure{},*tof{},*allocated{};
    Result* result{};
};
struct Workspace {
    View v{};int device{};cudaStream_t stream{};std::mutex mutex;
    ~Workspace() {
        cudaStreamSynchronize(stream);
        for(auto ptr:std::initializer_list<const double*>{v.in.dv,v.in.returns,v.in.tofs,v.in.return_tofs,v.in.mined,
            v.in.geometry_a,v.in.geometry_l,v.in.penalty,v.in.override_inflation,
            v.arrive,v.ready,v.mass,v.terminal})cudaFree(const_cast<double*>(ptr));
        for(auto ptr:std::initializer_list<const int32_t*>{v.in.steps,v.in.banned,v.prefix,v.previous,v.departure,v.tof,v.allocated})
            cudaFree(const_cast<int32_t*>(ptr));
        cudaFree(v.result);if(stream)cudaStreamDestroy(stream);
    }
};
__device__ size_t state(const View& v,int subset,int j){return size_t(subset)*v.p.k+j;}
__device__ double authority(const View& v,double mass,double tof) {
    return ((v.p.thrust/mass)*1e-3)*tof*86400.0;
}
__device__ double fraction(const View& v,int j,int l,int t,int h,double mass) {
    const auto& p=v.p;const size_t pair=size_t(j)*p.k+l;
    const double dv=v.in.dv[(pair*p.n+t)*p.nt+h],tof=v.in.tofs[h];
    const double a=authority(v,mass,tof);
    if(!isfinite(dv)||dv>p.hop_ratio*a)return INFINITY;
    const double ratio=dv/fmax(a,1e-12);
    double inflation=p.hop_flat;
    if(p.hop_model==1)inflation=p.hop_floor+p.hop_slope*ratio;
    if(p.hop_model==2) {
        const double x[5]={1,ratio,tof/365.25,fabs(v.in.geometry_a[pair])/.1,
            fabs(v.in.geometry_l[pair*p.n+t])/3.14159265358979323846};
        inflation=0;
        for(int i=0;i<5;++i)inflation+=x[i]*p.fit_coefficients[i];
        inflation=fmax(inflation,p.fit_floor);
    }
    // Retain the reference multiply then divide; do not simplify away rounding.
    return (mass*(1-exp(-(dv*inflation)/p.exhaust)))/mass;
}
__device__ double return_base(double tof) {
    constexpr double days[]={352,420,450,480,510,540,578,630,690,810};
    constexpr double values[]={1.323,1.383,1.295,1.195,1.099,.977,.885,.930,.932,1.014};
    if(tof<=days[0])return values[0];
    for(int i=1;i<10;++i)if(tof<=days[i])
        return values[i-1]+(values[i]-values[i-1])/(days[i]-days[i-1])*(tof-days[i-1]);
    return values[9];
}
__device__ double return_cost(const View& v,int j,int t,int h,double mass) {
    const size_t i=(size_t(j)*v.p.n+t)*v.p.nr;
    const double dv=v.in.returns[i+h],tof=v.in.return_tofs[h];
    double inflation=v.in.override_inflation[i+h];
    if(!isfinite(dv))return INFINITY;
    if(isnan(inflation)) {
        const double a=authority(v,mass,tof);
        if(dv>v.p.return_ratio*a)return INFINITY;
        inflation=v.p.return_model?fmax(.85,return_base(tof)*fmin(1.2,fmax(.85,1+.6*(dv/fmax(a,1e-12)-.33)))):
            v.p.return_flat;
    } else if(!isfinite(inflation))return INFINITY;
    return mass*(1-exp(-(dv*inflation)/v.p.exhaust));
}
__global__ void initialise(View v,size_t cells) {
    size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i<cells){v.arrive[i]=-INFINITY;v.ready[i]=-INFINITY;v.prefix[i]=0;
        v.previous[i]=-1;v.departure[i]=-1;v.tof[i]=-1;}
    if(i<size_t(1<<v.p.k)*v.p.k)v.allocated[i]=0;
    if(i==0){*v.result={};v.result->objective=-INFINITY;
        for(int j=0;j<16;++j)v.result->collected_at[j]=-1;}
}
__global__ void start(View v) {
    if(threadIdx.x||blockIdx.x)return;
    const size_t s=state(v,0,v.p.camp);v.allocated[s]=1;v.arrive[s*v.p.n]=0;
}
__global__ void prefixes(View v,int cardinality,bool initial_only) {
    const int s=blockIdx.x*blockDim.x+threadIdx.x;
    const int subset=s/v.p.k,j=s%v.p.k;
    if(subset>=(1<<v.p.k)||__popc(unsigned(subset))!=cardinality||(subset>>j&1))return;
    if(initial_only&&j!=v.p.camp)return;
    double best=-INFINITY;int index=0;
    for(int t=0;t<v.p.n;++t){const double value=v.arrive[size_t(s)*v.p.n+t];
        if(value>=best){best=value;index=t;}
        v.ready[size_t(s)*v.p.n+t]=best;v.prefix[size_t(s)*v.p.n+t]=index;}
}
__global__ void transitions(View v,int cardinality,double camp_mass,double price) {
    const size_t index=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    const size_t cells=size_t(1<<v.p.k)*v.p.k*v.p.n;if(index>=cells)return;
    const int t=index%v.p.n,s=index/v.p.n,subset=s/v.p.k,l=s%v.p.k;
    if(__popc(unsigned(subset))!=cardinality||(subset>>l&1))return;
    const bool skip=cardinality==0;
    if(skip&&l==v.p.camp)return;
    double best=-INFINITY;int prev=-1,dep=-1,tof_index=-1;bool allocated=false;
    // Ascending predecessor subset means descending removed-bit index.
    // Keep the source-order epsilon comparison instead of an unordered max.
    for(int j=v.p.k-1;j>=0;--j) {
        if(skip?j!=v.p.camp:!(subset>>j&1))continue;
        if(j==l||v.in.banned[j*v.p.k+l])continue;
        const int old=skip?0:subset&~(1<<j);const size_t source=state(v,old,j)*v.p.n;
        if(!isfinite(v.ready[source+v.p.n-1]))continue;
        allocated=true;
        const double mass=skip?camp_mass:v.mass[subset];
        double move_best=-INFINITY;int move_t=-1,move_h=-1;
        for(int h=0;h<v.p.nt;++h) {
            const int d=t-v.in.steps[h];if(d<0)continue;
            double base=v.ready[source+d];
            if(!skip)base+=v.in.mined[j*v.p.n+d];
            double value=base-(price*mass)*fraction(v,j,l,d,h,mass);
            value-=price*v.in.penalty[(size_t(j)*v.p.k+l)*v.p.n+d];
            if(value>move_best){move_best=value;move_t=d;move_h=h;}
        }
        if(move_best>best+1e-9){best=move_best;prev=skip?j+v.p.k:j;dep=move_t;tof_index=move_h;}
    }
    v.arrive[index]=best;v.previous[index]=prev;v.departure[index]=dep;v.tof[index]=tof_index;
    if(t==0)v.allocated[s]=allocated;
}
__global__ void terminals(View v,double price) {
    const int index=blockIdx.x*blockDim.x+threadIdx.x;
    if(index>=v.p.k*v.p.n)return;
    const int j=index/v.p.n,t=index%v.p.n,full=(1<<v.p.k)-1;
    const double base=v.ready[state(v,full&~(1<<j),j)*v.p.n+t]+v.in.mined[index];
    double best=-INFINITY;int winner=-1;
    for(int h=0;h<v.p.nr;++h){double value=base-price*return_cost(v,j,t,h,v.mass[full]);
        if(value>best){best=value;winner=h;}}
    v.terminal[2*index]=best;v.terminal[2*index+1]=winner;
}
__global__ void finish(View v,double camp_mass) {
    if(threadIdx.x||blockIdx.x)return;
    auto& r=*v.result;int j_best=-1,t_best=-1,h_best=-1;
    for(int s=0;s<(1<<v.p.k)*v.p.k;++s)r.states+=v.allocated[s];
    // NumPy argmax first selects the earliest departure/TOF within each j;
    // the +1e-9 comparison applies only between terminal locations.
    for(int j=v.p.k-1;j>=0;--j){double best=-INFINITY;int t_win=-1,h_win=-1;
        for(int t=0;t<v.p.n;++t){const int index=j*v.p.n+t;
            if(v.terminal[2*index]>best){best=v.terminal[2*index];t_win=t;h_win=int(v.terminal[2*index+1]);}}
        if(best>r.objective+1e-9){r.objective=best;j_best=j;t_best=t_win;h_best=h_win;}}
    if(j_best<0)return;
    r.feasible=1;r.terminal_j=j_best;r.terminal_t=t_best;r.terminal_r=h_best;
    r.return_dv=v.in.returns[(size_t(j_best)*v.p.n+t_best)*v.p.nr+h_best];
    r.collected_at[j_best]=t_best;
    int subset=((1<<v.p.k)-1)&~(1<<j_best),j=j_best;
    int arrival=v.prefix[state(v,subset,j)*v.p.n+t_best];
    while(subset!=0||j!=v.p.camp){
        const size_t index=state(v,subset,j)*v.p.n+arrival;
        const int encoded=v.previous[index],dep=v.departure[index],h=v.tof[index];
        if(encoded<0||dep<0||h<0||r.hops>=v.p.k){r.feasible=0;r.reserved=1;return;}
        const bool skipped=encoded>=v.p.k;const int prev=skipped?encoded-v.p.k:encoded;
        const int pos=r.hops++;r.source[pos]=prev;r.target[pos]=j;r.departure[pos]=dep;r.tof[pos]=h;
        const double mass=skipped?camp_mass:v.mass[subset];
        r.hop_propellant[pos]=mass*fraction(v,prev,j,dep,h,mass);
        r.hop_dv[pos]=v.in.dv[((size_t(prev)*v.p.k+j)*v.p.n+dep)*v.p.nt+h];
        r.penalty+=v.in.penalty[(size_t(prev)*v.p.k+j)*v.p.n+dep];
        if(skipped)r.reposition=1;
        else{r.collected_at[prev]=dep;subset&=~(1<<prev);}
        j=prev;arrival=v.prefix[state(v,subset,j)*v.p.n+dep];
    }
}
template<class T> bool allocate(T*& p,size_t n){return cudaMalloc(reinterpret_cast<void**>(&p),n*sizeof(T))==cudaSuccess;}
template<class T> bool copy(const T*& p,const T* src,size_t n,cudaStream_t stream){
    T* target=nullptr;if(!allocate(target,n))return false;p=target;
    return cudaMemcpyAsync(target,src,n*sizeof(T),cudaMemcpyHostToDevice,stream)==cudaSuccess;
}
}
static int create_workspace(const Policy* p,const Inputs* in,void** out,bool resident) {
    if(!p||!in||!out||*out||p->k<1||p->k>16||p->n<1||p->nt<1||p->nr<1||p->camp<0||p->camp>=p->k
        ||p->hop_model<0||p->hop_model>2||p->return_model<0||p->return_model>1)return 1;
    const size_t states=size_t(1<<p->k)*p->k,cells=states*p->n,pairs=size_t(p->k)*p->k;
    if(cells>INT_MAX||size_t(p->k)*p->n*p->nr>INT_MAX||pairs*p->n*p->nt>INT_MAX)return 1;
    const double* numbers=&p->thrust;
    for(int i=0;i<14;++i)if(!std::isfinite(numbers[i]))return 1;
    if(p->thrust<=0||p->exhaust<=0||p->hop_ratio<0||p->return_ratio<0)return 1;
    if((!resident&&(!in->dv||!in->returns))||!in->tofs||!in->return_tofs||!in->mined||!in->geometry_a||!in->geometry_l
        ||!in->penalty||!in->override_inflation||!in->steps||!in->banned)return 1;
    for(int h=0;h<p->nt;++h)if(in->steps[h]<1||!std::isfinite(in->tofs[h])||in->tofs[h]<=0)return 1;
    for(int h=0;h<p->nr;++h)if(!std::isfinite(in->return_tofs[h])||in->return_tofs[h]<=0)return 1;
    auto* w=new(std::nothrow) Workspace;if(!w)return 2;
    w->v.p=*p;auto& v=w->v;
    bool ok=cudaGetDevice(&w->device)==cudaSuccess&&cudaStreamCreateWithFlags(&w->stream,cudaStreamNonBlocking)==cudaSuccess;
    if(resident) {
        double* dv=nullptr;double* returns=nullptr;
        ok=ok&&allocate(dv,pairs*p->n*p->nt)&&allocate(returns,size_t(p->k)*p->n*p->nr);
        v.in.dv=dv;v.in.returns=returns;
    } else ok=ok&&copy(v.in.dv,in->dv,pairs*p->n*p->nt,w->stream)
        &&copy(v.in.returns,in->returns,size_t(p->k)*p->n*p->nr,w->stream);
    ok=ok
        &&copy(v.in.tofs,in->tofs,p->nt,w->stream)&&copy(v.in.return_tofs,in->return_tofs,p->nr,w->stream)
        &&copy(v.in.mined,in->mined,size_t(p->k)*p->n,w->stream)&&copy(v.in.geometry_a,in->geometry_a,pairs,w->stream)
        &&copy(v.in.geometry_l,in->geometry_l,pairs*p->n,w->stream)&&copy(v.in.penalty,in->penalty,pairs*p->n,w->stream)
        &&copy(v.in.override_inflation,in->override_inflation,size_t(p->k)*p->n*p->nr,w->stream)
        &&copy(v.in.steps,in->steps,p->nt,w->stream)&&copy(v.in.banned,in->banned,pairs,w->stream)
        &&allocate(v.arrive,cells)&&allocate(v.ready,cells)&&allocate(v.prefix,cells)&&allocate(v.previous,cells)
        &&allocate(v.departure,cells)&&allocate(v.tof,cells)&&allocate(v.allocated,states)
        &&allocate(v.mass,1<<p->k)&&allocate(v.terminal,size_t(p->k)*p->n*2)&&allocate(v.result,1);
    ok=(cudaStreamSynchronize(w->stream)==cudaSuccess)&&ok;
    if(!ok){delete w;return 2;}*out=w;return 0;
}
extern "C" int spacepdhcg_collect_create(const Policy* p,const Inputs* in,void** out) {
    return create_workspace(p,in,out,false);
}
extern "C" int spacepdhcg_collect_solve_v2(void* opaque,const double* masses,double camp_mass,double price,Result* result){
    auto* w=static_cast<Workspace*>(opaque);
    if(!w||!masses||!result||!std::isfinite(camp_mass)||camp_mass<=0||!std::isfinite(price)||price<=0)return 1;
    std::unique_lock<std::mutex> lock(w->mutex,std::try_to_lock);if(!lock.owns_lock())return 3;
    int device=-1;if(cudaGetDevice(&device)!=cudaSuccess)return 2;if(device!=w->device)return 1;
    auto v=w->v;const int states=(1<<v.p.k)*v.p.k;const size_t cells=size_t(states)*v.p.n;
    for(int i=0;i<(1<<v.p.k);++i)if(!std::isfinite(masses[i])||masses[i]<=0)return 1;
    auto status=cudaMemcpyAsync(v.mass,masses,size_t(1<<v.p.k)*sizeof(double),cudaMemcpyHostToDevice,w->stream);
    if(status!=cudaSuccess)return 2;
    initialise<<<(cells+255)/256,256,0,w->stream>>>(v,cells);start<<<1,1,0,w->stream>>>(v);
    prefixes<<<(states+127)/128,128,0,w->stream>>>(v,0,true);
    for(int layer=0;layer<v.p.k;++layer){
        transitions<<<(cells+127)/128,128,0,w->stream>>>(v,layer,camp_mass,price);
        prefixes<<<(states+127)/128,128,0,w->stream>>>(v,layer,false);
    }
    terminals<<<(v.p.k*v.p.n+127)/128,128,0,w->stream>>>(v,price);
    finish<<<1,1,0,w->stream>>>(v,camp_mass);
    status=cudaGetLastError();
    if(status==cudaSuccess)status=cudaMemcpyAsync(result,v.result,sizeof(Result),cudaMemcpyDeviceToHost,w->stream);
    const auto finished=cudaStreamSynchronize(w->stream);
    return status==cudaSuccess&&finished==cudaSuccess?0:2;
}
extern "C" int spacepdhcg_collect_solve(void* opaque,const double* masses,double camp_mass,double price,spacepdhcg_collect_result* result){
    if(!result)return 1;Result extended{};
    const int status=spacepdhcg_collect_solve_v2(opaque,masses,camp_mass,price,&extended);
    if(!status)std::memcpy(result,&extended,sizeof(*result));return status;
}
extern "C" int spacepdhcg_collect_destroy(void* opaque){
    auto* w=static_cast<Workspace*>(opaque);if(!w)return 0;
    int device=-1;if(cudaGetDevice(&device)!=cudaSuccess)return 2;if(device!=w->device)return 1;
    if(!w->mutex.try_lock())return 3;w->mutex.unlock();delete w;return 0;
}

namespace {
struct DeviceTable {
    int device{},n{},nt{};float* values{};cudaStream_t stream{};std::mutex mutex;
    ~DeviceTable(){if(stream)cudaStreamSynchronize(stream);cudaFree(values);if(stream)cudaStreamDestroy(stream);}
};
__global__ void round_table(const double* dv,const uint8_t* ok,const double* epochs,
    const double* tofs,int nt,size_t count,double end,float* output) {
    const size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i<count)output[i]=ok[i]&&epochs[i/nt]+tofs[i%nt]<=end+1e-9
        ?static_cast<float>(dv[i]):INFINITY;
}
__global__ void assemble_tables(const float* const* inputs,int k,int n,int nt,int nr,
    int t0,size_t count,double* pairs,double* returns) {
    const size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i>=count)return;
    const size_t pair_cells=size_t(k)*k*n*nt;
    const bool ret=i>=pair_cells;const size_t index=ret?i-pair_cells:i;
    const int tofs=ret?nr:nt;const size_t cells=size_t(n)*tofs;
    const auto* input=inputs[(ret?k*k:0)+index/cells];
    (ret?returns:pairs)[index]=input?static_cast<double>(input[size_t(t0)*tofs+index%cells]):INFINITY;
}
}
extern "C" int spacepdhcg_collect_table_create(spacepdhcg_orbitweaver_lambert_workspace* lambert,
    const spacepdhcg_orbitweaver_hop_elements* elements,const double* epochs,int32_t n,
    const double* tofs,int32_t nt,double end,void** output) {
    if(!lambert||!elements||!epochs||!tofs||!output||*output||n<=0||nt<=0
        ||int64_t(n)*nt>INT_MAX||!std::isfinite(end))return 1;
    auto* table=new(std::nothrow) DeviceTable;if(!table)return 2;
    table->n=n;table->nt=nt;const size_t count=size_t(n)*nt;
    const double *de=nullptr,*dt=nullptr;double* dv=nullptr;uint8_t* ok=nullptr;
    bool good=cudaGetDevice(&table->device)==cudaSuccess
        &&cudaStreamCreateWithFlags(&table->stream,cudaStreamNonBlocking)==cudaSuccess
        &&copy(de,epochs,n,table->stream)&&copy(dt,tofs,nt,table->stream)
        &&allocate(dv,count)&&allocate(ok,count)&&allocate(table->values,count);
    good=(cudaStreamSynchronize(table->stream)==cudaSuccess)&&good;
    if(good)good=spacepdhcg_orbitweaver_hop_grid_device(lambert,elements,de,n,dt,nt,dv,ok)==SPACEPDHCG_CUDA_SUCCESS;
    if(good){
        round_table<<<(count+127)/128,128,0,table->stream>>>(dv,ok,de,dt,nt,count,end,table->values);
        good=cudaGetLastError()==cudaSuccess;
    }
    good=(cudaStreamSynchronize(table->stream)==cudaSuccess)&&good;
    cudaFree(const_cast<double*>(de));cudaFree(const_cast<double*>(dt));cudaFree(dv);cudaFree(ok);
    if(!good){delete table;return 2;}*output=table;return 0;
}
extern "C" int spacepdhcg_collect_table_read(void* opaque,float* output) {
    auto* table=static_cast<DeviceTable*>(opaque);if(!table||!output)return 1;
    std::unique_lock<std::mutex> lock(table->mutex,std::try_to_lock);if(!lock.owns_lock())return 3;
    int device=-1;if(cudaGetDevice(&device)!=cudaSuccess)return 2;if(device!=table->device)return 1;
    const auto status=cudaMemcpyAsync(output,table->values,size_t(table->n)*table->nt*sizeof(float),cudaMemcpyDeviceToHost,table->stream);
    const auto done=cudaStreamSynchronize(table->stream);return status==cudaSuccess&&done==cudaSuccess?0:2;
}
extern "C" int spacepdhcg_collect_table_destroy(void* opaque) {
    auto* table=static_cast<DeviceTable*>(opaque);if(!table)return 0;
    int device=-1;if(cudaGetDevice(&device)!=cudaSuccess)return 2;if(device!=table->device)return 1;
    if(!table->mutex.try_lock())return 3;table->mutex.unlock();delete table;return 0;
}
extern "C" int spacepdhcg_collect_create_tables(const Policy* p,const Inputs* in,
    void* const* pairs,void* const* returns,int32_t t0,void** output) {
    if(!p||!in||!pairs||!returns||!output||*output||p->k<1||p->k>16||t0<0||p->n<1||p->nt<1||p->nr<1)return 1;
    int device=-1;if(cudaGetDevice(&device)!=cudaSuccess)return 2;
    std::vector<DeviceTable*> tables;
    for(int i=0;i<p->k*p->k+p->k;++i){
        const bool ret=i>=p->k*p->k;auto* table=static_cast<DeviceTable*>(ret?returns[i-p->k*p->k]:pairs[i]);
        if(!table){if(ret||!in->banned||!in->banned[i])return 1;continue;}
        if(table->device!=device||int64_t(t0)+p->n>table->n||table->nt!=(ret?p->nr:p->nt))return 1;
        if(std::find(tables.begin(),tables.end(),table)==tables.end())tables.push_back(table);
    }
    std::vector<std::unique_lock<std::mutex>> locks;
    for(auto* table:tables){locks.emplace_back(table->mutex,std::try_to_lock);if(!locks.back().owns_lock())return 3;}
    const int status=create_workspace(p,in,output,true);if(status)return status;
    auto* w=static_cast<Workspace*>(*output);
    std::vector<const float*> pointers;
    for(int i=0;i<p->k*p->k+p->k;++i){
        const bool ret=i>=p->k*p->k;auto* table=static_cast<DeviceTable*>(ret?returns[i-p->k*p->k]:pairs[i]);
        pointers.push_back(table?table->values:nullptr);
    }
    const float** device_pointers=nullptr;
    bool good=allocate(device_pointers,pointers.size());
    if(good)good=cudaMemcpyAsync(device_pointers,pointers.data(),pointers.size()*sizeof(float*),cudaMemcpyHostToDevice,w->stream)==cudaSuccess;
    if(good){
        const size_t count=size_t(p->k)*p->k*p->n*p->nt+size_t(p->k)*p->n*p->nr;
        assemble_tables<<<(count+255)/256,256,0,w->stream>>>(device_pointers,p->k,p->n,p->nt,p->nr,t0,count,
            const_cast<double*>(w->v.in.dv),const_cast<double*>(w->v.in.returns));
        good=cudaGetLastError()==cudaSuccess;
    }
    good=(cudaStreamSynchronize(w->stream)==cudaSuccess)&&good;
    cudaFree(device_pointers);
    if(!good){delete w;*output=nullptr;return 2;}return 0;
}
