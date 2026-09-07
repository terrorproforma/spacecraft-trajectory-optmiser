#include "spacepdhcg/cuda/gtoc12_neighbours_c_api.h"
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <algorithm>
#include <cmath>
#include <limits>
#include <mutex>
#include <new>

namespace {
constexpr double pi=3.141592653589793238462643383279502884;
constexpr double degrees=180.0/pi;
using Body=spacepdhcg_gtoc12_neighbour_body;
using Query=spacepdhcg_gtoc12_neighbour_query;
struct Derived { Body body; double ex,ey,ix,iy,p[3],q[3],motion; };
__global__ void derive(const Body* bodies,Derived* data,int n,double mu) {
    const int j=blockIdx.x*blockDim.x+threadIdx.x;
    if(j>=n)return;
    const auto b=bodies[j]; auto& d=data[j];d.body=b;
    d.ex=b.e*cos(b.node+b.perihelion);d.ey=b.e*sin(b.node+b.perihelion);
    d.ix=b.inclination*cos(b.node);d.iy=b.inclination*sin(b.node);
    const double cn=cos(b.node),sn=sin(b.node),cp=cos(b.perihelion),sp=sin(b.perihelion);
    const double ci=cos(b.inclination),si=sin(b.inclination);
    d.p[0]=cp*cn-sp*sn*ci;d.p[1]=cp*sn+sp*cn*ci;d.p[2]=sp*si;
    d.q[0]=-sp*cn-cp*sn*ci;d.q[1]=-sp*sn+cp*cn*ci;d.q[2]=cp*si;
    d.motion=sqrt(mu/(b.a*b.a*b.a));
}
__device__ bool position(const Derived& b,double epoch,double r[3]) {
    double mean=fmod(b.body.mean+b.motion*((epoch-b.body.epoch)*86400.0),2*pi);
    if(mean<0)mean+=2*pi;
    double eccentric=b.body.e>0.8 ? pi : mean;
    bool converged=false;
    for(int k=0;k<64;++k) {
        const double step=(eccentric-b.body.e*sin(eccentric)-mean)/(1-b.body.e*cos(eccentric));
        eccentric-=step;
        if(fabs(step)<1e-14){converged=true;break;}
    }
    const double f=2*atan2(sqrt(1+b.body.e)*sin(0.5*eccentric),sqrt(1-b.body.e)*cos(0.5*eccentric));
    const double radius=b.body.a*(1-b.body.e*b.body.e)/(1+b.body.e*cos(f));
    for(int k=0;k<3;++k)r[k]=radius*(b.p[k]*cos(f)+b.q[k]*sin(f));
    return converged&&isfinite(r[0])&&isfinite(r[1])&&isfinite(r[2]);
}
__device__ bool valid(const Query& q,int bodies) {
    return q.source_index>=0&&q.source_index<bodies&&q.neighbours>0&&isfinite(q.epoch)
        &&q.band_a>0&&isfinite(q.band_a)&&q.band_e>0&&isfinite(q.band_e)
        &&q.band_i>0&&isfinite(q.band_i)&&q.band_phase>0&&isfinite(q.band_phase)
        &&q.filter_scale>=0&&isfinite(q.filter_scale);
}
__device__ void deviations(const Derived& a,const Derived& b,double au,double& da,double& de,double& di) {
    da=fabs(a.body.a-b.body.a)/au;
    de=sqrt((a.ex-b.ex)*(a.ex-b.ex)+(a.ey-b.ey)*(a.ey-b.ey));
    di=degrees*sqrt((a.ix-b.ix)*(a.ix-b.ix)+(a.iy-b.iy)*(a.iy-b.iy));
}
__global__ void bands(const Derived* data,int bodies,const int* pool,int n,const Query* query,
                      double au,double* keys,int* order,int* inside,int* inside_count,int* invalid) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
    keys[i]=INFINITY;order[i]=i;inside[i]=0;
    const auto q=*query;
    if(!valid(q,bodies)){atomicExch(invalid,1);return;}
    if(pool[i]==q.source_index)return;
    double da,de,di;deviations(data[q.source_index],data[pool[i]],au,da,de,di);
    const double a=da/q.band_a,e=de/q.band_e,inc=di/q.band_i;
    keys[i]=a*a+e*e+inc*inc;
    inside[i]=da<=q.filter_scale*q.band_a&&de<=q.filter_scale*q.band_e&&di<=q.filter_scale*q.band_i;
    if(inside[i])atomicAdd(inside_count,1);
}
__global__ void eligibility(int n,const Query* query,const double* sorted,const int* order,
                           const int* inside,const int* inside_count,int* eligible) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
    if(*inside_count>=query->neighbours)eligible[i]=inside[i];
    else eligible[order[i]]=i<query->neighbours&&isfinite(sorted[i]);
}
__global__ void metrics(const Derived* data,const int* pool,int n,const Query* query,
                       const int* eligible,const double* tofs,int nt,double mu,double au,
                       double* proxy,double* positional,int* order,int* invalid) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
    proxy[i]=positional[i]=INFINITY;order[i]=i;
    if(!eligible[i]||*invalid)return;
    const auto q=*query;const auto& s=data[q.source_index];const auto& t=data[pool[i]];
    double rs[3],rt[3];if(!position(s,q.epoch,rs)||!position(t,q.epoch,rt)){atomicExch(invalid,1);return;}
    const double phase=atan2(rs[0]*rt[1]-rs[1]*rt[0],rt[0]*rs[0]+rt[1]*rs[1]+rt[2]*rs[2]);
    double da,de,di;deviations(s,t,au,da,de,di);
    const double a=da/q.band_a,inc=di/q.band_i,e=de/q.band_e,p=degrees*phase/q.band_phase;
    positional[i]=a*a+inc*inc+e*e+p*p;
    const double speed=sqrt(mu/s.body.a),ns=speed/s.body.a;
    const double dva=0.5*speed*fabs(t.body.a-s.body.a)/s.body.a;
    const double dvi=speed*sqrt((t.ix-s.ix)*(t.ix-s.ix)+(t.iy-s.iy)*(t.iy-s.iy));
    const double dve=0.5*speed*de;
    const double orbit=sqrt(dva*dva+dvi*dvi+dve*dve);
    double best=INFINITY;
    for(int k=0;k<nt;++k) {
        const double seconds=tofs[k]*86400;
        double residual=phase+(t.motion-ns)*seconds;
        residual=atan2(sin(residual),cos(residual));
        const double dv=orbit+(2.0/3.0)*speed*fabs(residual)/(ns*seconds);
        if(dv<best)best=dv;
    }
    proxy[i]=best;
}
__global__ void collect_proxy(const Derived* data,const int* pool,int n,const Query* q,
                             const double* keys,const int* order,int* selected,int64_t* output,int* count) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
    selected[i]=0;
    if(i<q->neighbours&&isfinite(keys[i])) {
        output[i]=data[pool[order[i]]].body.id;
        atomicAdd(count,1);
    }
}
__global__ void unique_position(const Derived* data,const int* pool,int n,const Query* q,
                               const double* keys,const int* order,const int64_t* output,
                               const int* proxy_count,int* keep) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
    keep[i]=0;
    if(i>=q->neighbours/2||!isfinite(keys[i]))return;
    const auto id=data[pool[order[i]]].body.id;
    for(int k=0;k<*proxy_count;++k)if(output[k]==id)return;
    keep[i]=1;
}
__global__ void compact(const Derived* data,const int* pool,int n,const int* order,
                       const int* keep,const int* prefix,int64_t* output,int* count,const int* invalid) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
    // count is finalized by a separate kernel after all readers have finished.
    if(keep[i])output[*count+prefix[i]]=data[pool[order[i]]].body.id;
    (void)invalid;
}
__global__ void finish(int n,const int* keep,const int* prefix,int* count,const int* invalid) {
    if(*invalid)*count=-1;else *count+=prefix[n-1]+keep[n-1];
}
spacepdhcg_cuda_status mapped(cudaError_t s) {
    return s==cudaSuccess?SPACEPDHCG_CUDA_SUCCESS:s==cudaErrorMemoryAllocation?SPACEPDHCG_CUDA_OUT_OF_MEMORY:SPACEPDHCG_CUDA_RUNTIME_ERROR;
}
}

struct spacepdhcg_gtoc12_neighbours {
    int bodies{},n{},nt{},device{};double mu{},au{};
    Body* raw{};Derived* data{};int* pool{};double* tofs{};
    double *keys{},*sorted{},*positional{};
    int *order{},*sorted_order{},*inside{},*eligible{},*keep{},*prefix{},*inside_count{},*invalid{},*count{};
    int64_t* output{};Query* query{};void* temp{};size_t temp_bytes{};
    cudaStream_t stream{};std::mutex mutex;
};

extern "C" spacepdhcg_cuda_status spacepdhcg_gtoc12_neighbours_launch_device(
    spacepdhcg_gtoc12_neighbours* w,const Query* query,int64_t* output,int* count,
    spacepdhcg_accelerator_stream stream) {
    if(!w||!query||!output||!count||stream.device.type!=SPACEPDHCG_DEVICE_CUDA||stream.device.id!=w->device)
        return SPACEPDHCG_CUDA_INVALID_ARGUMENT;
    int device=-1;auto status=cudaGetDevice(&device);if(status!=cudaSuccess)return mapped(status);
    if(device!=w->device)return SPACEPDHCG_CUDA_INVALID_ARGUMENT;
    const auto s=reinterpret_cast<cudaStream_t>(stream.native_handle);
    const int blocks=(w->n+127)/128;
#define TRY(call) do {status=(call);if(status!=cudaSuccess)return mapped(status);}while(false)
    TRY(cudaMemsetAsync(w->inside_count,0,sizeof(int),s));TRY(cudaMemsetAsync(w->invalid,0,sizeof(int),s));
    TRY(cudaMemsetAsync(count,0,sizeof(int),s));TRY(cudaMemsetAsync(output,0,w->n*sizeof(int64_t),s));
    bands<<<blocks,128,0,s>>>(w->data,w->bodies,w->pool,w->n,query,w->au,w->keys,w->order,w->inside,w->inside_count,w->invalid);
    TRY(cudaGetLastError());
    TRY(cub::DeviceRadixSort::SortPairs(w->temp,w->temp_bytes,w->keys,w->sorted,w->order,w->sorted_order,w->n,0,64,s));
    eligibility<<<blocks,128,0,s>>>(w->n,query,w->sorted,w->sorted_order,w->inside,w->inside_count,w->eligible);
    metrics<<<blocks,128,0,s>>>(w->data,w->pool,w->n,query,w->eligible,w->tofs,w->nt,w->mu,w->au,w->keys,w->positional,w->order,w->invalid);
    TRY(cudaGetLastError());
    TRY(cub::DeviceRadixSort::SortPairs(w->temp,w->temp_bytes,w->keys,w->sorted,w->order,w->sorted_order,w->n,0,64,s));
    collect_proxy<<<blocks,128,0,s>>>(w->data,w->pool,w->n,query,w->sorted,w->sorted_order,w->keep,output,count);
    TRY(cudaGetLastError());
    TRY(cub::DeviceRadixSort::SortPairs(w->temp,w->temp_bytes,w->positional,w->sorted,w->order,w->sorted_order,w->n,0,64,s));
    unique_position<<<blocks,128,0,s>>>(w->data,w->pool,w->n,query,w->sorted,w->sorted_order,output,count,w->keep);
    TRY(cudaGetLastError());
    TRY(cub::DeviceScan::ExclusiveSum(w->temp,w->temp_bytes,w->keep,w->prefix,w->n,s));
    compact<<<blocks,128,0,s>>>(w->data,w->pool,w->n,w->sorted_order,w->keep,w->prefix,output,count,w->invalid);
    finish<<<1,1,0,s>>>(w->n,w->keep,w->prefix,count,w->invalid);
    return mapped(cudaGetLastError());
#undef TRY
}

extern "C" spacepdhcg_cuda_status spacepdhcg_gtoc12_neighbours_destroy(spacepdhcg_gtoc12_neighbours** handle) {
    if(!handle||!*handle)return SPACEPDHCG_CUDA_INVALID_ARGUMENT;
    auto* w=*handle;std::unique_lock<std::mutex> lock(w->mutex,std::try_to_lock);
    if(!lock.owns_lock())return SPACEPDHCG_CUDA_BUSY;
    cudaStreamSynchronize(w->stream);
    for(void* p:{static_cast<void*>(w->raw),static_cast<void*>(w->data),static_cast<void*>(w->pool),
        static_cast<void*>(w->tofs),static_cast<void*>(w->keys),static_cast<void*>(w->sorted),static_cast<void*>(w->positional),
        static_cast<void*>(w->order),static_cast<void*>(w->sorted_order),static_cast<void*>(w->inside),static_cast<void*>(w->eligible),
        static_cast<void*>(w->keep),static_cast<void*>(w->prefix),static_cast<void*>(w->inside_count),static_cast<void*>(w->invalid),
        static_cast<void*>(w->count),static_cast<void*>(w->output),static_cast<void*>(w->query),w->temp})cudaFree(p);
    if(w->stream)cudaStreamDestroy(w->stream);
    lock.unlock();delete w;*handle=nullptr;return SPACEPDHCG_CUDA_SUCCESS;
}

extern "C" spacepdhcg_cuda_status spacepdhcg_gtoc12_neighbours_create(
    const Body* bodies,int nb,const int* pool,int n,const double* tofs,int nt,
    double mu,double au,int device,spacepdhcg_gtoc12_neighbours** handle) {
    if(!handle||*handle||!bodies||!pool||!tofs||nb<1||n<1||nt<1||n>nb
        ||!std::isfinite(mu)||mu<=0||!std::isfinite(au)||au<=0||device<0)
        return SPACEPDHCG_CUDA_INVALID_ARGUMENT;
    for(int i=0;i<nb;++i) {
        const auto& b=bodies[i];
        if(!std::isfinite(b.epoch)||!std::isfinite(b.a)||b.a<=0||!std::isfinite(b.e)||b.e<0||b.e>=1
            ||!std::isfinite(b.inclination)||!std::isfinite(b.node)||!std::isfinite(b.perihelion)||!std::isfinite(b.mean))
            return SPACEPDHCG_CUDA_INVALID_ARGUMENT;
    }
    for(int i=0;i<n;++i)if(pool[i]<0||pool[i]>=nb||(i&&bodies[pool[i-1]].id>=bodies[pool[i]].id))return SPACEPDHCG_CUDA_INVALID_ARGUMENT;
    for(int i=0;i<nt;++i)if(!std::isfinite(tofs[i])||tofs[i]<=0)return SPACEPDHCG_CUDA_INVALID_ARGUMENT;
    auto* w=new(std::nothrow)spacepdhcg_gtoc12_neighbours{};if(!w)return SPACEPDHCG_CUDA_OUT_OF_MEMORY;
    w->bodies=nb;w->n=n;w->nt=nt;w->mu=mu;w->au=au;w->device=device;
    auto status=cudaSetDevice(device);
#define ALLOC(field,bytes) if(status==cudaSuccess)status=cudaMalloc(reinterpret_cast<void**>(&w->field),(bytes))
    if(status==cudaSuccess)status=cudaStreamCreateWithFlags(&w->stream,cudaStreamNonBlocking);
    ALLOC(raw,size_t(nb)*sizeof(Body));ALLOC(data,size_t(nb)*sizeof(Derived));ALLOC(pool,size_t(n)*sizeof(int));ALLOC(tofs,size_t(nt)*sizeof(double));
    ALLOC(keys,size_t(n)*sizeof(double));ALLOC(sorted,size_t(n)*sizeof(double));ALLOC(positional,size_t(n)*sizeof(double));
    ALLOC(order,size_t(n)*sizeof(int));ALLOC(sorted_order,size_t(n)*sizeof(int));ALLOC(inside,size_t(n)*sizeof(int));ALLOC(eligible,size_t(n)*sizeof(int));
    ALLOC(keep,size_t(n)*sizeof(int));ALLOC(prefix,size_t(n)*sizeof(int));ALLOC(inside_count,sizeof(int));ALLOC(invalid,sizeof(int));ALLOC(count,sizeof(int));
    ALLOC(output,size_t(n)*sizeof(int64_t));ALLOC(query,sizeof(Query));
    size_t sort_bytes=0,scan_bytes=0;
    if(status==cudaSuccess)status=cub::DeviceRadixSort::SortPairs(nullptr,sort_bytes,w->keys,w->sorted,w->order,w->sorted_order,n,0,64,w->stream);
    if(status==cudaSuccess)status=cub::DeviceScan::ExclusiveSum(nullptr,scan_bytes,w->keep,w->prefix,n,w->stream);
    w->temp_bytes=std::max(sort_bytes,scan_bytes);ALLOC(temp,w->temp_bytes);
#undef ALLOC
    if(status==cudaSuccess)status=cudaMemcpyAsync(w->raw,bodies,size_t(nb)*sizeof(Body),cudaMemcpyHostToDevice,w->stream);
    if(status==cudaSuccess)status=cudaMemcpyAsync(w->pool,pool,size_t(n)*sizeof(int),cudaMemcpyHostToDevice,w->stream);
    if(status==cudaSuccess)status=cudaMemcpyAsync(w->tofs,tofs,size_t(nt)*sizeof(double),cudaMemcpyHostToDevice,w->stream);
    if(status==cudaSuccess){derive<<<(nb+127)/128,128,0,w->stream>>>(w->raw,w->data,nb,mu);status=cudaGetLastError();}
    const auto complete=cudaStreamSynchronize(w->stream);if(status==cudaSuccess)status=complete;
    if(status!=cudaSuccess){spacepdhcg_gtoc12_neighbours_destroy(&w);return mapped(status);}
    *handle=w;return SPACEPDHCG_CUDA_SUCCESS;
}

extern "C" spacepdhcg_cuda_status spacepdhcg_gtoc12_neighbours_host(
    spacepdhcg_gtoc12_neighbours* w,const Query* query,int64_t* output,int capacity,int* count) {
    if(!w||!query||!output||!count||query->neighbours<1||capacity<0)return SPACEPDHCG_CUDA_INVALID_ARGUMENT;
    const int required=static_cast<int>(std::min<int64_t>(w->n,int64_t(query->neighbours)+query->neighbours/2));
    if(capacity<required)return SPACEPDHCG_CUDA_INVALID_ARGUMENT;
    std::unique_lock<std::mutex> lock(w->mutex,std::try_to_lock);if(!lock.owns_lock())return SPACEPDHCG_CUDA_BUSY;
    int device=-1;auto s=cudaGetDevice(&device);if(s!=cudaSuccess)return mapped(s);
    if(device!=w->device)return SPACEPDHCG_CUDA_INVALID_ARGUMENT;
    s=cudaMemcpyAsync(w->query,query,sizeof(Query),cudaMemcpyHostToDevice,w->stream);
    spacepdhcg_cuda_status result= mapped(s);
    if(s==cudaSuccess)result=spacepdhcg_gtoc12_neighbours_launch_device(w,w->query,w->output,w->count,{{SPACEPDHCG_DEVICE_CUDA,w->device},reinterpret_cast<uintptr_t>(w->stream)});
    if(result==SPACEPDHCG_CUDA_SUCCESS)s=cudaMemcpyAsync(output,w->output,size_t(required)*sizeof(int64_t),cudaMemcpyDeviceToHost,w->stream);
    if(result==SPACEPDHCG_CUDA_SUCCESS&&s==cudaSuccess)s=cudaMemcpyAsync(count,w->count,sizeof(int),cudaMemcpyDeviceToHost,w->stream);
    const auto complete=cudaStreamSynchronize(w->stream);
    if(result!=SPACEPDHCG_CUDA_SUCCESS)return result;
    return mapped(s==cudaSuccess?complete:s);
}
