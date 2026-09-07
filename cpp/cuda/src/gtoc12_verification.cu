#include "spacepdhcg/cuda/gtoc12_verification_c_api.h"
#include "dop853_coefficients.cuh"
#include <cmath>
#include <limits>
#include <new>
#include <thread>

namespace {
constexpr unsigned mask = 0xffffffffu;
constexpr double day = 86400.0;
struct Scratch { double y[7], temp[7], k[13][7], thrust[3]; int status; };

__device__ double radius(const double* y) {
    return sqrt(y[0]*y[0] + y[1]*y[1] + y[2]*y[2]);
}
__device__ double control(const spacepdhcg_verify_sample* samples, int n, double t, int c) {
    if (!samples) return 0.0;
    int lo=0, hi=n;
    while (lo<hi) { int mid=(lo+hi)/2; if (samples[mid].seconds<=t) lo=mid+1; else hi=mid; }
    int points=min(4,n), j=max(0,min(n-2,lo-1));
    int start=max(0,min(n-points,j-(points-1)/2));
    double value=0.0;
    for (int k=0;k<points;++k) {
        double weight=1.0;
        for (int m=0;m<points;++m) if (m!=k)
            weight *= (t-samples[start+m].seconds)/(samples[start+k].seconds-samples[start+m].seconds);
        value += weight*samples[start+k].thrust[c];
    }
    return value;
}
__device__ void rhs(Scratch& s, const double* y, double* f, double t,
                    const spacepdhcg_verify_sample* samples, int count, int lane) {
    if (lane<3) s.thrust[lane]=control(samples,count,t,lane);
    __syncwarp(mask);
    if (lane<3) f[lane]=y[lane+3];
    else if (lane<6) {
        double r=radius(y);
        f[lane]=-1.32712440018e11/(r*r*r)*y[lane-3]+1e-3*s.thrust[lane-3]/y[6];
    } else if (lane==6) f[6]=-sqrt(s.thrust[0]*s.thrust[0]+s.thrust[1]*s.thrust[1]+s.thrust[2]*s.thrust[2])/(4000.0*9.80665);
    __syncwarp(mask);
}
__device__ bool valid_state(const double* y, int lane) {
    bool bad=lane<7 && (!isfinite(y[lane]) || (lane==6 && y[lane]<=0.0));
    bad |= lane==0 && !(radius(y)>0.0 && isfinite(radius(y)));
    return !__any_sync(mask,bad);
}

// Each warp independently advances one trajectory. Components/stages cooperate
// in shared memory; trajectories and blocks never synchronize with each other.
__device__ void segment(Scratch& s, double begin, double end,
    const spacepdhcg_verify_sample* samples, int count, int budget,
    spacepdhcg_verify_result& result, int lane) {
    if (end<=begin) return;
    double interval=samples ? day : 5*day;
    double t=begin, h=fmin(interval,end-begin);
    int64_t grid=1;
    bool rejected=false;
    rhs(s,s.y,s.k[0],t,samples,count,lane);
    if (lane==0) ++result.evaluations;
    while(t<end) {
        double target=fmin(end,begin+grid*interval);
        h=fmin(h,target-t);
        // Do not cross a change of interpolation stencil inside a step.
        if (samples) {
            int lo=0,hi=count;
            while(lo<hi) { int mid=(lo+hi)/2; if(samples[mid].seconds<=t) lo=mid+1; else hi=mid; }
            if(lo<count) h=fmin(h,samples[lo].seconds-t);
        }
        if (lane==0) {
            if (result.accepted_steps+result.rejected_steps>=budget) s.status=3;
            else if (!(h>0.0) || t+h==t) s.status=4;
        }
        __syncwarp(mask);
        if(s.status) return;
        for(int stage=1;stage<12;++stage) {
            if(lane<7) {
                double sum=0.0;
                for(int j=0;j<stage;++j) sum+=dop853::A[stage][j]*s.k[j][lane];
                s.temp[lane]=s.y[lane]+h*sum;
            }
            __syncwarp(mask);
            if(!valid_state(s.temp,lane)) { if(lane==0) s.status=2; __syncwarp(mask); return; }
            rhs(s,s.temp,s.k[stage],t+dop853::C[stage]*h,samples,count,lane);
        }
        if(lane<7) {
            double sum=0.0;
            for(int j=0;j<12;++j) sum+=dop853::B[j]*s.k[j][lane];
            s.temp[lane]=s.y[lane]+h*sum;
        }
        __syncwarp(mask);
        if(!valid_state(s.temp,lane)) { if(lane==0) s.status=2; __syncwarp(mask); return; }
        rhs(s,s.temp,s.k[12],t+h,samples,count,lane);
        double e5=0.0,e3=0.0;
        if(lane<7) {
            double scale=(lane<3 ? 1e-7 : (lane<6 ? 1e-10 : 1e-9))+1e-12*fmax(fabs(s.y[lane]),fabs(s.temp[lane]));
            for(int j=0;j<13;++j) { e5+=dop853::E5[j]*s.k[j][lane]; e3+=dop853::E3[j]*s.k[j][lane]; }
            e5/=scale; e3/=scale;
        }
        e5*=e5; e3*=e3;
        for(int delta=16;delta>0;delta/=2) {
            e5+=__shfl_down_sync(mask,e5,delta); e3+=__shfl_down_sync(mask,e3,delta);
        }
        e5=__shfl_sync(mask,e5,0); e3=__shfl_sync(mask,e3,0);
        double error=(e5==0.0 && e3==0.0) ? 0.0 : h*e5/sqrt(7.0*(e5+0.01*e3));
        if(lane==0) { result.evaluations+=12; if(!isfinite(error)) s.status=2; }
        __syncwarp(mask);
        if(s.status) return;
        double factor=error==0.0 ? 10.0 : fmin(10.0,fmax(0.2,0.9*pow(error,-0.125)));
        if(error<1.0) {
            if(lane<7) { s.y[lane]=s.temp[lane]; s.k[0][lane]=s.k[12][lane]; }
            if(lane==0) ++result.accepted_steps;
            t+=h;
            __syncwarp(mask);
            if(t>=target) { if(lane==0) result.minimum_radius_km=fmin(result.minimum_radius_km,radius(s.y)); ++grid; }
            if(rejected) factor=fmin(1.0,factor);
            rejected=false;
        } else { if(lane==0) ++result.rejected_steps; rejected=true; }
        h*=factor;
    }
}

__global__ void verify(const spacepdhcg_verify_leg* legs, int nlegs,
    const spacepdhcg_verify_arc* arcs, int narcs,
    const spacepdhcg_verify_sample* samples, int nsamples, int budget,
    spacepdhcg_verify_result* results) {
    int lane=threadIdx.x%32, warp=threadIdx.x/32;
    unsigned id=blockIdx.x*4+warp;
    if(id>=static_cast<unsigned>(nlegs)) return;
    __shared__ Scratch scratch[4];
    auto& s=scratch[warp]; auto& result=results[id]; const auto leg=legs[id];
    if(lane==0) {
        result={}; s.status=0;
        bool valid=isfinite(leg.duration_s) && leg.duration_s>=0 &&
            leg.arc_offset>=0 && leg.arc_count>=0 && leg.arc_offset<=narcs && leg.arc_count<=narcs-leg.arc_offset;
        double previous=0.0;
        if(valid) for(int a=0;a<leg.arc_count;++a) {
            auto arc=arcs[leg.arc_offset+a];
            if(arc.sample_offset<0 || arc.sample_count<2 || arc.sample_offset>nsamples || arc.sample_count>nsamples-arc.sample_offset) { valid=false; break; }
            const auto* p=samples+arc.sample_offset;
            for(int j=0;j<arc.sample_count;++j) {
                if(!isfinite(p[j].seconds) || p[j].seconds<previous || (j && p[j].seconds<=p[j-1].seconds) || p[j].seconds>leg.duration_s) valid=false;
                for(int k=0;k<3;++k) if(!isfinite(p[j].thrust[k])) valid=false;
            }
            previous=p[arc.sample_count-1].seconds;
        }
        if(!valid) s.status=1;
    }
    if(lane<7) s.y[lane]=leg.initial[lane];
    __syncwarp(mask);
    if(!valid_state(s.y,lane)) { if(lane==0) s.status=1; }
    __syncwarp(mask);
    if(lane==0) result.minimum_radius_km=radius(s.y);
    double epoch=0.0;
    for(int a=0;a<leg.arc_count && !s.status;++a) {
        auto arc=arcs[leg.arc_offset+a]; const auto* p=samples+arc.sample_offset;
        segment(s,epoch,p[0].seconds,nullptr,0,budget,result,lane);
        if(s.status) break;
        segment(s,p[0].seconds,p[arc.sample_count-1].seconds,p,arc.sample_count,budget,result,lane);
        epoch=p[arc.sample_count-1].seconds;
    }
    if(!s.status) segment(s,epoch,leg.duration_s,nullptr,0,budget,result,lane);
    __syncwarp(mask);
    if(lane<7) result.final_state[lane]=s.status ? nan("") : s.y[lane];
    if(lane==0) { result.status=s.status; if(s.status) result.minimum_radius_km=nan(""); }
}
} // namespace

extern "C" cudaError_t spacepdhcg_gtoc12_verify_launch(
    const spacepdhcg_verify_leg* legs, int32_t leg_count,
    const spacepdhcg_verify_arc* arcs, int32_t arc_count,
    const spacepdhcg_verify_sample* samples, int32_t sample_count,
    int32_t max_steps, spacepdhcg_verify_result* results, cudaStream_t stream) {
    if(leg_count<0 || arc_count<0 || sample_count<0 || max_steps<1 ||
       (leg_count && (!legs || !results)) || (arc_count && !arcs) || (sample_count && !samples)) return cudaErrorInvalidValue;
    if(!leg_count) return cudaSuccess;
    verify<<<(static_cast<unsigned>(leg_count)+3)/4,128,0,stream>>>(legs,leg_count,arcs,arc_count,samples,sample_count,max_steps,results);
    return cudaGetLastError();
}

struct spacepdhcg_verify_workspace {
    int device=0, legs=0, arcs=0, samples=0;
    std::thread::id owner=std::this_thread::get_id();
    cudaStream_t stream=nullptr;
    spacepdhcg_verify_leg* dlegs=nullptr;
    spacepdhcg_verify_arc* darcs=nullptr;
    spacepdhcg_verify_sample* dsamples=nullptr;
    spacepdhcg_verify_result* results=nullptr;
};
namespace {
cudaError_t owned(spacepdhcg_verify_workspace* w) {
    if(!w || w->owner!=std::this_thread::get_id()) return cudaErrorInvalidResourceHandle;
    int device=-1; auto status=cudaGetDevice(&device);
    return status!=cudaSuccess ? status : device==w->device ? cudaSuccess : cudaErrorInvalidDevice;
}
}
extern "C" cudaError_t spacepdhcg_gtoc12_verify_destroy(spacepdhcg_verify_workspace** address) {
    if(!address) return cudaErrorInvalidValue;
    auto* w=*address;
    if(!w) return cudaSuccess;
    auto status=owned(w); if(status!=cudaSuccess) return status;
    status=cudaStreamSynchronize(w->stream);
    for(void* p : {static_cast<void*>(w->dlegs), static_cast<void*>(w->darcs), static_cast<void*>(w->dsamples), static_cast<void*>(w->results)}) {
        auto next=cudaFree(p); if(status==cudaSuccess) status=next;
    }
    auto next=cudaStreamDestroy(w->stream); if(status==cudaSuccess) status=next;
    delete w; *address=nullptr; return status;
}
extern "C" cudaError_t spacepdhcg_gtoc12_verify_create(int32_t legs, int32_t arcs, int32_t samples,
    spacepdhcg_verify_workspace** out) {
    if(!out) return cudaErrorInvalidValue;
    *out=nullptr;
    if(legs<1 || arcs<0 || samples<0) return cudaErrorInvalidValue;
    auto* w=new(std::nothrow) spacepdhcg_verify_workspace;
    if(!w) return cudaErrorMemoryAllocation;
    w->legs=legs; w->arcs=arcs; w->samples=samples;
    auto status=cudaGetDevice(&w->device);
    if(status==cudaSuccess) status=cudaStreamCreateWithFlags(&w->stream,cudaStreamNonBlocking);
    if(status!=cudaSuccess) { delete w; return status; }
    status=cudaMalloc(&w->dlegs,static_cast<size_t>(legs)*sizeof(*w->dlegs));
    if(status==cudaSuccess && arcs) status=cudaMalloc(&w->darcs,static_cast<size_t>(arcs)*sizeof(*w->darcs));
    if(status==cudaSuccess && samples) status=cudaMalloc(&w->dsamples,static_cast<size_t>(samples)*sizeof(*w->dsamples));
    if(status==cudaSuccess) status=cudaMalloc(&w->results,static_cast<size_t>(legs)*sizeof(*w->results));
    if(status!=cudaSuccess) { spacepdhcg_gtoc12_verify_destroy(&w); return status; }
    *out=w; return cudaSuccess;
}
extern "C" cudaError_t spacepdhcg_gtoc12_verify_host(spacepdhcg_verify_workspace* w,
    const spacepdhcg_verify_leg* legs, int32_t leg_count,
    const spacepdhcg_verify_arc* arcs, int32_t arc_count,
    const spacepdhcg_verify_sample* samples, int32_t sample_count,
    int32_t max_steps, spacepdhcg_verify_result* results) {
    auto status=owned(w); if(status!=cudaSuccess) return status;
    if(leg_count<0 || leg_count>w->legs || arc_count<0 || arc_count>w->arcs ||
       sample_count<0 || sample_count>w->samples || max_steps<1 ||
       (leg_count && (!legs || !results)) || (arc_count && !arcs) || (sample_count && !samples)) return cudaErrorInvalidValue;
    if(!leg_count) return cudaSuccess;
    status=cudaMemcpyAsync(w->dlegs,legs,static_cast<size_t>(leg_count)*sizeof(*legs),cudaMemcpyHostToDevice,w->stream);
    if(status==cudaSuccess && arc_count) status=cudaMemcpyAsync(w->darcs,arcs,static_cast<size_t>(arc_count)*sizeof(*arcs),cudaMemcpyHostToDevice,w->stream);
    if(status==cudaSuccess && sample_count) status=cudaMemcpyAsync(w->dsamples,samples,static_cast<size_t>(sample_count)*sizeof(*samples),cudaMemcpyHostToDevice,w->stream);
    if(status==cudaSuccess) status=spacepdhcg_gtoc12_verify_launch(w->dlegs,leg_count,w->darcs,arc_count,w->dsamples,sample_count,max_steps,w->results,w->stream);
    if(status==cudaSuccess) status=cudaMemcpyAsync(results,w->results,static_cast<size_t>(leg_count)*sizeof(*results),cudaMemcpyDeviceToHost,w->stream);
    auto completion=cudaStreamSynchronize(w->stream);
    return status==cudaSuccess ? completion : status;
}
