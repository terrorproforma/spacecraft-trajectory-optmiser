// GTOC12 zero-revolution Lambert seed and independent-node Kepler sampling.
// The scan is distributed across 66 blocks; only two scalar root refinements
// and the branch comparison are serial. All trajectory sampling is parallel.
#pragma once
#include <cuda_runtime.h>
#include <cmath>

namespace gtoc12_seed {
constexpr int samples=8192, points=samples+1, scan_blocks=(points+255)/256;
constexpr double pi=3.141592653589793238462643383279502884;
constexpr double du=1.49597870691e8, mu=1.32712440018e11;
struct Geometry { double r1,r2,a; int valid; };
struct Branch { double v1[3],v2[3]; int valid; };
__device__ double dot(const double* x,const double* y) { return x[0]*y[0]+x[1]*y[1]+x[2]*y[2]; }
__device__ void stumpff(double z,double& c,double& s) {
    if (z>1e-8) { double r=sqrt(z); c=(1-cos(r))/z; s=(r-sin(r))/(r*r*r); }
    else if (z<-1e-8) { double r=sqrt(-z); c=(cosh(r)-1)/(-z); s=(sinh(r)-r)/(r*r*r); }
    else { c=.5-z/24+z*z/720-z*z*z/40320; s=1.0/6-z/120+z*z/5040-z*z*z/362880; }
}
__device__ Geometry geometry(const double* b,int direction) {
    const double r1=sqrt(dot(b,b))*du,r2=sqrt(dot(b+6,b+6))*du;
    const double cosine=fmin(1.0,fmax(-1.0,dot(b,b+6)*du*du/(r1*r2)));
    const double sine=(direction ? -1.0 : 1.0)*sqrt(fmax(0.0,1-cosine*cosine));
    const double a=sine*sqrt(r1*r2/(1-cosine));
    return {r1,r2,a,int(r1>0 && r2>0 && 1-cosine>1e-14 && fabs(sine)>1e-14 && fabs(a)>1e-14)};
}
__device__ double grid(int index) { return -4*pi*pi+(8*pi*pi-1e-8)*index/samples; }
__device__ double residual(double z,Geometry g,double tof,double* out_y=nullptr) {
    double c,s; stumpff(z,c,s);
    if (!g.valid || !isfinite(c) || !isfinite(s) || c<=0) return NAN;
    const double y=g.r1+g.r2+g.a*(z*s-1)/sqrt(c);
    if (!isfinite(y) || y<0) return NAN;
    if (out_y) *out_y=y;
    const double x=sqrt(y/c);
    return (x*x*x*s+g.a*sqrt(y))/sqrt(mu)-tof;
}
__global__ void evaluate(const double* times,int nodes,const double* b,double* values,int* last) {
    __shared__ int indices[256];
    const int index=blockIdx.x*256+threadIdx.x,direction=blockIdx.y;
    double value=NAN;
    if (index<points) {
        const double tu=sqrt(du*du*du/mu);
        value=residual(grid(index),geometry(b,direction),(times[nodes-1]-times[0])*tu);
        values[direction*points+index]=value;
    }
    indices[threadIdx.x]=isfinite(value) ? index : -1;
    __syncthreads();
    for (int offset=128;offset;offset/=2) {
        if (threadIdx.x<offset) indices[threadIdx.x]=max(indices[threadIdx.x],indices[threadIdx.x+offset]);
        __syncthreads();
    }
    if (!threadIdx.x) last[direction*scan_blocks+blockIdx.x]=indices[0];
}
__global__ void brackets(const double* values,const int* last,int* first) {
    const int index=blockIdx.x*256+threadIdx.x,direction=blockIdx.y;
    if (index>=points) return;
    const auto* v=values+direction*points;
    if (!isfinite(v[index])) return;
    int previous=index-1;
    const int begin=blockIdx.x*256;
    while (previous>=begin && !isfinite(v[previous])) --previous;
    if (previous<begin) {
        previous=-1;
        for (int block=int(blockIdx.x)-1;block>=0 && previous<0;--block)
            previous=last[direction*scan_blocks+block];
    }
    if (fabs(v[index])<=1e-8 || (previous>=0 && v[previous]*v[index]<0))
        atomicMin(first+direction,index);
}
__global__ void refine(const double* times,int nodes,const double* b,const double* values,
    const int* first,Branch* branches) {
    const int direction=threadIdx.x;
    if (direction>=2) return;
    auto& branch=branches[direction]; branch={};
    const int index=first[direction];
    if (index>=points) return;
    const Geometry g=geometry(b,direction);
    const double tu=sqrt(du*du*du/mu),tof=(times[nodes-1]-times[0])*tu;
    double z=grid(index);
    if (fabs(values[direction*points+index])>1e-8) {
        int previous=index-1;
        while (previous>=0 && !isfinite(values[direction*points+previous])) --previous;
        if (previous<0) return;
        double lo=grid(previous),hi=z,low=values[direction*points+previous];
        bool converged=false;
        for (int it=0;it<256;++it) {
            z=.5*(lo+hi);
            const double value=residual(z,g,tof);
            if (!isfinite(value)) { lo=z; continue; }
            if (fabs(value)<=1e-8 || fabs(hi-lo)<=1e-13) { converged=true; break; }
            if (low*value<0) hi=z;
            else { lo=z; low=value; }
        }
        if (!converged) return;
    }
    double y{};
    if (!isfinite(residual(z,g,tof,&y))) return;
    const double f=1-y/g.r1,gg=g.a*sqrt(y/mu),gd=1-y/g.r2;
    if (!isfinite(gg) || fabs(gg)<=1e-14) return;
    for (int j=0;j<3;++j) {
        branch.v1[j]=(b[6+j]*du-f*b[j]*du)/gg;
        branch.v2[j]=(gd*b[6+j]*du-b[j]*du)/gg;
    }
    branch.valid=1;
}
__global__ void choose(const Branch* branches,const double* b,int* selected) {
    double best=INFINITY;
    *selected=-1;
    const double vu=du/sqrt(du*du*du/mu);
    for (int i=0;i<2;++i) if (branches[i].valid) {
        double dv1=0,dv2=0;
        for (int j=0;j<3;++j) {
            const double a=branches[i].v1[j]-b[3+j]*vu,c=branches[i].v2[j]-b[9+j]*vu;
            dv1+=a*a; dv2+=c*c;
        }
        const double cost=sqrt(dv1)+sqrt(dv2);
        if (cost<best) { best=cost; *selected=i; }
    }
}
__device__ void equation(double chi,double alpha,double radius,double radial,double dt,double& f,double& df) {
    double c,s; const double z=alpha*chi*chi; stumpff(z,c,s);
    f=radial*chi*chi*c+(1-alpha*radius)*chi*chi*chi*s+radius*chi-sqrt(mu)*dt;
    df=radial*chi*(1-z*s)+(1-alpha*radius)*chi*chi*c+radius;
}
__global__ void sample(int nodes,const double* times,const double* b,const Branch* branches,
    const int* selected,int free_dep,int free_arr,double vinf,double* states,double* controls,int* invalid) {
    const double tu=sqrt(du*du*du/mu),vu=du/tu;
    for (int node=blockIdx.x*blockDim.x+threadIdx.x;node<nodes;node+=blockDim.x*gridDim.x) {
        double* x=states+7*node;
        if (*selected<0) {
            const double a=(times[node]-times[0])/(times[nodes-1]-times[0]);
            for (int j=0;j<6;++j) x[j]=(1-a)*b[j]+a*b[6+j];
        } else {
            const auto& branch=branches[*selected];
            double r[3]; for (int j=0;j<3;++j) r[j]=b[j]*du;
            const double radius=sqrt(dot(r,r)),radial=dot(r,branch.v1)/sqrt(mu);
            const double alpha=2/radius-dot(branch.v1,branch.v1)/mu,dt=(times[node]-times[0])*tu;
            double lo=0,hi=fabs(alpha)>1e-12 ? sqrt(mu)*fabs(alpha)*dt : sqrt(radius);
            if (hi==0) hi=sqrt(radius);
            for (int it=0;it<200;++it) {
                double f,df; equation(hi,alpha,radius,radial,dt,f,df);
                if (!(f<0)) break;
                lo=hi; hi*=2;
            }
            double chi=.5*(lo+hi);
            for (int it=0;it<200;++it) {
                double f,df; equation(chi,alpha,radius,radial,dt,f,df);
                if (f>0) hi=chi; else lo=chi;
                const double next=chi-f/(df>0 ? df : 1);
                const double updated=(next-lo)*(next-hi)<0 && df>0 ? next : .5*(lo+hi);
                const double step=updated-chi; chi=updated;
                if (fabs(step)<=1e-15*fmax(1.0,fabs(chi))) break;
            }
            double c,s; const double z=alpha*chi*chi; stumpff(z,c,s);
            const double f=1-chi*chi*c/radius,g=dt-chi*chi*chi*s/sqrt(mu);
            double next[3]; for (int j=0;j<3;++j) next[j]=f*r[j]+g*branch.v1[j];
            const double radius1=sqrt(dot(next,next));
            const double fd=sqrt(mu)*chi*(z*s-1)/(radius*radius1),gd=1-chi*chi*c/radius1;
            for (int j=0;j<3;++j) { x[j]=next[j]/du; x[3+j]=(fd*r[j]+gd*branch.v1[j])/vu; }
        }
        if (node==0 || node==nodes-1) {
            const double* end=b+(node==0 ? 0 : 6);
            const int free=node==0 ? free_dep : free_arr;
            double offset[3]; for (int j=0;j<3;++j) offset[j]=x[3+j]-end[3+j];
            const double norm=sqrt(dot(offset,offset));
            for (int j=0;j<3;++j) {
                x[j]=end[j];
                if (!free) x[3+j]=end[3+j];
                else if (norm>vinf && norm>0) x[3+j]=end[3+j]+offset[j]*(.98*vinf/norm);
            }
        }
        x[6]=1;
        for (int j=0;j<4;++j) controls[4*node+j]=j==3 ? 1e-3 : 0;
        for (int j=0;j<7;++j) if (!isfinite(x[j])) atomicExch(invalid,1);
    }
}
struct Scratch {
    double *times{},*boundary{},*values{};
    int *last{},*first{},*selected{},*invalid{};
    Branch* branches{};
    ~Scratch() {
        cudaFree(times); cudaFree(boundary); cudaFree(values); cudaFree(last); cudaFree(first);
        cudaFree(selected); cudaFree(invalid); cudaFree(branches);
    }
    bool create(int nodes,const double* t,const double* b,cudaStream_t stream) {
        return cudaMalloc(&times,nodes*sizeof(double))==cudaSuccess
            && cudaMalloc(&boundary,12*sizeof(double))==cudaSuccess
            && cudaMalloc(&values,2*points*sizeof(double))==cudaSuccess
            && cudaMalloc(&last,2*scan_blocks*sizeof(int))==cudaSuccess
            && cudaMalloc(&first,2*sizeof(int))==cudaSuccess
            && cudaMalloc(&selected,sizeof(int))==cudaSuccess
            && cudaMalloc(&invalid,sizeof(int))==cudaSuccess
            && cudaMalloc(&branches,2*sizeof(Branch))==cudaSuccess
            && cudaMemcpyAsync(times,t,nodes*sizeof(double),cudaMemcpyHostToDevice,stream)==cudaSuccess
            && cudaMemcpyAsync(boundary,b,12*sizeof(double),cudaMemcpyHostToDevice,stream)==cudaSuccess;
    }
    int launch(int nodes,int free_dep,int free_arr,double vinf,double* states,double* controls,cudaStream_t stream) {
        // 0x7f7f7f7f sentinel is larger than every sample index.
        if (cudaMemsetAsync(first,0x7f,2*sizeof(int),stream)!=cudaSuccess
            || cudaMemsetAsync(invalid,0,sizeof(int),stream)!=cudaSuccess) return 2;
        evaluate<<<dim3(scan_blocks,2),256,0,stream>>>(times,nodes,boundary,values,last);
        brackets<<<dim3(scan_blocks,2),256,0,stream>>>(values,last,first);
        refine<<<1,2,0,stream>>>(times,nodes,boundary,values,first,branches);
        choose<<<1,1,0,stream>>>(branches,boundary,selected);
        sample<<<min(256,(nodes+127)/128),128,0,stream>>>(nodes,times,boundary,branches,selected,
            free_dep,free_arr,vinf,states,controls,invalid);
        return cudaGetLastError()==cudaSuccess ? 0 : 2;
    }
};
}
