// Exercise the actual prepared QOCO scan and iterate-update entrypoints.
// No factorization: sanitizers can isolate scan/guard/stream-lifetime errors.
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include "../algebra/cuda/cuda_types.h"
extern "C" {
#include "cone.h"
void qoco_gpu_reset_control(QOCOSolver*);
void qoco_gpu_free_control(QOCOWorkspace*);
void qoco_gpu_scan_direction(QOCOWorkspace*);
const int* qoco_gpu_direction_flag(QOCOWorkspace*);
void qoco_gpu_take_step(QOCOSolver*);
void qoco_gpu_sync_control(QOCOSolver*);
int qoco_gpu_begin_reduction_scope();
void qoco_gpu_end_reduction_scope();
}
bool load_cuda_libraries();
#define REQUIRE(x) do { if (!(x)) { std::fprintf(stderr,"line %d: %s\n",__LINE__,#x); std::exit(1); } } while(0)
#define CUDA(x) REQUIRE((x)==cudaSuccess)
struct Vector {
    QOCOVectorf v{};
    double* allocation{};
    std::vector<double> original;
    Vector(int n,double value):original(n,value) {
        std::vector<double> padded(n+2,value); padded.front()=padded.back()=-999;
        CUDA(cudaMalloc(&allocation,padded.size()*sizeof(double)));
        CUDA(cudaMemcpy(allocation,padded.data(),padded.size()*sizeof(double),cudaMemcpyHostToDevice));
        v.d_data=allocation+1;v.len=n;v.data=original.data();
    }
    void put(const std::vector<double>& values) {
        REQUIRE(values.size()==original.size());
        if (!values.empty()) CUDA(cudaMemcpy(v.d_data,values.data(),values.size()*sizeof(double),cudaMemcpyHostToDevice));
    }
    std::vector<double> read() {
        std::vector<double> padded(original.size()+2);
        CUDA(cudaMemcpy(padded.data(),allocation,padded.size()*sizeof(double),cudaMemcpyDeviceToHost));
        REQUIRE(padded.front()==-999 && padded.back()==-999);
        return {padded.begin()+1,padded.end()-1};
    }
    ~Vector(){CUDA(cudaFree(allocation));}
};
static void run(int n,int p,int lp,const std::vector<int>& sizes,bool audit) {
    int m=lp; std::vector<int> starts;
    for(int q:sizes){starts.push_back(m);m+=q;}
    Vector x(n,2),y(p,3),s(m,4),z(m,5),ds(m,0),xyz(n+p+m,.25);
    auto clean=xyz.original;
    std::fill(clean.begin()+n+p,clean.end(),0);
    for(int i=0;i<static_cast<int>(sizes.size());++i){
        s.original[starts[i]]=4*sizes[i];z.original[starts[i]]=5*sizes[i];
    }
    QOCOProblemData data{};data.n=n;data.p=p;data.m=m;data.l=lp;data.nsoc=sizes.size();
    data.q=new_qoco_vectori(sizes.data(),sizes.size());
    QOCOWorkspace w{};w.data=&data;w.x=&x.v;w.y=&y.v;w.s=&s.v;w.z=&z.v;w.Ds=&ds.v;w.xyz=&xyz.v;
    w.soc_idx=new_qoco_vectori(starts.data(),starts.size());
    QOCOSettings settings{};settings.kkt_dynamic_reg=1e-13;
    QOCOSolution solution{};
    QOCOSolver solver{};solver.work=&w;solver.settings=&settings;solver.sol=&solution;
    qoco_gpu_reset_control(&solver);
    setenv("SPACEPDHCG_TEST_QOCO_DEVICE_STEPS_COMPARE",audit?"1":"0",1);
    std::vector<int> locations{0,n+p+m-1};
    if(n>256)locations.push_back(256);
    if(p)locations.push_back(n);
    if(m)locations.push_back(n+p);
    for(int location:locations) {
        for(int invalid=1;invalid>=0;--invalid) {
            x.put(x.original);y.put(y.original);s.put(s.original);z.put(z.original);
            auto direction=clean;
            if(invalid)direction[location]=NAN;
            xyz.put(direction);
            // Ds may already contain NaNs from the queued correction. A zero
            // step multiplied by NaN would corrupt the iterate; the guard must skip it.
            ds.put(std::vector<double>(m,invalid?NAN:0.0));
            qoco_gpu_scan_direction(&w);
            int flag=-1;CUDA(cudaMemcpy(&flag,qoco_gpu_direction_flag(&w),sizeof(flag),cudaMemcpyDeviceToHost));
            REQUIRE(flag==invalid && flag==check_nan(&xyz.v));
            qoco_gpu_take_step(&solver);
            qoco_gpu_sync_control(&solver);
            REQUIRE(w.a==(invalid?0.0:.99));
            const auto rx=x.read(),ry=y.read();
            for(int i=0;i<n;++i)REQUIRE(std::abs(rx[i]-(2+(invalid?0:.99*.25)))<1e-14);
            for(int i=0;i<p;++i)REQUIRE(std::abs(ry[i]-(3+(invalid?0:.99*.25)))<1e-14);
            REQUIRE(s.read()==s.original && z.read()==z.original);
            const auto unchanged=xyz.read();
            REQUIRE(std::memcmp(unchanged.data(),direction.data(),direction.size()*sizeof(double))==0);
            ds.read();
        }
    }
    for(double value:{INFINITY,-INFINITY}) {
        auto direction=clean;direction.back()=value;xyz.put(direction);qoco_gpu_scan_direction(&w);
        int flag=-1;CUDA(cudaMemcpy(&flag,qoco_gpu_direction_flag(&w),sizeof(flag),cudaMemcpyDeviceToHost));
        REQUIRE(flag==0 && check_nan(&xyz.v)==0); // Preserve NaN-only reference semantics.
    }
    qoco_gpu_free_control(&w);free_qoco_vectori(w.soc_idx);free_qoco_vectori(data.q);
}
int main() {
    REQUIRE(load_cuda_libraries());
    unsetenv("SPACEPDHCG_TEST_QOCO_DEVICE_CONTROL_DISABLE");
    for(int scoped=0;scoped<2;++scoped){
        if(scoped){REQUIRE(qoco_gpu_begin_reduction_scope()==0);REQUIRE(qoco_gpu_begin_reduction_scope()==0);}
        run(1,0,0,{},scoped==0);
        run(257,129,13,{3,5,257},scoped==0);
        run(257,129,262145,{},scoped==0);
        if(scoped){qoco_gpu_end_reduction_scope();qoco_gpu_end_reduction_scope();}
    }
    CUDA(cudaDeviceSynchronize());
    std::puts("PASS: actual QOCO NaN scan/guard, vector canaries, valid-after-invalid reset, Inf parity, audits and nested scopes");
}
