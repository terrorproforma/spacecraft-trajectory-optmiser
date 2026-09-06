// Independent finite cone shifts, non-finite parity, canaries, graph capture,
// and the compact pure-SOC identity initialisation invariant.
#include "qoco.h"
#include "../algebra/cuda/cuda_types.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>
extern "C" int qoco_gpu_begin_reduction_scope();
extern "C" void qoco_gpu_end_reduction_scope();
extern "C" cudaError_t qoco_gpu_acquire_scalar_workspace(size_t, double**, int*);
#define REQUIRE(x) do { if (!(x)) { std::fprintf(stderr,"line %d: %s\n",__LINE__,#x); std::exit(1); } } while(0)
template<class T> struct Buffer {
    T* p{};
    explicit Buffer(const std::vector<T>& values) {
        REQUIRE(cudaMalloc(&p, std::max(size_t(1), values.size()) * sizeof(T)) == cudaSuccess);
        if (!values.empty()) REQUIRE(cudaMemcpy(p, values.data(), values.size()*sizeof(T), cudaMemcpyHostToDevice) == cudaSuccess);
    }
    ~Buffer() { REQUIRE(cudaFree(p) == cudaSuccess); }
};
static void cones(int l, const std::vector<int>& sizes, int mode, bool capture) {
    std::vector<int> starts;
    int m = l;
    for (int size : sizes) { starts.push_back(m); m += size; }
    std::vector<double> input(m+2, 12345.0);
    for (int i=0; i<l; ++i) input[i+1] = mode==1 ? 4.0 : mode==2 ? 0.0 : double(i%7-2);
    for (size_t i=0; i<sizes.size(); ++i) {
        input[starts[i]+1] = mode==1 ? 100.0 : mode==2 ? 0.0 : -1.0;
        for (int j=1; j<sizes[i]; ++j) input[starts[i]+j+1] = mode==2 ? 0.0 : .25;
    }
    if (m && mode==3) input[1] = std::numeric_limits<double>::quiet_NaN();
    if (m && mode==4) input[1] = std::numeric_limits<double>::infinity();
    if (m && mode==5) input[m] = -std::numeric_limits<double>::infinity();
    if (m && mode==6) input[1+m/2] = std::numeric_limits<double>::quiet_NaN();
    Buffer<double> u(input);
    Buffer<int> q(sizes), offset(starts);
    QOCOVectori qview{const_cast<int*>(sizes.data()), q.p, int(sizes.size())};
    QOCOProblemData data{}; data.l=l; data.m=m; data.nsoc=int(sizes.size()); data.q=&qview;
    REQUIRE(qoco_gpu_begin_reduction_scope()==0);
    if (capture) {
        double* scratch{}; int temporary{};
        REQUIRE(qoco_gpu_acquire_scalar_workspace(size_t(l+sizes.size()+255)/256+3, &scratch, &temporary)==cudaSuccess);
        REQUIRE(!temporary);
        cudaGraph_t graph; cudaGraphExec_t executable;
        REQUIRE(cudaStreamBeginCapture(cudaStreamPerThread,cudaStreamCaptureModeThreadLocal)==cudaSuccess);
        bring2cone(u.p+1,offset.p,&data);
        REQUIRE(cudaStreamEndCapture(cudaStreamPerThread,&graph)==cudaSuccess);
        REQUIRE(cudaGraphInstantiate(&executable,graph,0)==cudaSuccess);
        REQUIRE(cudaGraphLaunch(executable,cudaStreamPerThread)==cudaSuccess);
        REQUIRE(cudaStreamSynchronize(cudaStreamPerThread)==cudaSuccess);
        REQUIRE(cudaGraphExecDestroy(executable)==cudaSuccess);
        REQUIRE(cudaGraphDestroy(graph)==cudaSuccess);
    } else bring2cone(u.p+1,offset.p,&data);
    std::vector<double> output(m+2);
    REQUIRE(cudaMemcpy(output.data(),u.p,output.size()*sizeof(double),cudaMemcpyDeviceToHost)==cudaSuccess);
    // Compare every returned bit against the retained original GPU implementation,
    // including NaN payloads; fingerprints below are only compact reporting.
    Buffer<double> reference(input);
    const char* previous = std::getenv("SPACEPDHCG_TEST_QOCO_HOST_INITIAL_CONE");
    const bool had_previous = previous != nullptr;
    const std::string saved = previous ? previous : "";
    REQUIRE(setenv("SPACEPDHCG_TEST_QOCO_HOST_INITIAL_CONE", "1", 1)==0);
    bring2cone(reference.p+1,offset.p,&data);
    if(had_previous) REQUIRE(setenv("SPACEPDHCG_TEST_QOCO_HOST_INITIAL_CONE",saved.c_str(),1)==0);
    else REQUIRE(unsetenv("SPACEPDHCG_TEST_QOCO_HOST_INITIAL_CONE")==0);
    std::vector<double> original(m+2);
    REQUIRE(cudaMemcpy(original.data(),reference.p,original.size()*sizeof(double),cudaMemcpyDeviceToHost)==cudaSuccess);
    REQUIRE(std::memcmp(output.data(),original.data(),original.size()*sizeof(double))==0);
    qoco_gpu_end_reduction_scope();
    REQUIRE(output.front()==input.front() && output.back()==input.back());
    if (mode<3) {
        double residual=-1e7;
        for(int i=0;i<l;++i) residual=std::max(residual,-input[i+1]);
        for(size_t i=0;i<sizes.size();++i) {
            double sum=0;
            for(int j=1;j<sizes[i];++j) sum+=input[starts[i]+j+1]*input[starts[i]+j+1];
            residual=std::max(residual,std::sqrt(sum)-input[starts[i]+1]);
        }
        auto expected=input;
        if(residual>=0) {
            const double shift=1.0+std::max(0.0,residual);
            for(int i=0;i<l;++i) expected[i+1]+=shift;
            for(int start:starts) expected[start+1]+=shift;
        }
        for(size_t i=0;i<output.size();++i)
            REQUIRE(std::abs(output[i]-expected[i]) <= 3e-12*(1+std::abs(expected[i])));
    }
    uint64_t hash=1469598103934665603ULL;
    for(double value:output) { uint64_t bits; std::memcpy(&bits,&value,sizeof(bits)); hash=(hash^bits)*1099511628211ULL; }
    std::printf("CONE {\"lp\":%d,\"soc\":%zu,\"mode\":%d,\"hash\":\"%016llx\"}\n",l,sizes.size(),mode,(unsigned long long)hash);
}
static void identity(int l) {
    std::vector<int> sizes{1,3,8}, starts;
    int count=l;
    for(int q:sizes) { starts.push_back(count); count+=q+1; }
    Buffer<double> values(std::vector<double>(count,7.0));
    Buffer<int> q(sizes), offset(starts);
    QOCOVectorf w{nullptr,values.p,count};
    QOCOVectori qs{sizes.data(),q.p,int(sizes.size())}, offsets{starts.data(),offset.p,int(starts.size())};
    QOCOProblemData data{}; data.l=l; data.nsoc=int(sizes.size()); data.q=&qs;
    set_nt_scaling_identity(&w,count,&offsets,&data);
    std::vector<double> output(count), expected(count);
    REQUIRE(cudaMemcpy(output.data(),values.p,count*sizeof(double),cudaMemcpyDeviceToHost)==cudaSuccess);
    for(int i=0;i<l;++i) expected[i]=1;
    for(int start:starts) expected[start]=expected[start+1]=1;
    for(int i=0;i<count;++i) REQUIRE(output[i]==expected[i]);
    std::printf("IDENTITY lp=%d PASS\n",l);
}
int main(int argc,char** argv) {
    set_cpu_mode(0);
    if(argc>1 && std::strcmp(argv[1],"identity")==0) { identity(0); identity(3); return 0; }
    const bool capture=argc>1 && std::strcmp(argv[1],"capture")==0;
    for(int l:{0,1,257,262145}) for(const auto& q: {std::vector<int>{},std::vector<int>{1,3,8},std::vector<int>(1031,3)})
        for(int mode=0;mode<7;++mode) cones(l,q,mode,capture);
    std::puts("PASS: 84 cone cases");
}
