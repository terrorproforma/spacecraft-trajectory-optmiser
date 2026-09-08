#include "../internal/native_qoco_gpu.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <vector>

void require(bool ok, const char* message) {
    if (!ok) { std::fprintf(stderr,"FAIL: %s\n",message); std::exit(1); }
}
void check(cudaError_t status) { require(status==cudaSuccess,cudaGetErrorString(status)); }
int main() {
    cudaStream_t stream{};
    check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    check(qoco_gpu_bound_types(0,0,nullptr,nullptr,nullptr,nullptr,nullptr,stream));
    require(qoco_gpu_bound_types(-1,0,nullptr,nullptr,nullptr,nullptr,nullptr,stream)==cudaErrorInvalidValue,"negative dimension");
    const double inf=std::numeric_limits<double>::infinity();
    const double nan=std::numeric_limits<double>::quiet_NaN();
    const double pairs[][2]={{-inf,inf},{-inf,2},{-2,inf},{-2,2},{2,2},
        {-0.0,0.0},{1,1.0000000000000002},{nan,1},{1,nan},{inf,inf},{-inf,-inf},{2,-2}};
    const unsigned char expected[]={0,1,2,3,4,4,3,255,255,0,0,3};
    for (const int scalar : {0,1,131073}) for (const int variable : {0,1,131075}) {
        const int count=scalar+variable;
        std::vector<double> lo(count+1),hi(count+1);
        std::vector<unsigned char> result(count);
        for (int i=0;i<count;++i) {lo[i+1]=pairs[i%12][0];hi[i+1]=pairs[i%12][1];}
        double *lower{},*upper{};
        check(cudaMalloc(&lower,lo.size()*sizeof(double)));
        check(cudaMalloc(&upper,hi.size()*sizeof(double)));
        check(cudaMemcpyAsync(lower,lo.data(),lo.size()*sizeof(double),cudaMemcpyHostToDevice,stream));
        check(cudaMemcpyAsync(upper,hi.data(),hi.size()*sizeof(double),cudaMemcpyHostToDevice,stream));
        check(qoco_gpu_bound_types(scalar,variable,lower+1,upper+1,lower+scalar+1,upper+scalar+1,result.data(),stream));
        for (int i=0;i<count;++i) require(result[i]==expected[i%12],"independent classification including grid-stride tail");
        if (scalar) require(qoco_gpu_bound_types(scalar,variable,nullptr,upper,lower,upper,result.data(),stream)==cudaErrorInvalidValue,"missing lower input");
        check(cudaFree(lower)); check(cudaFree(upper));
    }
    check(cudaStreamDestroy(stream));
    std::puts("GPU bound classification: independent free/one-sided/two-sided/equality/NaN/infinity/offset/empty/tail cases PASS");
}
