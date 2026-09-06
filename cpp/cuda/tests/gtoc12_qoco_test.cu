#include "spacepdhcg/cuda/gtoc12_qoco_c_api.h"
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <vector>

namespace {
void require(bool value,const char* message) {
    if (!value) { std::fprintf(stderr,"FAIL: %s\n",message); std::exit(1); }
}
void check(cudaError_t status) { require(status==cudaSuccess,cudaGetErrorString(status)); }
template<class T> struct Device {
    T* data{};
    explicit Device(size_t n) { check(cudaMalloc(&data,n*sizeof(T))); }
    ~Device() { cudaFree(data); }
};
}
int main() {
    require(std::getenv("SPACEPDHCG_QOCO_LIBRARY")!=nullptr,"explicit GPU QOCO library");
    const int k=3,n=4,variables=25*n-8;
    for (int hold : {0,1}) {
        const double omega=1.0/std::sqrt(2.7*2.7*2.7);
        std::vector<double> times(n),states(7*n),controls(4*n),fuel(n,0.003),result(variables);
        for (int i=0;i<n;++i) {
            times[i]=0.003*i;
            states[7*i]=2.7*std::cos(omega*times[i]); states[7*i+1]=2.7*std::sin(omega*times[i]);
            states[7*i+3]=-2.7*omega*std::sin(omega*times[i]); states[7*i+4]=2.7*omega*std::cos(omega*times[i]);
            states[7*i+6]=1.0;
        }
        if (!hold) fuel.back()=0.0;
        double boundary[12];
        for (int i=0;i<6;++i) { boundary[i]=states[i]; boundary[6+i]=states[7*k+i]; }
        spacepdhcg_gtoc12_qoco* w{};
        require(spacepdhcg_gtoc12_qoco_create(k,hold,1,1,0.15,0.03,times.data(),boundary,fuel.data(),1e-9,0,&w)==0,"create");
        spacepdhcg_gtoc12_conic_dimensions d{};
        require(spacepdhcg_gtoc12_qoco_get_dimensions(w,&d)==0 && d.variables==variables,"dimensions");
        spacepdhcg_gtoc12_conic_parameters p{0.1,0.3,13.0,0.3,0.05,0.02,0.001};
        Device<double> ds(states.size()),du(controls.size()); Device<spacepdhcg_gtoc12_conic_parameters> dp(1);
        cudaStream_t stream{}; check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
        for (int repeat=0;repeat<3;++repeat) {
            p.trust_state=0.1/(repeat+1); p.virtual_weight=13.0+repeat;
            check(cudaMemcpyAsync(ds.data,states.data(),states.size()*sizeof(double),cudaMemcpyHostToDevice,stream));
            check(cudaMemcpyAsync(du.data,controls.data(),controls.size()*sizeof(double),cudaMemcpyHostToDevice,stream));
            check(cudaMemcpyAsync(dp.data,&p,sizeof(p),cudaMemcpyHostToDevice,stream));
            spacepdhcg_gtoc12_qoco_report report{};
            const int code=spacepdhcg_gtoc12_qoco_solve_device(w,ds.data,du.data,dp.data,8,stream,&report);
            std::printf("hold=%d repeat=%d api=%d qoco=%d iters=%d residuals=%.12g,%.12g updates=%llu\n",
                hold,repeat,code,report.qoco_status,report.iterations,report.primal_residual,report.dual_residual,
                static_cast<unsigned long long>(report.device_numeric_updates));
            std::fflush(stdout);
            require(code==0 && report.qualified,"qualified device conic solve");
            require(report.requested_tolerance==1e-9 && report.primal_residual<=1e-9
                && report.dual_residual<=1e-9 && report.relative_gap<=1e-9,"explicit audit and gap gate");
            require(report.workspace_creations==1 && report.solves==static_cast<unsigned>(repeat+1),"retained solver");
            require(report.device_numeric_updates==static_cast<unsigned>(repeat),"device numeric updates");
            const double* primal{}; require(spacepdhcg_gtoc12_qoco_primal(w,&primal)==0,"device primal");
            check(cudaMemcpyAsync(result.data(),primal,variables*sizeof(double),cudaMemcpyDeviceToHost,stream));
            check(cudaStreamSynchronize(stream));
            double objective=0.0;
            for (int i=0;i<n;++i) objective+=fuel[i]*result[7*n+4*i+3];
            for (int i=11*n+7*k;i<11*n+14*k;++i) objective+=p.virtual_weight*result[i];
            require(std::abs(objective)<=1e-7,"known coast objective zero");
            double quadratic=0.0;
            for (int i=0;i<k;++i) for (int j=0;j<3;++j) {
                const double difference=result[7*n+4*(i+1)+j]-result[7*n+4*i+j];
                quadratic+=p.smoothness_weight*difference*difference;
            }
            require(std::abs(report.primal_objective-objective-quadratic)<=1e-12,"independent objective audit");
            require(std::abs(result[7*k+6]-1.0)<=1e-5,"known coast mass");
        }
        p.minimum_mass=-1.0;
        spacepdhcg_gtoc12_qoco_report rejected{};
        std::fill(result.begin(),result.end(),-123.0);
        require(spacepdhcg_gtoc12_qoco_solve_host(w,states.data(),controls.data(),&p,8,result.data(),&rejected)==3
            && !rejected.qualified,"invalid input rejected");
        for (double x:result) require(x==-123.0,"failed bridge leaves host primal untouched");
        p.minimum_mass=0.3;
        require(spacepdhcg_gtoc12_qoco_solve_host(w,states.data(),controls.data(),&p,8,result.data(),&rejected)==0,"recovery and bridge");
        check(cudaStreamDestroy(stream)); spacepdhcg_gtoc12_qoco_destroy(w);
        require(spacepdhcg_gtoc12_qoco_create(k,hold,1,1,0.15,0.03,times.data(),boundary,fuel.data(),0.0,0,&w)==1 && !w,"invalid tolerance");
    }
    std::puts("GTOC12 QOCO device/host solve, repeated numerical updates, qualification and lifecycle PASS");
}
