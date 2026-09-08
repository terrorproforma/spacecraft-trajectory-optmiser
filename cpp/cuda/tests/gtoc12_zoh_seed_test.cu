#include "spacepdhcg/cuda/gtoc12_verification_c_api.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <vector>

#define REQUIRE(x) do { if(!(x)) { std::fprintf(stderr,"line %d: %s\n",__LINE__,#x); std::exit(1); } } while(0)
#define CUDA(x) REQUIRE((x)==cudaSuccess)

int main() {
    constexpr int n=4;
    constexpr double du=1.49597870691e8,mu=1.32712440018e11,day=86400.0;
    const double tu=std::sqrt(du*du*du/mu),vu=du/tu;
    double times[n]={0,1.25*day/tu,3.25*day/tu,6.5*day/tu};
    double initial[7]={du,0,0,0,vu,0,3000};
    double thrust[3*n]={0.3,0.4,0, 0,0,0, 0,0,0.6, 0,0,0};
    double *dt,*di,*duv,*dx,*dc;
    spacepdhcg_verify_result* dr;
    cudaStream_t stream{}; CUDA(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    CUDA(cudaMalloc(&dt,sizeof(times))); CUDA(cudaMalloc(&di,sizeof(initial)));
    CUDA(cudaMalloc(&duv,sizeof(thrust))); CUDA(cudaMalloc(&dx,7*n*sizeof(double)));
    CUDA(cudaMalloc(&dc,4*n*sizeof(double))); CUDA(cudaMalloc(&dr,sizeof(*dr)));
    REQUIRE(spacepdhcg_gtoc12_verify_zoh_seed_launch(INT32_MAX/7,dt,di,duv,1,dx,dc,dr,stream)
        ==cudaErrorInvalidValue);
    auto upload=[&] {
        CUDA(cudaMemcpyAsync(dt,times,sizeof(times),cudaMemcpyHostToDevice,stream));
        CUDA(cudaMemcpyAsync(di,initial,sizeof(initial),cudaMemcpyHostToDevice,stream));
        CUDA(cudaMemcpyAsync(duv,thrust,sizeof(thrust),cudaMemcpyHostToDevice,stream));
    };
    upload(); CUDA(cudaStreamSynchronize(stream));
    cudaGraph_t graph{}; cudaGraphExec_t executable{};
    CUDA(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
    CUDA(spacepdhcg_gtoc12_verify_zoh_seed_launch(n,dt,di,duv,10000,dx,dc,dr,stream));
    CUDA(cudaStreamEndCapture(stream,&graph)); CUDA(cudaGraphInstantiate(&executable,graph,0));
    std::vector<double> states(7*n),controls(4*n);
    spacepdhcg_verify_result result{};
    auto download=[&] {
        CUDA(cudaMemcpyAsync(states.data(),dx,states.size()*8,cudaMemcpyDeviceToHost,stream));
        CUDA(cudaMemcpyAsync(controls.data(),dc,controls.size()*8,cudaMemcpyDeviceToHost,stream));
        CUDA(cudaMemcpyAsync(&result,dr,sizeof(result),cudaMemcpyDeviceToHost,stream));
        CUDA(cudaStreamSynchronize(stream));
    };
    for(int test=0;test<7;++test) {
        times[1]=1.25*day/tu; initial[6]=3000; thrust[11]=0;
        if(test==1) times[1]=times[0];
        if(test==2) initial[6]=-1;
        if(test==3) thrust[11]=0.1; // inactive final sample must be zero
        if(test==4) times[1]=std::numeric_limits<double>::quiet_NaN();
        if(test==5) thrust[11]=std::numeric_limits<double>::quiet_NaN();
        upload(); CUDA(cudaGraphLaunch(executable,stream)); download();
        if(test>0 && test<6) {
            REQUIRE(result.status==1);
            for(double x:states) REQUIRE(std::isnan(x));
            for(double x:controls) REQUIRE(std::isnan(x));
            continue;
        }
        REQUIRE(result.status==0 && result.accepted_steps>0);
        const double expected_mass=3000-(0.5*1.25+0.6*3.25)*day/(4000*9.80665);
        REQUIRE(std::fabs(result.final_state[6]-expected_mass)<1e-9);
        for(int j=0;j<7;++j) REQUIRE(states[j]==initial[j]/(j<3 ? du : j<6 ? vu : initial[6]));
        for(int i=0;i<n;++i) {
            for(int j=0;j<3;++j) REQUIRE(controls[4*i+j]==thrust[3*i+j]/0.6);
            REQUIRE(std::fabs(controls[4*i+3]-(i==0 ? 5.0/6.0 : i==2 ? 1.0 : 0.0))<1e-15);
        }
        REQUIRE(states[13]==states[20]); // coast consumes no mass
    }
    // Compare physical endpoints with the independent arc dispatch: identical
    // piecewise-constant control, including the intervening coast.
    spacepdhcg_verify_workspace* w{}; CUDA(spacepdhcg_gtoc12_verify_create(1,2,4,&w));
    spacepdhcg_verify_leg leg{};
    for(int j=0;j<7;++j) leg.initial[j]=initial[j];
    leg.duration_s=6.5*day; leg.arc_count=2;
    spacepdhcg_verify_arc arcs[2]={{0,2},{2,2}};
    spacepdhcg_verify_sample samples[4]={{0,{.3,.4,0}},{1.25*day,{.3,.4,0}},
        {3.25*day,{0,0,.6}},{6.5*day,{0,0,.6}}};
    spacepdhcg_verify_result independently_dispatched{};
    CUDA(spacepdhcg_gtoc12_verify_host(w,&leg,1,arcs,2,samples,4,10000,&independently_dispatched));
    REQUIRE(independently_dispatched.status==0);
    for(int j=0;j<7;++j) REQUIRE(std::fabs(result.final_state[j]-independently_dispatched.final_state[j])<(j<3 ? 2e-6 : 1e-9));
    CUDA(spacepdhcg_gtoc12_verify_destroy(&w));
    // Global budget exhaustion invalidates every partially produced node.
    CUDA(spacepdhcg_gtoc12_verify_zoh_seed_launch(n,dt,di,duv,1,dx,dc,dr,stream)); download();
    REQUIRE(result.status==3);
    for(double x:states) REQUIRE(std::isnan(x));
    for(double x:controls) REQUIRE(std::isnan(x));
    // All-coast circular orbit: analytic position and mass at every node.
    for(double& value:thrust) value=0;
    upload(); CUDA(cudaGraphLaunch(executable,stream)); download();
    REQUIRE(result.status==0);
    for(int i=0;i<n;++i) {
        REQUIRE(std::fabs(states[7*i]-std::cos(times[i]))*du<2e-6);
        REQUIRE(std::fabs(states[7*i+1]-std::sin(times[i]))*du<2e-6);
        REQUIRE(states[7*i+6]==1.0 && controls[4*i+3]==0.0);
    }
    CUDA(cudaGraphExecDestroy(executable)); CUDA(cudaGraphDestroy(graph));
    CUDA(cudaFree(dt)); CUDA(cudaFree(di)); CUDA(cudaFree(duv)); CUDA(cudaFree(dx)); CUDA(cudaFree(dc)); CUDA(cudaFree(dr));
    CUDA(cudaStreamDestroy(stream));
    std::puts("PASS: exact ZOH schedule, physical/scaled units, analytic mass/coast, graph replay, invalid inputs, global step budget");
}
