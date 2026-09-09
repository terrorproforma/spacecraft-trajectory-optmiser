#include "spacepdhcg/cuda/gtoc12_scvx_c_api.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#define REQUIRE(x) do { if(!(x)) { std::fprintf(stderr,"line %d: %s\n",__LINE__,#x); std::exit(1); } } while(0)

int main() {
    constexpr int nodes=4;
    constexpr double du=1.49597870691e8,mu=1.32712440018e11,day=86400.0;
    const double tu=std::sqrt(du*du*du/mu),vu=du/tu;
    const double times[nodes]={0,1.25*day/tu,3.25*day/tu,6.5*day/tu};
    double initial[7]={du,0,0,0,vu,0,3000};
    double thrust[3*nodes]={.3,.4,0, 0,0,0, 0,0,.6, 0,-0.0,0};
    double original_initial[7],original_thrust[3*nodes];
    std::memcpy(original_initial,initial,sizeof(initial));
    std::memcpy(original_thrust,thrust,sizeof(thrust));
    std::vector<double> reference(7*nodes),reference_u(4*nodes),scaled(7*nodes),scaled_u(4*nodes);
    REQUIRE(spacepdhcg_gtoc12_zoh_seed_evaluate_host(nodes,times,initial,thrust,
        reference.data(),reference_u.data())==0);
    int finite_calls=0,invalid_calls=0;
    for(double target:{3000.0,1500.0,3300.0}) {
        REQUIRE(spacepdhcg_gtoc12_scaled_zoh_seed_evaluate_host(nodes,times,initial,thrust,
            target,scaled.data(),scaled_u.data())==0);
        ++finite_calls;
        const double factor=target/3000.0;
        for(int i=0;i<nodes;++i) {
            for(int j=0;j<3;++j) {
                REQUIRE(std::fabs(scaled[7*i+j]-reference[7*i+j])*du<1e-3);
                REQUIRE(std::fabs(scaled[7*i+3+j]-reference[7*i+3+j])*vu<1e-9);
                REQUIRE(scaled_u[4*i+j]==(thrust[3*i+j]*factor)/0.6);
            }
            const double x=thrust[3*i]*factor,y=thrust[3*i+1]*factor,z=thrust[3*i+2]*factor;
            REQUIRE(std::fabs(scaled_u[4*i+3]-std::sqrt(x*x+y*y+z*z)/.6)<1e-15);
            REQUIRE(std::fabs(scaled[7*i+6]-reference[7*i+6])*target<1e-8);
        }
        const double expected=target-factor*(.5*1.25+.6*3.25)*day/(4000*9.80665);
        REQUIRE(std::fabs(scaled.back()*target-expected)<1e-8);
        if(target==3000.0) {
            REQUIRE(std::memcmp(reference.data(),scaled.data(),reference.size()*8)==0);
            REQUIRE(std::memcmp(reference_u.data(),scaled_u.data(),reference_u.size()*8)==0);
        }
        // Scaling up is not clipping: the initializer may exceed physical
        // thrust; subsequent SCvx/certification must enforce that bound.
        if(target==3300.0) REQUIRE(scaled_u[4*2+2]*.6>.6);
        REQUIRE(std::memcmp(initial,original_initial,sizeof(initial))==0);
        REQUIRE(std::memcmp(thrust,original_thrust,sizeof(thrust))==0);
    }
    for(double target:{0.0,-1.0,std::numeric_limits<double>::infinity(),std::numeric_limits<double>::quiet_NaN()}) {
        std::fill(scaled.begin(),scaled.end(),42.0);
        REQUIRE(spacepdhcg_gtoc12_scaled_zoh_seed_evaluate_host(nodes,times,initial,thrust,
            target,scaled.data(),scaled_u.data())==1);
        for(double value:scaled) REQUIRE(value==42.0);
        ++invalid_calls;
    }
    initial[6]=0.0;
    REQUIRE(spacepdhcg_gtoc12_scaled_zoh_seed_evaluate_host(nodes,times,initial,thrust,
        1500.0,scaled.data(),scaled_u.data())==1);
    ++invalid_calls;
    initial[6]=std::numeric_limits<double>::min();
    REQUIRE(spacepdhcg_gtoc12_scaled_zoh_seed_evaluate_host(nodes,times,initial,thrust,
        std::numeric_limits<double>::max(),scaled.data(),scaled_u.data())==3);
    for(double value:scaled) REQUIRE(std::isnan(value));
    ++invalid_calls;
    initial[6]=std::numeric_limits<double>::max();
    REQUIRE(spacepdhcg_gtoc12_scaled_zoh_seed_evaluate_host(nodes,times,initial,thrust,
        std::numeric_limits<double>::min(),scaled.data(),scaled_u.data())==3);
    for(double value:scaled) REQUIRE(std::isnan(value));
    ++invalid_calls;
    initial[6]=3000.0;
    // Finite scale/components can still underflow the squared physical norm.
    REQUIRE(spacepdhcg_gtoc12_scaled_zoh_seed_evaluate_host(nodes,times,initial,thrust,
        3e-197,scaled.data(),scaled_u.data())==3);
    for(double value:scaled) REQUIRE(std::isnan(value));
    ++invalid_calls;
    REQUIRE(cudaDeviceSynchronize()==cudaSuccess);
    std::printf("SCALED_ZOH_SEED_PASS reference_calls=1 finite_scaled_calls=%d invalid_calls=%d optimizer_calls=0\n",
        finite_calls,invalid_calls);
}
