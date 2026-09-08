#include "../internal/gtoc12_workspace_reuse.h"
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

void require(bool ok,const char* message) {
    if(!ok){std::fprintf(stderr,"workspace reuse: %s\n",message);std::exit(1);}
}
int main() {
    for(int hold:{0,1}) for(int departure:{0,1}) for(int arrival:{0,1}) {
        const int k=7,n=k+1;
        std::vector<double> times(n),states(7*n),controls(4*n),fuel(n);
        double boundary[12]={2.6,0.2,0.1,0.0,0.6,0.0,2.7,0.3,0.1,-0.1,0.6,0.0};
        for(int i=0;i<n;++i) {
            times[i]=0.03*i;fuel[i]=0.001*(i+1);
            states[7*i]=2.6+0.01*i;states[7*i+1]=0.2;states[7*i+4]=0.6;states[7*i+6]=1.0-0.001*i;
            controls[4*i]=0.01;controls[4*i+3]=0.02;
        }
        spacepdhcg_gtoc12_conic* reused{};
        require(!spacepdhcg_gtoc12_conic_create(k,hold,departure,arrival,1.2,0.01,times.data(),boundary,fuel.data(),&reused),"initial creation");
        spacepdhcg_gtoc12_conic_dimensions dims{};
        require(!spacepdhcg_gtoc12_conic_get_dimensions(reused,&dims),"dimensions");
        const auto count=dims.a_nonzeros+dims.rows+dims.variables+dims.p_nonzeros;
        std::vector<double> actual(count),expected(count),unchanged(count);
        spacepdhcg_gtoc12_conic_parameters params{0.2,0.3,10000.0,0.3,0.1,0.1,0.01};
        for(int trial=0;trial<4;++trial) {
            for(int i=0;i<n;++i){times[i]=2.0*trial+0.025*(i+0.03*trial*i*i);fuel[i]=0.002*(trial+1)*(i+1);}
            boundary[trial]+=0.1;
            const double kappa=0.8+0.3*trial,mass_flow=0.02+0.01*trial;
            require(!gtoc12_conic_rebind(reused,kappa,mass_flow,times.data(),boundary,fuel.data()),"rebind");
            spacepdhcg_gtoc12_conic* fresh{};
            require(!spacepdhcg_gtoc12_conic_create(k,hold,departure,arrival,kappa,mass_flow,times.data(),boundary,fuel.data(),&fresh),"fresh creation");
            require(!spacepdhcg_gtoc12_conic_evaluate_host(reused,states.data(),controls.data(),&params,8,actual.data()),"reused evaluation");
            require(!spacepdhcg_gtoc12_conic_evaluate_host(fresh,states.data(),controls.data(),&params,8,expected.data()),"fresh evaluation");
            require(actual==expected,"all coefficients/RHS/objective/Hessian bitwise equal");
            const auto saved=fuel[0];fuel[0]=-1.0;
            require(gtoc12_conic_rebind(reused,kappa,mass_flow,times.data(),boundary,fuel.data())==1,"reject invalid rebind");
            fuel[0]=saved;
            require(!spacepdhcg_gtoc12_conic_evaluate_host(reused,states.data(),controls.data(),&params,8,unchanged.data()),"evaluate after invalid input");
            require(unchanged==actual,"invalid rebind leaves inputs unchanged");
            spacepdhcg_gtoc12_conic_destroy(fresh);
        }
        spacepdhcg_gtoc12_conic_destroy(reused);
    }
    require(cudaGetLastError()==cudaSuccess,"CUDA completion");
    std::puts("32 refreshed/fresh conic pairs pass bitwise; invalid rebinds preserve state");
}
