// Pure Lorentz-cone QP with a closed-form projection optimum, repeatedly updated.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include "qoco.h"
extern "C" int qoco_gpu_begin_reduction_scope();
extern "C" void qoco_gpu_end_reduction_scope();
extern "C" int qoco_gpu_primal_start(QOCOSolver*, int);
#define REQUIRE(x) do { if (!(x)) { std::fprintf(stderr,"line %d: %s\n",__LINE__,#x); std::exit(1); } } while(0)
int main() {
    double px[3]{1,1,1}, gx[3]{-1,-1,-1}, c[3]{0,-1,2}, h[3]{};
    int offsets[4]{0,1,2,3}, indices[3]{0,1,2}, size=3;
    QOCOCscMatrix P{},G{};
    qoco_set_csc(&P,3,3,3,px,offsets,indices);
    qoco_set_csc(&G,3,3,3,gx,offsets,indices);
    QOCOSettings settings{};
    set_default_settings(&settings);
    const bool strict=std::getenv("QOCO_PURE_SOC_STRICT")!=nullptr;
    settings.abstol=settings.reltol=settings.abstol_inacc=settings.reltol_inacc=strict?1e-13:1e-10;
    auto* solver=static_cast<QOCOSolver*>(std::malloc(sizeof(QOCOSolver)));
    REQUIRE(qoco_setup(solver,3,3,0,&P,c,nullptr,nullptr,&G,h,0,1,&size,&settings)==0);
    for(int repeat=0;repeat<12;++repeat) {
        REQUIRE(qoco_gpu_begin_reduction_scope()==0);
        const double p=1+.03*repeat;
        for(double& value:px) value=p;
        c[1]=-1-.1*repeat; c[2]=2-.03*repeat;
        qoco_update_matrix_data(solver,px,nullptr,nullptr);
        qoco_update_vector_data(solver,c,nullptr,nullptr);
        REQUIRE(qoco_gpu_primal_start(solver,repeat%2)==0);
        const int status=qoco_solve(solver);
        const double expected=-(c[1]*c[1]+c[2]*c[2])/(4*p);
        const double t=std::hypot(c[1],c[2])/(2*p);
        std::printf("SOC_DIAGNOSTIC repeat=%d objective_error=%.17g x_errors=[%.17g,%.17g,%.17g] pres=%.17g dres=%.17g\n",
            repeat,solver->sol->obj-expected,solver->sol->x[0]-t,
            solver->sol->x[1]+c[1]/(2*p),solver->sol->x[2]+c[2]/(2*p),solver->sol->pres,solver->sol->dres);
        REQUIRE(status==QOCO_SOLVED || status==QOCO_SOLVED_INACCURATE);
        REQUIRE(std::abs(solver->sol->obj-expected)<1e-8);
        const double violation=std::fmax(0.0,std::hypot(solver->sol->x[1],solver->sol->x[2])-solver->sol->x[0]);
        REQUIRE(violation<1e-8);
        // Strong convexity and the known optimal cone multiplier imply
        // ||x-x*||² <= 2(objective_error + lambda*cone_violation)/p.
        // Use the independently checked 1e-8 objective/feasibility limits.
        const double bound=strict?1e-7:std::sqrt(2e-8*(1+.5*std::hypot(c[1],c[2]))/p);
        const double distance=std::hypot(solver->sol->x[0]-t,
            std::hypot(solver->sol->x[1]+c[1]/(2*p),solver->sol->x[2]+c[2]/(2*p)));
        REQUIRE(distance<bound);
        REQUIRE(solver->sol->pres<1e-8 && solver->sol->dres<1e-8);
        std::printf("PURE_SOC repeat=%d status=%d ipm=%d objective=%.17g expected=%.17g\n",
            repeat,status,solver->sol->iters,solver->sol->obj,expected);
        REQUIRE(qoco_gpu_primal_start(solver,2)==0);
        qoco_gpu_end_reduction_scope();
    }
    qoco_cleanup(solver);
}
