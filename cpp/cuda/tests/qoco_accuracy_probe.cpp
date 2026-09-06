// Test-only policy probe. Production QOCO is an unchanged linked dependency.
#include <dlfcn.h>
#include <cstdlib>
#include <cstring>
extern "C" {
#include "qoco.h"
}
extern "C" QOCOInt qoco_solve(QOCOSolver* solver) {
    const auto actual=reinterpret_cast<QOCOInt(*)(QOCOSolver*)>(dlsym(RTLD_NEXT,"qoco_solve"));
    const char* mode=std::getenv("SPACEPDHCG_TEST_QOCO_ACCURACY_MODE");
    if(!actual || !mode) std::abort();
    auto* settings=solver->settings;
    const double previous_tol=settings->ir_tol;
    const int previous_max=settings->max_ir_iters;
    if(std::strcmp(mode,"ir10")==0 || std::strcmp(mode,"ir12")==0) {
        settings->ir_tol=std::strcmp(mode,"ir10")==0 ? 1e-10 : 1e-12;
        settings->max_ir_iters=10;
    } else if(std::strcmp(mode,"best")!=0 && std::strcmp(mode,"control")!=0) std::abort();
    const int status=actual(solver);
    settings->ir_tol=previous_tol;settings->max_ir_iters=previous_max;
    if(std::strcmp(mode,"best")==0 && status==QOCO_SOLVED_INACCURATE) {
        if(restore_best_iterate(solver)) {unscale_variables(solver->work);copy_solution(solver);}
    }
    return solver->sol->status;
}
