// Test-only proxy, linked against the isolated QOCO library. All other symbols
// resolve through that dependency. No fault injection enters production code.
#include <dlfcn.h>
#include <cstdlib>
extern "C" {
#include "qoco.h"
}
static thread_local bool fail_next{};
extern "C" void qoco_test_fail_next_solve() { fail_next=true; }
extern "C" QOCOInt qoco_solve(QOCOSolver* solver) {
    if (fail_next) {
        fail_next=false; solver->sol->status=3; solver->sol->iters=0;
        return 3; // QOCO numerical failure, injected before any numerical work.
    }
    const auto actual=reinterpret_cast<QOCOInt(*)(QOCOSolver*)>(dlsym(RTLD_NEXT,"qoco_solve"));
    if(!actual) std::abort();
    return actual(solver);
}
