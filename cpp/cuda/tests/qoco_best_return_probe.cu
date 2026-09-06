// Test-only audit: verify that an inaccurate exit returns the saved best point
// in physical units. The downloads below must never be used for timing.
#include "qoco.h"
#include "../algebra/cuda/cuda_types.h"
#include <cuda_runtime.h>
#include <dlfcn.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {
void require(bool ok, const char* message) {
    if (!ok) {
        std::fprintf(stderr, "best-return audit failed: %s\n", message);
        std::abort();
    }
}
std::vector<double> download(const QOCOVectorf* value) {
    std::vector<double> result(value->len);
    require(result.empty() || cudaMemcpy(result.data(), value->d_data,
                result.size() * sizeof(double), cudaMemcpyDeviceToHost) == cudaSuccess,
            "device download");
    return result;
}
void compare(const QOCOVectorf* live, const QOCOVectorf* saved,
             const QOCOVectorf* scaling, double factor) {
    require(live->len == saved->len && live->len == scaling->len, "vector dimensions");
    const auto actual = download(live), best = download(saved), scale = download(scaling);
    for (size_t i = 0; i < actual.size(); ++i) {
        const double expected = (best[i] * scale[i]) * factor;
        require(std::isfinite(actual[i]) && std::isfinite(expected) &&
                    std::abs(actual[i] - expected) <= 1e-14 * std::max(1.0, std::abs(expected)),
                "live vector differs from unscaled saved best iterate");
    }
}
}

extern "C" QOCOInt qoco_solve(QOCOSolver* solver) {
    const auto actual = reinterpret_cast<QOCOInt (*)(QOCOSolver*)>(dlsym(RTLD_NEXT, "qoco_solve"));
    const auto capability = reinterpret_cast<int (*)()>(dlsym(RTLD_NEXT, "qoco_restores_inaccurate_best"));
    require(actual && capability && capability() == 1, "best-return capability");
    const int status = actual(solver);
    require(status == solver->sol->status, "status consistency");
    if (status == QOCO_SOLVED_INACCURATE) {
        const auto* w = solver->work;
        const auto* s = solver->sol;
        require(w->best_valid && std::isfinite(w->best_metric) && w->best_metric <= 1.0,
                "qualified best iterate");
        require(s->pres == w->best_pres && s->dres == w->best_dres &&
                    s->gap == w->best_gap && s->obj == w->best_obj,
                "reported metrics differ from saved best iterate");
        compare(w->x, w->best_x, w->scaling->Druiz, 1.0);
        compare(w->s, w->best_s, w->scaling->Finvruiz, 1.0);
        compare(w->y, w->best_y, w->scaling->Eruiz, w->scaling->kinv);
        compare(w->z, w->best_z, w->scaling->Fruiz, w->scaling->kinv);
        std::fprintf(stderr, "{\"case\":\"best_return_audit\",\"iterations\":%d,"
                     "\"best_iteration\":%d,\"best_metric\":%.17g,\"passed\":true}\n",
                     s->iters, w->best_iter, w->best_metric);
    }
    return status;
}
