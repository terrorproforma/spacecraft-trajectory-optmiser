// Host equilibration must publish all scaled numerical data to the GPU.
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
extern "C" {
#include "qoco_api.h"
#include "../algebra/cuda/cuda_types.h"
int qoco_gpu_begin_reduction_scope();
void qoco_gpu_end_reduction_scope();
}
static bool parity(QOCOVectorf* vector, const char* name) {
    double gpu{};
    if (cudaMemcpy(&gpu, vector->d_data, sizeof(gpu), cudaMemcpyDeviceToHost) != cudaSuccess)
        return false;
    std::printf("%s host=%.17g device=%.17g\n", name, vector->data[0], gpu);
    return gpu == vector->data[0];
}
int main(int argc, char** argv) {
    double px[]{3}, ax[]{2}, gx[]{-4}, c[]{1}, b[]{2}, h[]{4};
    int offsets[]{0, 1}, rows[]{0};
    QOCOCscMatrix p{}, a{}, g{};
    qoco_set_csc(&p, 1, 1, 1, px, offsets, rows);
    qoco_set_csc(&a, 1, 1, 1, ax, offsets, rows);
    qoco_set_csc(&g, 1, 1, 1, gx, offsets, rows);
    QOCOSettings settings{}; set_default_settings(&settings);
    settings.verbose = 0; settings.ruiz_iters = 4;
    auto* solver = static_cast<QOCOSolver*>(std::calloc(1, sizeof(QOCOSolver)));
    if (!solver || qoco_setup(solver, 1, 1, 1, &p, c, &a, b, &g, h, 1, 0, nullptr,
                              &settings) != 0) return 1;
    bool ok = true;
    for (int stage = 0; stage < 2; ++stage) {
        if (stage) {
            // Also cover changing from scaled data to zero Ruiz passes.
            settings.ruiz_iters = argc > 1 ? std::atoi(argv[1]) : 4;
            ok &= qoco_update_settings(solver, &settings) == 0;
            ax[0] = 4;
            qoco_update_matrix_data(solver, nullptr, ax, nullptr);
        }
        ok &= parity(solver->work->data->c, "c");
        ok &= parity(solver->work->data->b, "b");
        ok &= parity(solver->work->data->h, "h");
        if (qoco_gpu_begin_reduction_scope() != 0) { qoco_cleanup(solver); return 1; }
        const int status = qoco_solve(solver);
        qoco_gpu_end_reduction_scope();
        std::printf("stage=%d status=%d x=%.17g expected=%.17g\n",
                    stage, status, solver->sol->x[0], b[0] / ax[0]);
        ok &= status == QOCO_SOLVED || status == QOCO_SOLVED_INACCURATE;
        ok &= std::abs(solver->sol->x[0] - b[0] / ax[0]) < 1e-5;
    }
    qoco_cleanup(solver);
    ok &= cudaDeviceSynchronize() == cudaSuccess;
    std::puts(ok ? "Host Ruiz vector synchronization passed" : "FAIL: stale scaled device vectors");
    return ok ? 0 : 1;
}
