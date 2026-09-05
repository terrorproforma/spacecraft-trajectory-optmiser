// Opt-in device results and accepted-start ownership, independent of the adapter.
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
extern "C" {
#include "qoco_api.h"
#include "../algebra/cuda/cuda_types.h"
int qoco_gpu_begin_reduction_scope();
void qoco_gpu_end_reduction_scope();
int qoco_gpu_set_device_io(QOCOSolver*, int);
int qoco_gpu_primal_start(QOCOSolver*, int);
int qoco_gpu_download_solution(QOCOSolver*);
int qoco_gpu_get_solution(QOCOSolver*, int, int, int,
                         const double**, const double**, const double**);
}
static void require(bool ok, const char* message) {
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", message); std::exit(1); }
}
static double read(const double* device) {
    double value{};
    require(cudaMemcpy(&value, device, sizeof(value), cudaMemcpyDeviceToHost) == cudaSuccess,
            "device read");
    return value;
}
static void solve(QOCOSolver* solver) {
    require(qoco_gpu_begin_reduction_scope() == 0, "reduction scope");
    const int status = qoco_solve(solver);
    qoco_gpu_end_reduction_scope();
    require(status == QOCO_SOLVED || status == QOCO_SOLVED_INACCURATE, "solve");
}
static QOCOSolver* create(int ruiz) {
    // Non-unit coefficients exercise scaling of the saved unscaled primal.
    double px[]{3}, ax[]{2}, gx[]{-4}, c[]{0}, b[]{2}, h[]{0};
    int offsets[]{0, 1}, rows[]{0};
    QOCOCscMatrix p{}, a{}, g{};
    qoco_set_csc(&p, 1, 1, 1, px, offsets, rows);
    qoco_set_csc(&a, 1, 1, 1, ax, offsets, rows);
    qoco_set_csc(&g, 1, 1, 1, gx, offsets, rows);
    QOCOSettings settings{}; set_default_settings(&settings);
    settings.verbose = 0; settings.ruiz_iters = ruiz;
    auto* solver = static_cast<QOCOSolver*>(std::calloc(1, sizeof(QOCOSolver)));
    require(solver != nullptr, "allocation");
    require(qoco_setup(solver, 1, 1, 1, &p, c, &a, b, &g, h, 1, 0, nullptr,
                       &settings) == 0, "setup");
    return solver;
}
int main(int argc, char** argv) {
    const int ruiz = argc > 1 ? std::atoi(argv[1]) : 0;
    auto* solver = create(ruiz);
    auto* other = create(ruiz);
    require(qoco_gpu_primal_start(solver, 1) != 0, "reject missing accepted start");
    solve(solver);
    require(std::abs(solver->sol->x[0] - 1) < 1e-5, "legacy host result");
    require(qoco_gpu_set_device_io(solver, 1) == 0, "enable device IO");
    solver->sol->x[0] = solver->sol->s[0] = solver->sol->y[0] = solver->sol->z[0] = -777;
    double b[]{8}; qoco_update_vector_data(solver, nullptr, b, nullptr);
    solve(solver);
    require(solver->sol->x[0] == -777 && solver->sol->s[0] == -777
            && solver->sol->y[0] == -777 && solver->sol->z[0] == -777,
            "device solve must not modify host vectors");
    const double *x{}, *y{}, *z{};
    require(qoco_gpu_get_solution(solver, 1, 1, 1, &x, &y, &z) == 0, "device result");
    require(std::abs(read(x) - 4) < 1e-5, "device equality");
    require(qoco_gpu_primal_start(solver, 2) == 0, "accept device primal");
    const double accepted = read(solver->work->x0->d_data);
    require(std::abs(accepted - 4) < 1e-5, "accepted start is unscaled");
    require(qoco_gpu_primal_start(solver, 0) == 0, "cold solve retains saved start");
    b[0] = 12; qoco_update_vector_data(solver, nullptr, b, nullptr);
    solve(solver); // Deliberately do not accept this candidate.
    require(read(solver->work->x0->d_data) == accepted, "rejected solve preserves accepted start");
    require(qoco_gpu_download_solution(solver) == 0, "explicit host export");
    require(std::abs(solver->sol->x[0] - 6) < 1e-5, "exported equality");
    require(qoco_gpu_primal_start(solver, 1) == 0 && solver->work->use_x0,
            "enable retained accepted start");
    require(read(solver->work->x0->d_data) == accepted, "enable does not overwrite cache");
    b[0] = 16; qoco_update_vector_data(solver, nullptr, b, nullptr);
    solve(solver);
    require(qoco_gpu_get_solution(solver, 1, 1, 1, &x, &y, &z) == 0, "warm device result");
    require(std::abs(read(x) - 8) < 1e-5, "warm equality");
    solve(other);
    require(std::abs(other->sol->x[0] - 1) < 1e-5, "other solver retains legacy mode");
    require(qoco_gpu_primal_start(other, 1) != 0, "accepted caches isolated");
    double host_start[]{8}; qoco_set_x0(solver, host_start);
    require(qoco_gpu_primal_start(solver, 1) != 0, "host overwrite invalidates accepted cache");
    require(qoco_gpu_set_device_io(solver, 0) == 0, "restore host output");
    require(qoco_gpu_primal_start(solver, 0) == 0, "disable warm start");
    solve(solver);
    require(std::abs(solver->sol->x[0] - 8) < 1e-5, "restored host equality");
    qoco_cleanup(other); qoco_cleanup(solver);
    require(cudaDeviceSynchronize() == cudaSuccess, "CUDA completion");
    std::printf("QOCO device IO and accepted-primal ownership passed (Ruiz %d)\n", ruiz);
}
