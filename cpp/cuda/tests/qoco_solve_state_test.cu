// Reuse one GPU solver across different right-hand sides. A forced iteration
// limit must never recover the best iterate of the previous problem.
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
extern "C" {
#include "qoco_api.h"
int qoco_gpu_begin_reduction_scope();
void qoco_gpu_end_reduction_scope();
}
static void require(bool ok, const char* message) {
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", message); std::exit(1); }
}
static int solve(QOCOSolver* solver) {
    require(qoco_gpu_begin_reduction_scope() == 0, "reduction scope");
    const int status = qoco_solve(solver);
    qoco_gpu_end_reduction_scope();
    return status;
}
int main(int argc, char** argv) {
    double px[]{1}, ax[]{1}, gx[]{-1}, c[]{0}, b[]{1}, h[]{0};
    int offsets[]{0, 1}, rows[]{0};
    QOCOCscMatrix p{}, a{}, g{};
    qoco_set_csc(&p, 1, 1, 1, px, offsets, rows);
    qoco_set_csc(&a, 1, 1, 1, ax, offsets, rows);
    qoco_set_csc(&g, 1, 1, 1, gx, offsets, rows);
    QOCOSettings settings{}; set_default_settings(&settings);
    settings.verbose = 0; settings.ruiz_iters = argc > 1 ? std::atoi(argv[1]) : 0;
    // qoco_cleanup owns and frees the solver object as well as its buffers.
    auto* allocation = static_cast<QOCOSolver*>(std::calloc(1, sizeof(QOCOSolver)));
    require(allocation != nullptr, "solver allocation");
    QOCOSolver& solver = *allocation;
    require(qoco_setup(&solver, 1, 1, 1, &p, c, &a, b, &g, h, 1, 0, nullptr,
                       &settings) == 0, "setup");
    const int initial = solve(&solver);
    require(initial == QOCO_SOLVED || initial == QOCO_SOLVED_INACCURATE, "initial solve");
    require(std::abs(solver.sol->x[0] - 1) < 1e-5, "initial equality");
    // Model a previous exact optimum: an unbeatable old progress metric.
    // The old buffers are the real previous solution, not a new allocation.
    solver.work->best_metric = 0;
    solver.work->best_valid = 1;
    solver.work->best_iter = 777;
    solver.sol->iters = 777;
    solver.sol->ir_iters = 777;
    settings.max_iters = 1;
    settings.abstol = settings.reltol = 1e-14;
    settings.abstol_inacc = settings.reltol_inacc = 1e-14;
    require(qoco_update_settings(&solver, &settings) == 0, "tight settings");
    b[0] = 4;
    qoco_update_vector_data(&solver, nullptr, b, nullptr);
    qoco_set_x0(&solver, nullptr);
    const int limited = solve(&solver);
    std::printf("{\"case\":\"changed_rhs_recovery\",\"expected_x\":4,\"x\":%.17g,"
                "\"status\":%d,\"best_iteration\":%d,\"iterations\":%d,\"ir_iterations\":%d}\n",
                solver.sol->x[0], limited, solver.work->best_iter,
                solver.sol->iters, solver.sol->ir_iters);
    require(limited == QOCO_MAX_ITER || limited == QOCO_SOLVED_INACCURATE,
            "exercise iteration-limit recovery");
    require(solver.work->best_iter <= 1, "stale best iteration restored");
    require(solver.sol->iters <= 1 && solver.sol->ir_iters < 777, "stale solve counters");
    require(std::abs(solver.sol->x[0] - b[0]) < 1e-4, "old problem solution restored");
    // An explicit warm start remains enabled; resetting history must not
    // invalidate caller-selected initialization or the retained GPU workspace.
    const double warm[]{4}; qoco_set_x0(&solver, warm);
    auto* work = solver.work;
    settings.max_iters = 100; settings.abstol = settings.reltol = 1e-8;
    settings.abstol_inacc = settings.reltol_inacc = 1e-5;
    require(qoco_update_settings(&solver, &settings) == 0, "restore settings");
    b[0] = 6; qoco_update_vector_data(&solver, nullptr, b, nullptr);
    const int final = solve(&solver);
    require(final == QOCO_SOLVED || final == QOCO_SOLVED_INACCURATE, "warm solve");
    require(solver.work == work && solver.work->use_x0, "workspace/warm-start retained");
    require(std::abs(solver.sol->x[0] - b[0]) < 1e-5, "warm equality");
    qoco_cleanup(&solver);
    require(cudaDeviceSynchronize() == cudaSuccess, "CUDA completion");
    std::puts("QOCO solve-state isolation passed");
}
