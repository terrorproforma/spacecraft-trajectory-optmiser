// Retained-workspace integration probe: an independently solved two-variable QP.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include "qoco.h"
extern "C" int qoco_gpu_begin_reduction_scope();
extern "C" void qoco_gpu_end_reduction_scope();

#define REQUIRE(x) do { if (!(x)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #x); std::exit(1); } } while (0)

int main() {
    QOCOFloat px[] = {1.0, 2.0}, ax[] = {1.0, 1.0}, gx[] = {-1.0, -1.0};
    QOCOInt pp[] = {0, 1, 2}, pi[] = {0, 1}, ap[] = {0, 1, 2}, ai[] = {0, 0};
    QOCOFloat c[] = {-1.0, -1.0}, b[] = {1.5}, h[] = {0.0, 0.0};
    QOCOCscMatrix P{}, A{}, G{};
    qoco_set_csc(&P, 2, 2, 2, px, pp, pi);
    qoco_set_csc(&A, 1, 2, 2, ax, ap, ai);
    qoco_set_csc(&G, 2, 2, 2, gx, pp, pi);
    QOCOSettings settings;
    set_default_settings(&settings);
    settings.abstol = settings.reltol = 1e-9;
    settings.abstol_inacc = settings.reltol_inacc = 1e-9;
    // Deliberately noticeable regularization exercises true-KKT refinement.
    settings.kkt_static_reg_P = settings.kkt_static_reg_A = settings.kkt_static_reg_G = 1e-4;
    auto* solver = static_cast<QOCOSolver*>(std::malloc(sizeof(QOCOSolver)));
    REQUIRE(qoco_setup(solver, 2, 2, 1, &P, c, &A, b, &G, h, 2, 0, nullptr, &settings) == QOCO_NO_ERROR);
    const int limits[] = {5, 0, 1, 5, 0, 10, 1, 5};
    const double tolerances[] = {1e-12, 1e-12, 1e-12, 1e-2, 1e-2, 1e-14, 1e-10, 1e-12};
    for (int repeat = 0; repeat < 16; ++repeat) {
        // Match the native adapter's scoped queued-operator API contract.
        REQUIRE(qoco_gpu_begin_reduction_scope() == 0);
        const int index = repeat % 8;
        settings.max_ir_iters = limits[index];
        settings.ir_tol = tolerances[index];
        settings.max_iters = repeat == 14 ? 1 : 200;
        settings.verbose = std::getenv("QOCO_PROBE_VERBOSE") ? 1 : 0;
        REQUIRE(qoco_update_settings(solver, &settings) == QOCO_NO_ERROR);
        // Mutate values and RHS while preserving topology and the same workspace.
        px[0] = 1.0 + .05 * repeat;
        px[1] = 2.0 + .03 * repeat;
        b[0] = 1.5 + .02 * repeat;
        c[0] = -1.0 + .01 * repeat;
        qoco_update_matrix_data(solver, px, nullptr, nullptr);
        qoco_update_vector_data(solver, c, b, nullptr);
        const int status = qoco_solve(solver);
        const double expected1 = (px[0] * b[0] + c[0] - c[1]) / (px[0] + px[1]);
        const double expected0 = b[0] - expected1;
        const double expected = .5 * (px[0] * expected0 * expected0 + px[1] * expected1 * expected1)
                              + c[0] * expected0 + c[1] * expected1;
        REQUIRE(solver->sol->ir_iters >= 0);
        REQUIRE(solver->sol->ir_iters <= 2 * solver->sol->iters * settings.max_ir_iters);
        if (!settings.max_ir_iters) REQUIRE(solver->sol->ir_iters == 0);
        if (repeat != 14) {
            REQUIRE(status == QOCO_SOLVED || status == QOCO_SOLVED_INACCURATE);
            REQUIRE(std::abs(solver->sol->obj - expected) < 1e-7);
            REQUIRE(solver->sol->pres < 1e-8 && solver->sol->dres < 1e-8);
        }
        std::printf("IR_REUSE {\"repeat\":%d,\"limit\":%d,\"tolerance\":%.17g,\"status\":%d,"
                    "\"ipm\":%d,\"ir\":%d,\"step_ir\":%d,\"objective\":%.17g,\"expected\":%.17g}\n",
                    repeat, settings.max_ir_iters, settings.ir_tol, status, solver->sol->iters,
                    solver->sol->ir_iters, solver->work->ir_iters, solver->sol->obj, expected);
        qoco_gpu_end_reduction_scope();
    }
    qoco_cleanup(solver);
}
