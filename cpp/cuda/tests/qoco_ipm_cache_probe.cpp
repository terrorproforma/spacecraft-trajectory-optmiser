// Independent QP optima across interleaved workspaces, scaling/settings changes,
// cache invalidation, nested caller scopes, and termination/recovery.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include "qoco.h"
extern "C" int qoco_gpu_begin_reduction_scope();
extern "C" void qoco_gpu_end_reduction_scope();
extern "C" int qoco_gpu_primal_start(QOCOSolver*, int);
#define REQUIRE(x) do { if (!(x)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #x); std::exit(1); } } while (0)

struct Fixture {
    QOCOFloat px[2]{1.0, 2.0}, ax[2]{1.0, 1.0}, gx[2]{-1.0, -1.0};
    QOCOInt pp[3]{0, 1, 2}, pi[2]{0, 1}, ap[3]{0, 1, 2}, ai[2]{0, 0};
    QOCOFloat c[2]{-1.0, -1.0}, b[1]{1.5}, h[2]{};
    QOCOCscMatrix P{}, A{}, G{};
    QOCOSettings settings{};
    QOCOSolver* solver{};
    Fixture() {
        qoco_set_csc(&P, 2, 2, 2, px, pp, pi);
        qoco_set_csc(&A, 1, 2, 2, ax, ap, ai);
        qoco_set_csc(&G, 2, 2, 2, gx, pp, pi);
        set_default_settings(&settings);
        settings.ruiz_iters = std::getenv("QOCO_IPM_PROBE_RUIZ") ? 3 : 0;
        settings.kkt_static_reg_P = settings.kkt_static_reg_A = settings.kkt_static_reg_G = 1e-4;
        solver = static_cast<QOCOSolver*>(std::malloc(sizeof(QOCOSolver)));
        REQUIRE(qoco_setup(solver, 2, 2, 1, &P, c, &A, b, &G, h, 2, 0, nullptr, &settings) == QOCO_NO_ERROR);
    }
    ~Fixture() { qoco_cleanup(solver); }
    void run(int id, int repeat) {
        REQUIRE(qoco_gpu_begin_reduction_scope() == 0);
        const double scale = (repeat % 3 == 0 ? 1.0 : repeat % 3 == 1 ? 10.0 : .1) * (id + 1);
        px[0] = scale * (1.0 + .05 * repeat);
        px[1] = scale * (2.0 + .03 * repeat);
        c[0] = scale * (-1.0 + .01 * repeat);
        c[1] = -scale;
        b[0] = 1.5 + .02 * repeat + .1 * id;
        settings.abstol = settings.reltol = repeat % 2 ? 1e-10 : 1e-9;
        settings.abstol_inacc = settings.reltol_inacc = settings.abstol;
        settings.ir_tol = repeat % 2 ? 1e-14 : 1e-12;
        settings.max_ir_iters = repeat % 4 == 0 ? 0 : 10;
        settings.max_iters = repeat == 14 ? 1 : 200;
        settings.kkt_static_reg_G = repeat < 8 ? 1e-4 : 2e-4;
        if (std::getenv("QOCO_IPM_PROBE_WARM")) {
            REQUIRE(qoco_gpu_primal_start(solver, repeat % 2) == 0);
            settings.kkt_dynamic_reg = repeat % 2 ? 1e-10 : 1e-9;
        }
        REQUIRE(qoco_update_settings(solver, &settings) == QOCO_NO_ERROR);
        qoco_update_matrix_data(solver, px, nullptr, nullptr);
        qoco_update_vector_data(solver, c, b, nullptr);
        if (std::getenv("QOCO_IPM_PROBE_TERMINAL_TOGGLE")) {
            if (repeat % 2) setenv("SPACEPDHCG_TEST_QOCO_IPM_TERMINAL_DISABLE", "1", 1);
            else unsetenv("SPACEPDHCG_TEST_QOCO_IPM_TERMINAL_DISABLE");
        }
        const int status = qoco_solve(solver);
        const double x1 = (px[0] * b[0] + c[0] - c[1]) / (px[0] + px[1]);
        const double x0 = b[0] - x1;
        const double expected = .5 * (px[0] * x0 * x0 + px[1] * x1 * x1) + c[0] * x0 + c[1] * x1;
        if (repeat != 14) {
            REQUIRE(status == QOCO_SOLVED || status == QOCO_SOLVED_INACCURATE);
            REQUIRE(std::abs(solver->sol->obj - expected) < 1e-7);
            REQUIRE(solver->sol->pres < 1e-8 && solver->sol->dres < 1e-8);
        }
        REQUIRE(solver->sol->ir_iters >= 0);
        REQUIRE(solver->sol->ir_iters <= 2 * solver->sol->iters * settings.max_ir_iters);
        if (std::getenv("QOCO_IPM_PROBE_VECTORS")) {
            const auto* sol = solver->sol;
            if (repeat != 14) {
                REQUIRE(std::abs(sol->x[0] - x0) < 1e-7 && std::abs(sol->x[1] - x1) < 1e-7);
                REQUIRE(std::abs(sol->s[0] - x0) < 1e-7 && std::abs(sol->s[1] - x1) < 1e-7);
                REQUIRE(std::abs(sol->y[0] + px[0] * x0 + c[0]) < 1e-7);
                REQUIRE(std::abs(sol->z[0]) < 1e-7 && std::abs(sol->z[1]) < 1e-7);
            }
            std::printf("IPM_VECTORS [%d,%d,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g]\n",
                id, repeat, sol->x[0], sol->x[1], sol->y[0], sol->s[0], sol->s[1], sol->z[0], sol->z[1]);
        }
        std::printf("IPM_REUSE {\"workspace\":%d,\"repeat\":%d,\"k\":%.17g,\"status\":%d,\"ipm\":%d,\"ir\":%d,\"step_ir\":%d,\"objective\":%.17g,\"expected\":%.17g,\"warm\":%d,\"dynamic_reg\":%.17g}\n",
            id, repeat, solver->work->scaling->k, status, solver->sol->iters,
            solver->sol->ir_iters, solver->work->ir_iters, solver->sol->obj, expected,
            int(solver->work->use_x0), solver->settings->kkt_dynamic_reg);
        if (std::getenv("QOCO_IPM_PROBE_WARM") && repeat != 14)
            REQUIRE(qoco_gpu_primal_start(solver, 2) == 0);
        qoco_gpu_end_reduction_scope();
    }
};

int main() {
    REQUIRE(qoco_gpu_begin_reduction_scope() == 0);
    {
        Fixture a, b;
        for (int repeat = 0; repeat < 16; ++repeat) {
            if (repeat % 2) { b.run(1, repeat); a.run(0, repeat); }
            else { a.run(0, repeat); b.run(1, repeat); }
        }
    }
    qoco_gpu_end_reduction_scope();
}
