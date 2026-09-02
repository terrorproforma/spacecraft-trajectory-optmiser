#include "cuda_test_support.hpp"

#include <algorithm>
#include <atomic>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <thread>
#include <vector>

namespace test = spacepdhcg::cuda::test;

namespace {

void initialise_inconsistent_problem(test::ProblemStorage& problem) {
    problem.fingerprint = 0x8e2e5df339787de1ULL;
    problem.variables = 1;
    problem.scalar_rows = 2;
    problem.affine_rows = 0;
    problem.h_q_offsets = {0, 1};
    problem.h_q_indices = {0};
    problem.h_a_offsets = {0, 2};
    problem.h_a_indices = {0, 1};
    problem.h_q = {1.0};
    problem.h_a = {1.0, 1.0};
    problem.h_c = {-0.25};
    problem.h_scalar_lower = {0.0, 1.0};
    problem.h_scalar_upper = {0.0, 1.0};
    problem.h_variable_lower = {
        -std::numeric_limits<double>::infinity(),
    };
    problem.h_variable_upper = {
        std::numeric_limits<double>::infinity(),
    };
    problem.materialise();
}

void initialise_scalar_property_problem(
    test::ProblemStorage& problem,
    double q,
    double c,
    double lower,
    double upper
) {
    problem.fingerprint =
        0x933d7b711fe57311ULL
        ^ std::bit_cast<std::uint64_t>(q)
        ^ std::bit_cast<std::uint64_t>(c);
    problem.variables = 1;
    problem.scalar_rows = 0;
    problem.affine_rows = 0;
    problem.h_q_offsets = {0, 1};
    problem.h_q_indices = {0};
    problem.h_a_offsets = {0, 0};
    problem.h_q = {q};
    problem.h_c = {c};
    problem.h_variable_lower = {lower};
    problem.h_variable_upper = {upper};
    problem.materialise();
}

void require_equal(
    const std::vector<double>& left,
    const std::vector<double>& right,
    const char* message
) {
    test::require(left.size() == right.size(), message);
    test::require(
        std::memcmp(
            left.data(),
            right.data(),
            left.size() * sizeof(double)
        ) == 0,
        message
    );
}

void check_rejected_recovery_rollback() {
    test::ProblemStorage baseline_problem(false, true);
    initialise_inconsistent_problem(baseline_problem);
    auto* baseline = test::create_workspace(baseline_problem);
    const auto baseline_diagnostics = test::solve_and_wait(
        baseline,
        baseline_problem,
        test::solve_options(1.0e-12, 300'000U)
    );
    const auto baseline_primal =
        baseline_problem.primal.download(baseline_problem.stream);
    const auto baseline_dual =
        baseline_problem.dual.download(baseline_problem.stream);

    test::ProblemStorage recovery_problem(false, true);
    initialise_inconsistent_problem(recovery_problem);
    auto* recovery = test::create_workspace(recovery_problem);
    const auto recovery_diagnostics = test::solve_and_wait(
        recovery,
        recovery_problem,
        test::solve_options(1.0e-12, 350'000U)
    );
    const auto recovered_primal =
        recovery_problem.primal.download(recovery_problem.stream);
    const auto recovered_dual =
        recovery_problem.dual.download(recovery_problem.stream);

    test::require(
        baseline_diagnostics.termination
            == SPACEPDHCG_CUDA_TERMINATION_ITERATION_LIMIT,
        "inconsistent baseline unexpectedly qualified"
    );
    test::require(
        recovery_diagnostics.recovery_attempt_count == 1U
            && recovery_diagnostics.recovery_rejected_count == 1U
            && recovery_diagnostics.recovery_count == 0U,
        "inconsistent active set did not reject recovery"
    );
    test::require(
        recovery_diagnostics.recovery_trigger_reason
            == SPACEPDHCG_CUDA_RECOVERY_TIGHT_ITERATION_LIMIT,
        "recovery trigger reason was not explicit"
    );
    test::require(
        recovery_diagnostics.recovery_outcome_reason
            != SPACEPDHCG_CUDA_RECOVERY_QUALIFIED,
        "inconsistent active set was incorrectly qualified"
    );
    test::require(
        recovery_diagnostics.recovery_seconds > 0.0
            && std::isfinite(recovery_diagnostics.recovery_initial_residual)
            && std::isfinite(recovery_diagnostics.recovery_final_residual),
        "recovery cost or residual diagnostics are missing"
    );
    require_equal(
        baseline_primal,
        recovered_primal,
        "rejected recovery did not restore the PDHG primal"
    );
    require_equal(
        baseline_dual,
        recovered_dual,
        "rejected recovery did not restore the PDHG dual"
    );
    test::destroy_workspace(recovery);
    test::destroy_workspace(baseline);
}

void check_nonfinite_input() {
    auto problem = test::make_box_problem(false, true);
    problem.h_c[0] = std::numeric_limits<double>::quiet_NaN();
    problem.upload_numeric();
    auto* workspace = test::create_workspace(problem);
    auto diagnostics = test::solve_and_wait(
        workspace,
        problem,
        test::solve_options(1.0e-8, 350'000U)
    );
    test::require(
        diagnostics.termination
            == SPACEPDHCG_CUDA_TERMINATION_NUMERICAL_FAILURE,
        "non-finite CQP input did not fail numerically"
    );
    test::require(
        diagnostics.recovery_attempt_count == 0U,
        "recovery ran on non-finite solver state"
    );
    test::destroy_workspace(workspace);
}

void check_invalid_cone_dual() {
    auto problem = test::make_soc_problem(false, true);
    auto* workspace = test::create_workspace(problem);
    static_cast<void>(test::solve_and_wait(workspace, problem));
    problem.primal.upload({1.0, 0.0}, problem.stream);
    problem.dual.upload({1.0, 0.0, 1.0}, problem.stream);
    test::status_require(
        spacepdhcg_cuda_workspace_warm_start_async(
            workspace,
            SPACEPDHCG_CUDA_WARM_START_PRIMAL_DUAL,
            &problem.exchange.iterates,
            problem.exchange.consumer_stream
        ),
        "invalid cone dual warm start"
    );
    test::status_require(
        spacepdhcg_cuda_workspace_wait(workspace),
        "invalid cone dual warm wait"
    );
    test::status_require(
        spacepdhcg_cuda_workspace_residuals_async(
            workspace,
            problem.exchange.consumer_stream
        ),
        "invalid cone dual residual"
    );
    test::status_require(
        spacepdhcg_cuda_workspace_wait(workspace),
        "invalid cone dual residual wait"
    );
    spacepdhcg_cuda_diagnostics diagnostics{};
    test::status_require(
        spacepdhcg_cuda_workspace_diagnostics(workspace, &diagnostics),
        "invalid cone dual diagnostics"
    );
    std::printf(
        "{\"case\":\"invalid_cone_dual\",\"affine\":%.9g,"
        "\"stationarity\":%.9g,\"natural\":%.9g}\n",
        diagnostics.affine_cone_distance_inf,
        diagnostics.stationarity_inf,
        diagnostics.natural_residual_inf
    );
    test::require(
        diagnostics.affine_cone_distance_inf < 1.0e-12
            && diagnostics.stationarity_inf < 1.0e-12
            && diagnostics.natural_residual_inf > 0.5,
        "canonical residual accepted an invalid cone dual"
    );
    test::destroy_workspace(workspace);
}

void check_properties() {
    std::mt19937_64 generator(0x4d5f1709ULL);
    std::uniform_real_distribution<double> q_distribution(0.25, 4.0);
    std::uniform_real_distribution<double> c_distribution(-3.0, 3.0);
    for (int sample = 0; sample < 16; ++sample) {
        const double q = q_distribution(generator);
        const double c = c_distribution(generator);
        const double lower = -1.0;
        const double upper = 1.0;
        test::ProblemStorage problem(false, (sample % 2) != 0);
        initialise_scalar_property_problem(problem, q, c, lower, upper);
        auto* workspace = test::create_workspace(problem);
        const auto diagnostics = test::solve_and_wait(
            workspace,
            problem,
            test::solve_options(1.0e-7, 100'000U)
        );
        const auto primal = problem.primal.download(problem.stream);
        const double expected = std::clamp(-c / q, lower, upper);
        test::require(
            diagnostics.termination == SPACEPDHCG_CUDA_TERMINATION_OPTIMAL,
            "random scalar/box property did not converge"
        );
        test::require_close(
            primal[0],
            expected,
            2.0e-5,
            "random scalar/box optimum"
        );
        test::destroy_workspace(workspace);
    }

    std::uniform_real_distribution<double> target_distribution(-2.0, 2.0);
    for (int sample = 0; sample < 8; ++sample) {
        auto problem = test::make_soc_problem(false, (sample % 2) != 0);
        const double target_x = target_distribution(generator);
        const double target_y = target_distribution(generator);
        problem.h_c = {-target_x, -target_y};
        problem.upload_numeric();
        auto* workspace = test::create_workspace(problem);
        const auto diagnostics = test::solve_and_wait(
            workspace,
            problem,
            test::solve_options(2.0e-6, 200'000U)
        );
        const auto primal = problem.primal.download(problem.stream);
        const double norm = std::hypot(target_x, target_y);
        const double scale = norm > 1.0 ? 1.0 / norm : 1.0;
        test::require(
            diagnostics.termination == SPACEPDHCG_CUDA_TERMINATION_OPTIMAL,
            "random SOC property did not converge"
        );
        test::require_close(
            primal[0],
            scale * target_x,
            2.0e-3,
            "random SOC x optimum"
        );
        test::require_close(
            primal[1],
            scale * target_y,
            2.0e-3,
            "random SOC y optimum"
        );
        test::destroy_workspace(workspace);
    }
}

void check_warm_modes_and_checkpoint() {
    auto problem = test::make_box_problem(false, true);
    auto* workspace = test::create_workspace(problem);
    for (const auto mode : {
             SPACEPDHCG_CUDA_WARM_START_NONE,
             SPACEPDHCG_CUDA_WARM_START_PRIMAL,
             SPACEPDHCG_CUDA_WARM_START_PRIMAL_DUAL,
         }) {
        problem.primal.upload({0.25, 0.75}, problem.stream);
        problem.dual.upload({0.0}, problem.stream);
        const auto* iterates =
            mode == SPACEPDHCG_CUDA_WARM_START_NONE
            ? nullptr
            : &problem.exchange.iterates;
        test::status_require(
            spacepdhcg_cuda_workspace_warm_start_async(
                workspace,
                mode,
                iterates,
                problem.exchange.consumer_stream
            ),
            "warm mode"
        );
        test::status_require(
            spacepdhcg_cuda_workspace_wait(workspace),
            "warm mode wait"
        );
        const auto diagnostics = test::solve_and_wait(workspace, problem);
        test::require(
            diagnostics.termination == SPACEPDHCG_CUDA_TERMINATION_OPTIMAL,
            "warm-mode solve did not converge"
        );
    }
    test::status_require(
        spacepdhcg_cuda_workspace_warm_start_async(
            workspace,
            SPACEPDHCG_CUDA_WARM_START_FULL_RETAINED,
            nullptr,
            problem.exchange.consumer_stream
        ),
        "full-retained warm mode"
    );
    test::status_require(
        spacepdhcg_cuda_workspace_wait(workspace),
        "full-retained warm wait"
    );

    std::size_t bytes = 0U;
    test::status_require(
        spacepdhcg_cuda_workspace_checkpoint_bytes(workspace, &bytes),
        "recovery checkpoint bytes"
    );
    test::CudaBuffer<double> checkpoint(bytes / sizeof(double), false);
    const auto checkpoint_view = test::view(
        checkpoint.get(),
        checkpoint.size(),
        false,
        SPACEPDHCG_SCALAR_FLOAT64,
        SPACEPDHCG_ACCESS_READ_WRITE
    );
    test::status_require(
        spacepdhcg_cuda_workspace_checkpoint_async(
            workspace,
            checkpoint_view,
            problem.exchange.consumer_stream
        ),
        "recovery checkpoint"
    );
    test::status_require(
        spacepdhcg_cuda_workspace_wait(workspace),
        "recovery checkpoint wait"
    );
    test::require(
        spacepdhcg_cuda_workspace_restore_async(
            workspace,
            problem.fingerprint ^ 1U,
            checkpoint_view,
            problem.exchange.consumer_stream
        ) == SPACEPDHCG_CUDA_TOPOLOGY_MISMATCH,
        "checkpoint topology invalidation was accepted"
    );
    test::status_require(
        spacepdhcg_cuda_workspace_restore_async(
            workspace,
            problem.fingerprint,
            checkpoint_view,
            problem.exchange.consumer_stream
        ),
        "recovery checkpoint restore"
    );
    const auto restored = test::solve_and_wait(workspace, problem);
    test::require(
        restored.termination == SPACEPDHCG_CUDA_TERMINATION_OPTIMAL,
        "restored checkpoint solve failed"
    );
    test::destroy_workspace(workspace);
}

void check_cancellation_and_destruction(const bool sanitizer) {
    test::ProblemStorage problem(false, true);
    initialise_inconsistent_problem(problem);
    auto* workspace = test::create_workspace(problem);
    const auto options = test::solve_options(1.0e-20, 350'000U);
    test::status_require(
        spacepdhcg_cuda_workspace_solve_async(
            workspace,
            &options,
            problem.exchange.consumer_stream
        ),
        "recovery cancellation launch"
    );
    test::status_require(
        spacepdhcg_cuda_workspace_cancel(workspace),
        "recovery cancellation"
    );
    test::status_require(
        spacepdhcg_cuda_workspace_wait(workspace),
        "recovery cancellation wait"
    );
    spacepdhcg_cuda_diagnostics diagnostics{};
    test::status_require(
        spacepdhcg_cuda_workspace_diagnostics(workspace, &diagnostics),
        "recovery cancellation diagnostics"
    );
    test::require(
        diagnostics.termination == SPACEPDHCG_CUDA_TERMINATION_CANCELLED
            || (sanitizer
                && diagnostics.termination
                    == SPACEPDHCG_CUDA_TERMINATION_ITERATION_LIMIT),
        "recovery cancellation neither cancelled nor completed before the store"
    );
    test::destroy_workspace(workspace);

    test::ProblemStorage destruction_problem(false, true);
    initialise_inconsistent_problem(destruction_problem);
    auto* destruction = test::create_workspace(destruction_problem);
    test::status_require(
        spacepdhcg_cuda_workspace_solve_async(
            destruction,
            &options,
            destruction_problem.exchange.consumer_stream
        ),
        "recovery destruction launch"
    );
    test::destroy_workspace(destruction);
}

void check_cross_thread_cancellation_during_wait() {
    // The G4 session cancels a running attempt from a watchdog thread while the owner thread
    // blocks in spacepdhcg_cuda_workspace_wait. The inconsistent problem never converges and
    // the budget is far beyond any test horizon, so only cancellation can end the solve.
    test::ProblemStorage problem(false, true);
    initialise_inconsistent_problem(problem);
    auto* workspace = test::create_workspace(problem);
    const auto options = test::solve_options(1.0e-20, 4'000'000'000ULL);
    test::status_require(
        spacepdhcg_cuda_workspace_solve_async(
            workspace,
            &options,
            problem.exchange.consumer_stream
        ),
        "cross-thread cancellation launch"
    );
    std::atomic<int> cancel_status{-1};
    const auto started = std::chrono::steady_clock::now();
    std::thread watchdog([&]() {
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        cancel_status.store(
            static_cast<int>(spacepdhcg_cuda_workspace_cancel(workspace))
        );
    });
    test::status_require(
        spacepdhcg_cuda_workspace_wait(workspace),
        "cross-thread cancellation wait"
    );
    watchdog.join();
    const double elapsed = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - started
    ).count();
    test::require(
        cancel_status.load() == static_cast<int>(SPACEPDHCG_CUDA_SUCCESS),
        "cross-thread cancellation did not reach the running solve"
    );
    spacepdhcg_cuda_diagnostics diagnostics{};
    test::status_require(
        spacepdhcg_cuda_workspace_diagnostics(workspace, &diagnostics),
        "cross-thread cancellation diagnostics"
    );
    test::require(
        diagnostics.termination == SPACEPDHCG_CUDA_TERMINATION_CANCELLED,
        "cross-thread cancellation did not terminate the solve as cancelled"
    );
    test::require(
        elapsed < 60.0,
        "owner wait did not return promptly after cross-thread cancellation"
    );
    test::destroy_workspace(workspace);
}

}  // namespace

int main(const int argc, char** argv) {
    const bool sanitizer =
        argc == 2 && std::strcmp(argv[1], "--sanitizer") == 0;
    check_rejected_recovery_rollback();
    check_nonfinite_input();
    check_invalid_cone_dual();
    check_warm_modes_and_checkpoint();
    check_cancellation_and_destruction(sanitizer);
    if (!sanitizer) {
        check_cross_thread_cancellation_during_wait();
        check_properties();
    }
    std::printf(
        "{\"case\":\"recovery\",\"rank_deficient\":true,"
        "\"inconsistent_active_set\":true,\"invalid_cone_dual\":true,"
        "\"nonfinite\":true,\"exhaustion\":true,\"rollback\":true,"
        "\"cancellation\":true,\"destruction\":true,"
        "\"topology_invalidation\":true,\"checkpoint_restore\":true,"
        "\"default_and_nondefault_stream\":true,\"warm_modes\":4,"
        "\"deterministic\":true,\"random_properties\":%d}\n",
        sanitizer ? 0 : 24
    );
    return 0;
}
