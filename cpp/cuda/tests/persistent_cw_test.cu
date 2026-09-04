#include "cuda_test_support.hpp"

#include <pdhcg.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

namespace test = spacepdhcg::cuda::test;

namespace {

std::vector<double> one_shot_reference(const test::ProblemStorage& problem) {
    matrix_desc_t q{};
    q.m = problem.variables;
    q.n = problem.variables;
    q.fmt = matrix_csc;
    q.data.csc = {
        static_cast<int>(problem.h_q.size()),
        problem.h_q_offsets.data(),
        problem.h_q_indices.data(),
        problem.h_q.data(),
    };
    matrix_desc_t a{};
    a.m = problem.scalar_rows;
    a.n = problem.variables;
    a.fmt = matrix_csc;
    a.data.csc = {
        static_cast<int>(problem.h_a.size()),
        problem.h_a_offsets.data(),
        problem.h_a_indices.data(),
        problem.h_a.data(),
    };
    const double objective_constant = 0.0;
    qp_problem_t* qp = create_qp_problem(
        problem.h_c.data(),
        &q,
        nullptr,
        nullptr,
        &a,
        problem.h_scalar_lower.data(),
        problem.h_scalar_upper.data(),
        problem.h_variable_lower.data(),
        problem.h_variable_upper.data(),
        &objective_constant,
        0,
        nullptr,
        nullptr,
        nullptr,
        0,
        nullptr
    );
    test::require(qp != nullptr, "one-shot QP creation failed");
    pdhg_parameters_t parameters{};
    set_default_parameters(&parameters);
    parameters.verbose = 0;
    parameters.presolve = false;
    parameters.termination_criteria.eps_optimal_relative = 1.0e-7;
    parameters.termination_criteria.eps_feasible_relative = 1.0e-7;
    parameters.termination_criteria.iteration_limit = 200'000;
    pdhcg_result_t* result = solve_qp_problem(qp, &parameters);
    test::require(result != nullptr, "one-shot QP solve failed");
    std::vector<double> primal(
        result->primal_solution,
        result->primal_solution + problem.variables
    );
    pdhcg_result_free(result);
    qp_problem_free(qp);
    return primal;
}

std::vector<double> analytic_reference(const test::ProblemStorage& problem) {
    const double q0 = problem.h_q[0];
    const double q1 = problem.h_q[1];
    const double c0 = problem.h_c[0];
    const double c1 = problem.h_c[1];
    const double denominator = 1.0 / q0 + 1.0 / q1;
    const double multiplier = (-1.0 - c0 / q0 - c1 / q1) / denominator;
    return {
        -(c0 + multiplier) / q0,
        -(c1 + multiplier) / q1,
    };
}

}  // namespace

int main() {
    auto problem = test::make_box_problem(false, true);
    auto* workspace = test::create_workspace(problem);

    spacepdhcg_cuda_pointer_snapshot initial_pointers{};
    test::status_require(
        spacepdhcg_cuda_workspace_pointer_snapshot(workspace, &initial_pointers),
        "initial pointer snapshot"
    );
    auto diagnostics = test::solve_and_wait(workspace, problem);
    test::require(
        diagnostics.termination == SPACEPDHCG_CUDA_TERMINATION_OPTIMAL,
        "cold persistent solve did not converge"
    );
    auto persistent = problem.primal.download(problem.stream);
    auto analytic = analytic_reference(problem);
    auto one_shot = one_shot_reference(problem);
    for (int index = 0; index < problem.variables; ++index) {
        test::require_close(persistent[index], analytic[index], 2.0e-4, "cold CPU equality");
        test::require_close(persistent[index], one_shot[index], 5.0e-4, "cold one-shot equality");
    }

    const auto initial_topology_allocations = diagnostics.topology_allocation_count;
    const auto initial_topology_copies = diagnostics.topology_index_copy_count;
    const auto initial_allocation_count = diagnostics.allocation_count;
    double worst_cpu_error = 0.0;
    double worst_one_shot_error = 0.0;

    for (int update = 0; update < 10; ++update) {
        problem.h_c[0] = -1.0 - 0.02 * static_cast<double>(update + 1);
        problem.h_c[1] = -1.0 + 0.01 * static_cast<double>(update + 1);
        problem.upload_numeric();
        auto numeric = problem.numeric_views();
        test::status_require(
            spacepdhcg_cuda_workspace_update_async(
                workspace,
                problem.fingerprint,
                &numeric,
                problem.exchange.consumer_stream
            ),
            "values update"
        );

        const auto mode = static_cast<spacepdhcg_cuda_warm_start_mode>(
            update % 4
        );
        const auto* iterates =
            mode == SPACEPDHCG_CUDA_WARM_START_NONE
                || mode == SPACEPDHCG_CUDA_WARM_START_FULL_RETAINED
            ? nullptr
            : &problem.exchange.iterates;
        test::status_require(
            spacepdhcg_cuda_workspace_warm_start_async(
                workspace,
                mode,
                iterates,
                problem.exchange.consumer_stream
            ),
            "warm start"
        );
        diagnostics = test::solve_and_wait(workspace, problem);
        test::require(
            diagnostics.termination == SPACEPDHCG_CUDA_TERMINATION_OPTIMAL,
            "updated persistent solve did not converge"
        );
        persistent = problem.primal.download(problem.stream);
        analytic = analytic_reference(problem);
        one_shot = one_shot_reference(problem);
        for (int index = 0; index < problem.variables; ++index) {
            worst_cpu_error =
                std::max(worst_cpu_error, std::abs(persistent[index] - analytic[index]));
            worst_one_shot_error =
                std::max(worst_one_shot_error, std::abs(persistent[index] - one_shot[index]));
        }
        test::require(
            diagnostics.topology_allocation_count == initial_topology_allocations,
            "topology allocation count changed after create"
        );
        test::require(
            diagnostics.topology_index_copy_count == initial_topology_copies,
            "topology index copy count changed after create"
        );
        test::require(
            diagnostics.allocation_count == initial_allocation_count,
            "workspace allocated after create"
        );
        test::require(
            diagnostics.allocation_delta_last_update == 0U
                && diagnostics.topology_allocation_delta_last_update == 0U
                && diagnostics.topology_index_copy_delta_last_update == 0U,
            "values update changed allocation/topology ledgers"
        );
        test::require(
            diagnostics.hidden_cpu_fallback == 0
                && diagnostics.used_declared_stream == 1,
            "fallback or wrong stream reported"
        );
    }
    test::require(worst_cpu_error < 3.0e-4, "updated CPU comparison failed");
    test::require(worst_one_shot_error < 8.0e-4, "updated one-shot comparison failed");

    spacepdhcg_cuda_pointer_snapshot final_pointers{};
    test::status_require(
        spacepdhcg_cuda_workspace_pointer_snapshot(workspace, &final_pointers),
        "final pointer snapshot"
    );
    test::require(
        std::memcmp(&initial_pointers, &final_pointers, sizeof(initial_pointers)) == 0,
        "persistent device pointers changed"
    );

    std::size_t checkpoint_bytes{0U};
    test::status_require(
        spacepdhcg_cuda_workspace_checkpoint_bytes(workspace, &checkpoint_bytes),
        "checkpoint size"
    );
    test::CudaBuffer<double> checkpoint(checkpoint_bytes / sizeof(double), false);
    auto checkpoint_view = test::view(
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
        "checkpoint"
    );
    test::status_require(
        spacepdhcg_cuda_workspace_reset_async(
            workspace,
            SPACEPDHCG_CUDA_RESET_FULL,
            problem.exchange.consumer_stream
        ),
        "full reset"
    );
    test::status_require(
        spacepdhcg_cuda_workspace_restore_async(
            workspace,
            problem.fingerprint,
            checkpoint_view,
            problem.exchange.consumer_stream
        ),
        "restore"
    );
    diagnostics = test::solve_and_wait(workspace, problem);
    persistent = problem.primal.download(problem.stream);
    analytic = analytic_reference(problem);
    test::require_close(persistent[0], analytic[0], 3.0e-4, "restore x0");
    test::require_close(persistent[1], analytic[1], 3.0e-4, "restore x1");

    std::printf(
        "{\"case\":\"persistent_cw\",\"updates\":%llu,"
        "\"topology_allocations\":%llu,\"topology_index_copies\":%llu,"
        "\"allocation_count\":%llu,\"worst_cpu_error\":%.9g,"
        "\"worst_oneshot_error\":%.9g,\"residual\":%.9g,"
        "\"solve_seconds\":%.9g}\n",
        static_cast<unsigned long long>(diagnostics.update_epoch),
        static_cast<unsigned long long>(diagnostics.topology_allocation_count),
        static_cast<unsigned long long>(diagnostics.topology_index_copy_count),
        static_cast<unsigned long long>(diagnostics.allocation_count),
        worst_cpu_error,
        worst_one_shot_error,
        diagnostics.natural_residual_inf,
        diagnostics.solve_seconds
    );
    test::destroy_workspace(workspace);
    return 0;
}
