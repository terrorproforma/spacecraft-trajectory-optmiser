#include "cuda_test_support.hpp"

#include <pdhcg.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <string>
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
    matrix_desc_t f{};
    f.m = problem.affine_rows;
    f.n = problem.variables;
    f.fmt = matrix_csc;
    f.data.csc = {
        static_cast<int>(problem.h_f.size()),
        problem.h_f_offsets.data(),
        problem.h_f_indices.data(),
        problem.h_f.data(),
    };
    cone_spec_t cone{};
    cone.type = CONE_STANDARD_SOC;
    cone.start_idx = 0;
    cone.v_dim = 1;
    cone.power_alpha = 0.0;
    cone.is_fixed = nullptr;
    const double objective_constant = 0.0;
    qp_problem_t* qp = create_qp_problem(
        problem.h_c.data(),
        &q,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        problem.h_variable_lower.data(),
        problem.h_variable_upper.data(),
        &objective_constant,
        0,
        nullptr,
        &f,
        problem.h_affine_offset.data(),
        1,
        &cone
    );
    test::require(qp != nullptr, "one-shot SOCP creation failed");
    pdhg_parameters_t parameters{};
    set_default_parameters(&parameters);
    parameters.verbose = 0;
    parameters.presolve = false;
    parameters.termination_criteria.eps_optimal_relative = 1.0e-7;
    parameters.termination_criteria.eps_feasible_relative = 1.0e-7;
    parameters.termination_criteria.iteration_limit = 250'000;
    pdhcg_result_t* result = solve_qp_problem(qp, &parameters);
    test::require(result != nullptr, "one-shot SOCP solve failed");
    std::vector<double> primal(
        result->primal_solution,
        result->primal_solution + problem.variables
    );
    pdhcg_result_free(result);
    qp_problem_free(qp);
    return primal;
}

double standard_soc_distance(double x0, double x1, double radius) {
    const double norm = std::hypot(x0, x1);
    if (norm <= radius) {
        return 0.0;
    }
    if (norm <= -radius) {
        return std::max({std::abs(x0), std::abs(x1), std::abs(radius)});
    }
    const double projected_radius = 0.5 * (norm + radius);
    const double scale = norm > 0.0 ? projected_radius / norm : 0.0;
    return std::max({
        std::abs(x0 - scale * x0),
        std::abs(x1 - scale * x1),
        std::abs(radius - projected_radius),
    });
}

}  // namespace

int main(const int argc, char** argv) {
    const bool compare_one_shot =
        argc == 2 && std::string(argv[1]) == "--with-one-shot";
    auto problem = test::make_soc_problem(true, true);
    auto* workspace = test::create_workspace(problem);
    auto diagnostics = test::solve_and_wait(
        workspace,
        problem,
        test::solve_options(5.0e-6, 150'000U)
    );
    test::require(
        diagnostics.termination == SPACEPDHCG_CUDA_TERMINATION_OPTIMAL,
        "persistent SOCP did not converge"
    );
    const auto persistent = problem.primal.download(problem.stream);
    const auto one_shot = compare_one_shot
        ? one_shot_reference(problem)
        : std::vector<double>(2U, std::numeric_limits<double>::quiet_NaN());
    test::require_close(persistent[0], 1.0, 8.0e-4, "SOCP CPU x0");
    test::require_close(persistent[1], 0.0, 8.0e-4, "SOCP CPU x1");
    if (compare_one_shot) {
        test::require_close(persistent[0], one_shot[0], 1.5e-3, "SOCP one-shot x0");
        test::require_close(persistent[1], one_shot[1], 1.5e-3, "SOCP one-shot x1");
    }

    const double cpu_cone_distance =
        standard_soc_distance(persistent[0], persistent[1], 1.0);
    test::require_close(
        diagnostics.affine_cone_distance_inf,
        cpu_cone_distance,
        2.0e-7,
        "independent affine cone residual"
    );
    test::require(
        diagnostics.natural_residual_inf < 2.0e-4
            && diagnostics.relative_primal_residual < 2.0e-5,
        "independent SOCP residual is too large"
    );
    test::require(
        diagnostics.hidden_cpu_fallback == 0
            && diagnostics.used_declared_stream == 1,
        "SOCP solve reported fallback or stream mismatch"
    );
    test::status_require(
        spacepdhcg_cuda_workspace_residuals_async(
            workspace,
            problem.exchange.consumer_stream
        ),
        "independent residual launch"
    );
    test::status_require(spacepdhcg_cuda_workspace_wait(workspace), "independent residual wait");
    test::status_require(
        spacepdhcg_cuda_workspace_diagnostics(workspace, &diagnostics),
        "independent residual diagnostics"
    );

    std::printf(
        "{\"case\":\"persistent_soc\",\"managed\":true,"
        "\"one_shot_comparison\":%s,"
        "\"x\":[%.12g,%.12g],\"oneshot\":[%.12g,%.12g],"
        "\"cone_distance\":%.9g,\"natural_residual\":%.9g,"
        "\"topology_allocations\":%llu,\"topology_index_copies\":%llu}\n",
        compare_one_shot ? "true" : "false",
        persistent[0],
        persistent[1],
        one_shot[0],
        one_shot[1],
        diagnostics.affine_cone_distance_inf,
        diagnostics.natural_residual_inf,
        static_cast<unsigned long long>(diagnostics.topology_allocation_count),
        static_cast<unsigned long long>(diagnostics.topology_index_copy_count)
    );
    test::destroy_workspace(workspace);
    return 0;
}
