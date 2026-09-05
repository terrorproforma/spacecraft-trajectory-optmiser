#include "cuda_test_support.hpp"

#include <algorithm>
#include <chrono>
#include <thread>

namespace test = spacepdhcg::cuda::test;

namespace {

void compare_standalone_residuals(spacepdhcg_cuda_workspace* workspace,
                                test::ProblemStorage& problem, int blocks) {
    spacepdhcg_cuda_diagnostics results[2]{};
    for (int variant = 0; variant < 2; ++variant) {
        test::status_require(spacepdhcg_cuda_workspace_set_execution_blocks(
            workspace, variant == 0 ? 0 : std::max(1, blocks)), "residual execution policy");
        test::status_require(spacepdhcg_cuda_workspace_residuals_async(
            workspace, problem.exchange.consumer_stream), "standalone residual recomputation");
        test::status_require(spacepdhcg_cuda_workspace_wait(workspace), "standalone residual wait");
        test::status_require(spacepdhcg_cuda_workspace_diagnostics(workspace, &results[variant]),
                             "standalone residual diagnostics");
        test::require(results[variant].residual_seconds > 0.0, "residual time is measured");
    }
    const auto& serial = results[0];
    const auto& parallel = results[1];
    spacepdhcg_cuda_pointer_snapshot pointers{};
    test::status_require(spacepdhcg_cuda_workspace_pointer_snapshot(workspace, &pointers),
                         "objective resident primal");
    std::vector<double> primal(problem.variables);
    test::cuda_require(cudaMemcpyAsync(primal.data(), reinterpret_cast<const void*>(pointers.primal),
        primal.size() * sizeof(double), cudaMemcpyDeviceToHost, problem.stream), "objective primal download");
    test::cuda_require(cudaStreamSynchronize(problem.stream), "objective primal wait");
    // Independent definition: c'x + 1/2 x'Qx. The dual must never contribute
    // to this value, including at equality-constrained optima with nonzero duals.
    double expected_objective = 0.0;
    for (int column = 0; column < problem.variables; ++column) {
        expected_objective += problem.h_c[column] * primal[column];
        for (int i = problem.h_q_offsets[column]; i < problem.h_q_offsets[column + 1]; ++i) {
            expected_objective += 0.5 * problem.h_q[i] * primal[column] * primal[problem.h_q_indices[i]];
        }
    }
    for (const auto& result : results) {
        test::require_close(result.objective, expected_objective,
            1e-10 * std::max(1.0, std::abs(expected_objective)), "objective excludes dual contributions");
    }
#define CHECK_RESIDUAL(field) \
    test::require_close(parallel.field, serial.field, \
        1e-10 * std::max(1.0, std::abs(serial.field)), "standalone residual parity: " #field)
    CHECK_RESIDUAL(objective);
    CHECK_RESIDUAL(scalar_primal_violation_inf); CHECK_RESIDUAL(box_violation_inf);
    CHECK_RESIDUAL(affine_cone_distance_inf); CHECK_RESIDUAL(stationarity_inf);
    CHECK_RESIDUAL(natural_residual_inf); CHECK_RESIDUAL(complementarity_inf);
    CHECK_RESIDUAL(relative_primal_residual); CHECK_RESIDUAL(relative_dual_residual);
    CHECK_RESIDUAL(scaling_min); CHECK_RESIDUAL(scaling_max);
#undef CHECK_RESIDUAL
    test::require(serial.iterations == parallel.iterations, "residual retains completed iterations");
    test::require(serial.termination == parallel.termination, "residual retains termination");
    test::require(serial.recovery_iterations == parallel.recovery_iterations,
                  "residual retains recovery iterations");
    test::require(serial.allocation_count == parallel.allocation_count,
                  "standalone residual uses persistent scratch");
    test::status_require(spacepdhcg_cuda_workspace_set_execution_blocks(workspace, blocks),
                         "restore execution policy");
}

// Repeated equality-constrained QPs have the independent analytic solution
// x=(2/3,1/3). Replication changes the grid size without changing each solution.
void replicated_box(test::ProblemStorage& problem, int count) {
    problem.variables = 2 * count;
    problem.scalar_rows = count;
    problem.h_q_offsets.push_back(0);
    problem.h_a_offsets.push_back(0);
    for (int i = 0; i < problem.variables; ++i) {
        problem.h_q_offsets.push_back(i + 1);
        problem.h_q_indices.push_back(i);
        problem.h_q.push_back(i % 2 == 0 ? 1.0 : 2.0);
        problem.h_a_offsets.push_back(i + 1);
        problem.h_a_indices.push_back(i / 2);
        problem.h_a.push_back(1.0);
        problem.h_c.push_back(-1.0);
        problem.h_variable_lower.push_back(0.0);
        problem.h_variable_upper.push_back(1.0);
    }
    problem.h_scalar_lower.assign(count, 1.0);
    problem.h_scalar_upper.assign(count, 1.0);
    problem.materialise();
}

void run_case(test::ProblemStorage& problem, int blocks, bool soc) {
    auto* workspace = test::create_workspace(problem);
    test::status_require(spacepdhcg_cuda_workspace_wait(workspace), "create wait");
    test::status_require(spacepdhcg_cuda_workspace_set_execution_blocks(workspace, blocks), "select grid");
    const auto result = test::solve_and_wait(workspace, problem);
    test::require(result.termination == SPACEPDHCG_CUDA_TERMINATION_OPTIMAL, "cooperative convergence");
    const auto primal = problem.primal.download(problem.stream);
    for (int i = 0; i < problem.variables; ++i) {
        const double expected = soc ? (i == 0 ? 1.0 : 0.0) : (i % 2 == 0 ? 2.0/3.0 : 1.0/3.0);
        test::require(std::isfinite(primal[i]), "finite cooperative solution");
        test::require_close(primal[i], expected, 2.0e-6, "independent analytic solution");
    }
    test::require(result.natural_residual_inf <= 2.0e-6, "unchanged canonical tolerance");
    compare_standalone_residuals(workspace, problem, blocks);
    std::printf("{\"case\":\"cooperative_%s\",\"variables\":%d,\"blocks\":%d,\"scaling_seconds\":%.9g,\"solve_seconds\":%.9g,\"iterations\":%llu,\"residual\":%.9g}\n",
                soc ? "soc" : "box", problem.variables, blocks, result.scaling_seconds,
                result.solve_seconds, static_cast<unsigned long long>(result.iterations), result.natural_residual_inf);
    // Reuse path must take a uniform branch even when its counters change.
    const auto repeated = test::solve_and_wait(workspace, problem);
    test::require(repeated.residual_seconds == 0.0, "new solve clears old residual timing");
    test::require(repeated.termination == SPACEPDHCG_CUDA_TERMINATION_OPTIMAL, "cooperative repeated solve");
    test::require(repeated.allocation_count == result.allocation_count, "persistent grid buffers");
    auto numeric = problem.numeric_views();
    test::status_require(spacepdhcg_cuda_workspace_update_async(
        workspace, problem.fingerprint, &numeric, problem.exchange.consumer_stream), "identical update");
    test::status_require(spacepdhcg_cuda_workspace_wait(workspace), "update wait");
    spacepdhcg_cuda_diagnostics updated{};
    // Change metrics are exported when the next report is evaluated.
    updated = test::solve_and_wait(workspace, problem);
    test::require(updated.coefficient_change_norm == 0.0, "unchanged infinite bounds have zero change");
    test::require(updated.coefficient_change_max == 0.0, "unchanged coefficients have zero change");
    problem.h_c[0] += 0.25;
    problem.upload_numeric();
    test::status_require(spacepdhcg_cuda_workspace_update_async(
        workspace, problem.fingerprint, &numeric, problem.exchange.consumer_stream), "changed coefficient update");
    test::status_require(spacepdhcg_cuda_workspace_wait(workspace), "changed coefficient wait");
    updated = test::solve_and_wait(workspace, problem);
    test::require_close(updated.coefficient_change_norm, 0.25, 1.0e-15, "coefficient norm reduction");
    test::require_close(updated.coefficient_change_max, soc ? 0.125 : 0.25, 1.0e-15, "coefficient maximum reduction");
    test::status_require(spacepdhcg_cuda_workspace_refresh_scaling_async(
        workspace, problem.exchange.consumer_stream), "device-only explicit refresh");
    test::status_require(spacepdhcg_cuda_workspace_wait(workspace), "explicit refresh wait");
    updated = test::solve_and_wait(workspace, problem);
    test::require(updated.termination == SPACEPDHCG_CUDA_TERMINATION_OPTIMAL, "solve after explicit refresh");
    // Recompute a deliberately infeasible resident point, not merely the cached
    // converged report. This exercises nonzero box/equality/cone/stationarity terms.
    spacepdhcg_cuda_pointer_snapshot pointers{};
    test::status_require(spacepdhcg_cuda_workspace_pointer_snapshot(workspace, &pointers),
                         "resident primal for residual regression");
    auto displaced = primal;
    for (size_t i = 0; i < displaced.size(); ++i) displaced[i] += i % 2 == 0 ? -2.0 : 3.0;
    test::cuda_require(cudaMemcpyAsync(reinterpret_cast<void*>(pointers.primal), displaced.data(),
        displaced.size() * sizeof(double), cudaMemcpyHostToDevice, problem.stream), "displace resident primal");
    compare_standalone_residuals(workspace, problem, blocks);
    test::destroy_workspace(workspace);
}

void cancellation_case(int blocks, int delay_ms) {
    test::ProblemStorage problem(false, true);
    problem.variables = 1;
    problem.scalar_rows = 2;
    problem.h_q_offsets = {0, 1};
    problem.h_q_indices = {0};
    problem.h_q = {1.0};
    problem.h_a_offsets = {0, 2};
    problem.h_a_indices = {0, 1};
    problem.h_a = {1.0, 1.0};
    problem.h_c = {-0.25};
    problem.h_scalar_lower = {0.0, 1.0};
    problem.h_scalar_upper = {0.0, 1.0};
    problem.h_variable_lower = {-INFINITY};
    problem.h_variable_upper = {INFINITY};
    problem.materialise();
    auto* workspace = test::create_workspace(problem);
    test::status_require(spacepdhcg_cuda_workspace_wait(workspace), "cancellation create wait");
    test::status_require(spacepdhcg_cuda_workspace_set_execution_blocks(workspace, blocks), "cancellation grid");
    auto options = test::solve_options(1.0e-12, 100'000'000U);
    // Cancellation must report the final resident point even between scheduled checks.
    options.residual_check_frequency = 1'000'000U;
    for (int repeat = 0; repeat < 3; ++repeat) {
        test::status_require(spacepdhcg_cuda_workspace_solve_async(
            workspace, &options, problem.exchange.consumer_stream), "cancellation solve");
        std::thread canceller([&] {
            std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms));
            test::status_require(spacepdhcg_cuda_workspace_cancel(workspace), "cooperative cancel");
        });
        const auto started = std::chrono::steady_clock::now();
        test::status_require(spacepdhcg_cuda_workspace_wait(workspace), "cancellation wait");
        canceller.join();
        const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
        test::require(elapsed < 2.0, "bounded cooperative cancellation");
        spacepdhcg_cuda_diagnostics diagnostics{};
        test::status_require(spacepdhcg_cuda_workspace_diagnostics(workspace, &diagnostics), "cancellation report");
        test::require(diagnostics.termination == SPACEPDHCG_CUDA_TERMINATION_CANCELLED, "uniform cancelled result");
        const auto primal = problem.primal.download(problem.stream);
        test::require_close(diagnostics.objective,
            0.5 * primal[0] * primal[0] - 0.25 * primal[0], 1.0e-12,
            "cancelled objective must describe the returned resident primal");
        test::status_require(spacepdhcg_cuda_workspace_residuals_async(
            workspace, problem.exchange.consumer_stream), "cancelled residual recheck");
        test::status_require(spacepdhcg_cuda_workspace_wait(workspace), "cancelled residual wait");
        spacepdhcg_cuda_diagnostics fresh{};
        test::status_require(spacepdhcg_cuda_workspace_diagnostics(workspace, &fresh),
                             "cancelled residual diagnostics");
        test::require_close(diagnostics.natural_residual_inf, fresh.natural_residual_inf,
                            1.0e-12, "cancelled report must describe the current residual");
    }
    test::destroy_workspace(workspace);
}

void signed_zero_scaling_case() {
    // Structural -0.0 must not suppress a positive Ruiz row maximum when that
    // maximum is computed with integer atomics on the double's bit pattern.
    test::ProblemStorage problem(false, true);
    problem.variables = 3;
    problem.scalar_rows = 1;
    problem.h_q_offsets = {0, 1, 2, 3};
    problem.h_q_indices = {0, 1, 2};
    problem.h_q = {1.0, 2.0, 1.0};
    problem.h_a_offsets = {0, 1, 2, 3};
    problem.h_a_indices = {0, 0, 0};
    problem.h_a = {4.0, 4.0, -0.0};
    problem.h_c = {-1.0, -1.0, 0.0};
    problem.h_scalar_lower = {4.0};
    problem.h_scalar_upper = {4.0};
    problem.h_variable_lower = {0.0, 0.0, -INFINITY};
    problem.h_variable_upper = {1.0, 1.0, INFINITY};
    problem.materialise();
    auto* workspace = test::create_workspace(problem);
    test::status_require(spacepdhcg_cuda_workspace_wait(workspace), "signed-zero create");
    std::vector<double> serial;
    for (int blocks : {0, 1, 8}) {
        test::status_require(spacepdhcg_cuda_workspace_set_execution_blocks(workspace, blocks), "signed-zero grid");
        test::status_require(spacepdhcg_cuda_workspace_refresh_scaling_async(workspace, problem.exchange.consumer_stream), "signed-zero refresh");
        test::status_require(spacepdhcg_cuda_workspace_wait(workspace), "signed-zero refresh wait");
        spacepdhcg_cuda_pointer_snapshot pointers{};
        test::status_require(spacepdhcg_cuda_workspace_pointer_snapshot(workspace, &pointers), "signed-zero snapshot");
        std::vector<double> scales(4);
        test::cuda_require(cudaMemcpy(scales.data(), reinterpret_cast<void*>(pointers.scaling), 4 * sizeof(double), cudaMemcpyDeviceToHost), "signed-zero scales");
        if (blocks == 0) serial = scales;
        else for (size_t i = 0; i < scales.size(); ++i) {
            test::require_close(scales[i], serial[i], 1.0e-14, "signed-zero scaling parity");
        }
    }
    const auto result = test::solve_and_wait(workspace, problem);
    test::require(result.termination == SPACEPDHCG_CUDA_TERMINATION_OPTIMAL, "signed-zero solve");
    const auto primal = problem.primal.download(problem.stream);
    test::require_close(primal[0], 2.0/3.0, 2.0e-6, "signed-zero analytic x0");
    test::require_close(primal[1], 1.0/3.0, 2.0e-6, "signed-zero analytic x1");
    test::require_close(primal[2], 0.0, 2.0e-6, "signed-zero analytic x2");
    test::destroy_workspace(workspace);
}

void mixed_variable_cones_case(int blocks) {
    test::ProblemStorage problem(false, true);
    constexpr int cones = 128;
    problem.variables = 3 * cones;
    problem.h_q_offsets.push_back(0);
    problem.h_a_offsets.assign(problem.variables + 1, 0);
    for (int i = 0; i < problem.variables; ++i) {
        problem.h_q_offsets.push_back(i + 1);
        problem.h_q_indices.push_back(i);
        problem.h_q.push_back(1.0);
        problem.h_c.push_back(i % 3 == 0 ? -2.0 : 0.0);
        problem.h_variable_lower.push_back(-INFINITY);
        problem.h_variable_upper.push_back(INFINITY);
    }
    for (int cone = 0; cone < cones; ++cone) {
        problem.variable_cones.push_back({cone % 2 == 0
            ? SPACEPDHCG_CUDA_CONE_SECOND_ORDER : SPACEPDHCG_CUDA_CONE_ROTATED_SECOND_ORDER,
            3 * cone, 1, 0.0});
    }
    problem.materialise();
    auto* workspace = test::create_workspace(problem);
    test::status_require(spacepdhcg_cuda_workspace_wait(workspace), "mixed cones create");
    test::status_require(spacepdhcg_cuda_workspace_set_execution_blocks(workspace, blocks), "mixed cones grid");
    const auto result = test::solve_and_wait(workspace, problem);
    test::require(result.termination == SPACEPDHCG_CUDA_TERMINATION_OPTIMAL, "mixed cones convergence");
    const auto primal = problem.primal.download(problem.stream);
    for (int cone = 0; cone < cones; ++cone) {
        test::require_close(primal[3 * cone], 1.0, 2.0e-6, "mixed cone vector optimum");
        test::require_close(primal[3 * cone + 1], cone % 2 == 0 ? 0.0 : std::sqrt(0.5),
            2.0e-6, "mixed cone first scalar");
        test::require_close(primal[3 * cone + 2], cone % 2 == 0 ? 1.0 : std::sqrt(0.5),
            2.0e-6, "mixed cone radius");
    }
    test::destroy_workspace(workspace);
}
}  // namespace

int main(int argc, char** argv) {
    const bool sanitizer = argc == 2 && std::strcmp(argv[1], "--sanitizer") == 0;
    test::require(argc == 1 || sanitizer, "expected optional --sanitizer");
    int device = 0;
    cudaDeviceProp properties{};
    test::cuda_require(cudaGetDevice(&device), "device");
    test::cuda_require(cudaGetDeviceProperties(&properties, device), "properties");
    if (!properties.cooperativeLaunch) { std::puts("SKIP: cooperative launch unsupported"); return 77; }
    for (const int blocks : {0, 1, 2, 8, 32}) {
        auto box = test::make_box_problem(false, true);
        run_case(box, blocks, false);
        auto soc = test::make_soc_problem(false, true);
        run_case(soc, blocks, true);
        test::ProblemStorage large(false, true);
        replicated_box(large, sanitizer ? 64 : 4096);
        run_case(large, blocks, false);
    }
    for (const int delay : {0, 1, 10}) cancellation_case(8, delay);
    cancellation_case(0, 10);
    signed_zero_scaling_case();
    for (int blocks : {0, 8, 32}) mixed_variable_cones_case(blocks);
    return 0;
}
