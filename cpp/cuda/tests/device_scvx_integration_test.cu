#include "cuda_test_support.hpp"
#include "spacepdhcg/cuda/device_scvx_c_api.h"
#include "spacepdhcg/scvx/low_thrust_driver.hpp"
#include "spacepdhcg/scvx/powered_descent_3dof_driver.hpp"
#include "spacepdhcg/transcription/hcw_rendezvous.hpp"
#include "spacepdhcg/transcription/low_thrust.hpp"
#include "spacepdhcg/transcription/powered_descent_3dof.hpp"
#include "spacepdhcg/transcription/powered_descent_6dof.hpp"

#include <pdhcg.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <map>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace test = spacepdhcg::cuda::test;
namespace core = spacepdhcg::core;
namespace dynamics = spacepdhcg::dynamics;
namespace transcription = spacepdhcg::transcription;

namespace {

bool sanitizer_mode = false;
bool tight_residual_mode = false;
bool diagnostic_mode = false;
bool dump_mode = false;
bool tight_after_loose_mode = false;
bool default_stream_mode = false;
bool refresh_before_tight_mode = false;
bool repeated_tight_mode = false;
std::uint64_t tight_iteration_limit = 1'000'000U;

spacepdhcg_cuda_cone_kind cone_kind(const spacepdhcg::ConeKind kind) {
    switch (kind) {
        case spacepdhcg::ConeKind::second_order:
            return SPACEPDHCG_CUDA_CONE_SECOND_ORDER;
        case spacepdhcg::ConeKind::rotated_second_order:
            return SPACEPDHCG_CUDA_CONE_ROTATED_SECOND_ORDER;
        case spacepdhcg::ConeKind::exponential:
            return SPACEPDHCG_CUDA_CONE_EXPONENTIAL;
        case spacepdhcg::ConeKind::power:
            return SPACEPDHCG_CUDA_CONE_POWER;
        case spacepdhcg::ConeKind::positive_semidefinite:
            return SPACEPDHCG_CUDA_CONE_POSITIVE_SEMIDEFINITE;
    }
    return SPACEPDHCG_CUDA_CONE_SECOND_ORDER;
}

void materialise(
    test::ProblemStorage& storage,
    const core::FixedStructure& structure,
    const core::NumericValues& values
) {
    storage.fingerprint = structure.fingerprint();
    storage.variables = structure.quadratic.columns;
    storage.scalar_rows = structure.scalar_constraint.rows;
    storage.affine_rows =
        structure.affine_cone.has_value() ? structure.affine_cone->rows : 0;
    storage.h_q_offsets.assign(
        structure.quadratic.offsets.begin(), structure.quadratic.offsets.end()
    );
    storage.h_q_indices.assign(
        structure.quadratic.indices.begin(), structure.quadratic.indices.end()
    );
    storage.h_a_offsets.assign(
        structure.scalar_constraint.offsets.begin(),
        structure.scalar_constraint.offsets.end()
    );
    storage.h_a_indices.assign(
        structure.scalar_constraint.indices.begin(),
        structure.scalar_constraint.indices.end()
    );
    if (structure.affine_cone.has_value()) {
        storage.h_f_offsets.assign(
            structure.affine_cone->offsets.begin(), structure.affine_cone->offsets.end()
        );
        storage.h_f_indices.assign(
            structure.affine_cone->indices.begin(), structure.affine_cone->indices.end()
        );
    } else {
        storage.h_f_offsets.assign(static_cast<std::size_t>(storage.variables) + 1U, 0);
    }
    storage.h_q = values.quadratic;
    storage.h_a = values.scalar_constraint;
    storage.h_f = values.affine_cone;
    storage.h_c = values.linear_objective;
    storage.h_scalar_lower = values.scalar_lower;
    storage.h_scalar_upper = values.scalar_upper;
    storage.h_affine_offset = values.affine_offset;
    storage.h_variable_lower = values.variable_lower;
    storage.h_variable_upper = values.variable_upper;
    for (const auto& cone : structure.affine_cones) {
        storage.affine_cones.push_back(
            {cone_kind(cone.kind), cone.start, cone.vector_dimension, cone.power_alpha}
        );
    }
    for (const auto& cone : structure.variable_cones) {
        storage.variable_cones.push_back(
            {cone_kind(cone.kind), cone.start, cone.vector_dimension, cone.power_alpha}
        );
    }
    storage.materialise();
}

std::map<std::pair<std::size_t, std::size_t>, int> positions(
    const core::CscPattern& pattern
) {
    std::map<std::pair<std::size_t, std::size_t>, int> result;
    for (spacepdhcg::Index column = 0; column < pattern.columns; ++column) {
        const auto begin = pattern.offsets[static_cast<std::size_t>(column)];
        const auto end = pattern.offsets[static_cast<std::size_t>(column) + 1U];
        for (spacepdhcg::Index slot = begin; slot < end; ++slot) {
            result[{
                static_cast<std::size_t>(pattern.indices[static_cast<std::size_t>(slot)]),
                static_cast<std::size_t>(column),
            }] = slot;
        }
    }
    return result;
}

struct DynamicsMaps {
    std::vector<int> state;
    std::vector<int> control;
    std::vector<int> next;
    std::vector<int> virtual_control;
};

template <typename StateRange, typename ControlRange, typename VirtualRange>
DynamicsMaps make_maps(
    const core::CscPattern& pattern,
    const std::size_t intervals,
    const std::size_t state_dimension,
    const std::size_t control_dimension,
    const std::size_t dynamics_row_start,
    StateRange state_range,
    ControlRange control_range,
    VirtualRange virtual_range,
    const bool has_virtual
) {
    const auto lookup = positions(pattern);
    DynamicsMaps maps;
    maps.state.reserve(intervals * state_dimension * state_dimension);
    maps.control.reserve(intervals * state_dimension * control_dimension);
    maps.next.reserve(intervals * state_dimension);
    if (has_virtual) {
        maps.virtual_control.reserve(intervals * state_dimension);
    }
    for (std::size_t interval = 0; interval < intervals; ++interval) {
        const auto current = state_range(interval);
        const auto next = state_range(interval + 1U);
        const auto control = control_range(interval);
        const auto virtual_control = virtual_range(interval);
        for (std::size_t row = 0; row < state_dimension; ++row) {
            const auto matrix_row =
                dynamics_row_start + interval * state_dimension + row;
            for (std::size_t column = 0; column < state_dimension; ++column) {
                maps.state.push_back(lookup.at({matrix_row, current.start + column}));
            }
            for (std::size_t column = 0; column < control_dimension; ++column) {
                maps.control.push_back(lookup.at({matrix_row, control.start + column}));
            }
            maps.next.push_back(lookup.at({matrix_row, next.start + row}));
            if (has_virtual) {
                maps.virtual_control.push_back(
                    lookup.at({matrix_row, virtual_control.start + row})
                );
            }
        }
    }
    return maps;
}

spacepdhcg_accelerator_buffer_view null_int_view() {
    return test::view(
        nullptr, 0U, false, SPACEPDHCG_SCALAR_INT32, SPACEPDHCG_ACCESS_READ_ONLY
    );
}

struct IntegrationResult {
    spacepdhcg_cuda_diagnostics diagnostics{};
    std::uint64_t topology_allocations{0U};
    std::uint64_t topology_copies{0U};
    std::uint64_t update_allocations{0U};
    double maximum_solution_magnitude{0.0};
};

cone_type_t upstream_cone_kind(const spacepdhcg_cuda_cone_kind kind) {
    switch (kind) {
        case SPACEPDHCG_CUDA_CONE_SECOND_ORDER:
            return CONE_STANDARD_SOC;
        case SPACEPDHCG_CUDA_CONE_ROTATED_SECOND_ORDER:
            return CONE_ROTATED_SOC;
        case SPACEPDHCG_CUDA_CONE_EXPONENTIAL:
            return CONE_EXPONENTIAL;
        case SPACEPDHCG_CUDA_CONE_POWER:
            return CONE_POWER;
        case SPACEPDHCG_CUDA_CONE_POSITIVE_SEMIDEFINITE:
            return CONE_PSD;
    }
    return CONE_STANDARD_SOC;
}

template <typename T>
void print_diagnostic_vector(const char* name, const std::vector<T>& values) {
    std::printf("PD3DATA %s", name);
    for (const auto value : values) {
        if constexpr (std::is_floating_point_v<T>) {
            if (std::isinf(value)) {
                std::printf(value < 0.0 ? " -inf" : " inf");
            } else {
                std::printf(" %.17g", value);
            }
        } else {
            std::printf(" %lld", static_cast<long long>(value));
        }
    }
    std::printf("\n");
}

void print_diagnostic_problem(const test::ProblemStorage& problem) {
    std::printf(
        "PD3DATA dimensions %d %d %d\n",
        problem.variables,
        problem.scalar_rows,
        problem.affine_rows
    );
    print_diagnostic_vector("q_offsets", problem.h_q_offsets);
    print_diagnostic_vector("q_indices", problem.h_q_indices);
    print_diagnostic_vector("a_offsets", problem.h_a_offsets);
    print_diagnostic_vector("a_indices", problem.h_a_indices);
    print_diagnostic_vector("f_offsets", problem.h_f_offsets);
    print_diagnostic_vector("f_indices", problem.h_f_indices);
    print_diagnostic_vector("q", problem.h_q);
    print_diagnostic_vector("a", problem.h_a);
    print_diagnostic_vector("f", problem.h_f);
    print_diagnostic_vector("c", problem.h_c);
    print_diagnostic_vector("scalar_lower", problem.h_scalar_lower);
    print_diagnostic_vector("scalar_upper", problem.h_scalar_upper);
    print_diagnostic_vector("affine_offset", problem.h_affine_offset);
    print_diagnostic_vector("variable_lower", problem.h_variable_lower);
    print_diagnostic_vector("variable_upper", problem.h_variable_upper);
    for (const auto& cone : problem.affine_cones) {
        std::printf(
            "PD3CONE affine %d %d %d %.17g\n",
            static_cast<int>(cone.kind),
            cone.start,
            cone.vector_dimension,
            cone.power_alpha
        );
    }
    for (const auto& cone : problem.variable_cones) {
        std::printf(
            "PD3CONE variable %d %d %d %.17g\n",
            static_cast<int>(cone.kind),
            cone.start,
            cone.vector_dimension,
            cone.power_alpha
        );
    }
}

void run_upstream_diagnostic(test::ProblemStorage& problem) {
    problem.h_q = problem.q.download(problem.stream);
    problem.h_a = problem.a.download(problem.stream);
    problem.h_f = problem.f.download(problem.stream);
    problem.h_c = problem.c.download(problem.stream);
    problem.h_scalar_lower = problem.scalar_lower.download(problem.stream);
    problem.h_scalar_upper = problem.scalar_upper.download(problem.stream);
    problem.h_affine_offset = problem.affine_offset.download(problem.stream);
    problem.h_variable_lower = problem.variable_lower.download(problem.stream);
    problem.h_variable_upper = problem.variable_upper.download(problem.stream);
    print_diagnostic_problem(problem);
    if (dump_mode) {
        return;
    }

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
    std::vector<cone_spec_t> affine_cones;
    affine_cones.reserve(problem.affine_cones.size());
    for (const auto& cone : problem.affine_cones) {
        affine_cones.push_back({
            upstream_cone_kind(cone.kind),
            cone.start,
            cone.vector_dimension,
            cone.power_alpha,
            nullptr,
        });
    }
    std::vector<cone_spec_t> variable_cones;
    variable_cones.reserve(problem.variable_cones.size());
    for (const auto& cone : problem.variable_cones) {
        variable_cones.push_back({
            upstream_cone_kind(cone.kind),
            cone.start,
            cone.vector_dimension,
            cone.power_alpha,
            nullptr,
        });
    }
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
        static_cast<int>(variable_cones.size()),
        variable_cones.empty() ? nullptr : variable_cones.data(),
        &f,
        problem.h_affine_offset.data(),
        static_cast<int>(affine_cones.size()),
        affine_cones.data()
    );
    test::require(qp != nullptr, "diagnostic one-shot QP creation failed");
    pdhg_parameters_t parameters{};
    set_default_parameters(&parameters);
    parameters.verbose = 0;
    parameters.presolve = false;
    parameters.termination_criteria.eps_optimal_relative = 1.0e-6;
    parameters.termination_criteria.eps_feasible_relative = 1.0e-6;
    parameters.termination_criteria.iteration_limit = 1'000'000;
    pdhcg_result_t* result = solve_qp_problem(qp, &parameters);
    test::require(result != nullptr, "diagnostic one-shot solve failed");
    std::printf(
        "{\"case\":\"pd3_upstream_oneshot\",\"start\":\"cold\",\"termination\":%d,"
        "\"iterations\":%d,\"relative_primal\":%.17g,\"relative_dual\":%.17g,"
        "\"objective_gap\":%.17g,\"relative_objective_gap\":%.17g}\n",
        static_cast<int>(result->termination_reason),
        result->total_count,
        result->relative_primal_residual,
        result->relative_dual_residual,
        result->objective_gap,
        result->relative_objective_gap
    );
    set_start_values(qp, result->primal_solution, result->dual_solution);
    pdhcg_result_t* warm_result = solve_qp_problem(qp, &parameters);
    test::require(warm_result != nullptr, "diagnostic warm one-shot solve failed");
    std::printf(
        "{\"case\":\"pd3_upstream_oneshot\",\"start\":\"primal_dual\","
        "\"termination\":%d,\"iterations\":%d,\"relative_primal\":%.17g,"
        "\"relative_dual\":%.17g,\"objective_gap\":%.17g,"
        "\"relative_objective_gap\":%.17g}\n",
        static_cast<int>(warm_result->termination_reason),
        warm_result->total_count,
        warm_result->relative_primal_residual,
        warm_result->relative_dual_residual,
        warm_result->objective_gap,
        warm_result->relative_objective_gap
    );
    pdhcg_result_free(warm_result);
    pdhcg_result_free(result);
    qp_problem_free(qp);
}

template <std::size_t StateDimension, std::size_t ControlDimension>
IntegrationResult run_resident_sequence(
    const core::FixedStructure& structure,
    const core::NumericValues& values,
    const std::vector<double>& reference_states,
    const std::vector<double>& reference_controls,
    const DynamicsMaps& maps,
    const std::size_t intervals,
    const std::size_t dynamics_row_start,
    const spacepdhcg_cuda_dynamics_config& dynamics_config
) {
    test::ProblemStorage problem(false, !default_stream_mode);
    materialise(problem, structure, values);
    test::CudaBuffer<double> states(reference_states.size(), false);
    test::CudaBuffer<double> controls(reference_controls.size(), false);
    test::CudaBuffer<double> propagated(intervals * StateDimension, false);
    test::CudaBuffer<double> transition(intervals * StateDimension * StateDimension, false);
    test::CudaBuffer<double> sensitivity(intervals * StateDimension * ControlDimension, false);
    test::CudaBuffer<double> offset(intervals * StateDimension, false);
    test::CudaBuffer<int> state_positions(maps.state.size(), false);
    test::CudaBuffer<int> control_positions(maps.control.size(), false);
    test::CudaBuffer<int> next_positions(maps.next.size(), false);
    test::CudaBuffer<int> virtual_positions(maps.virtual_control.size(), false);
    states.upload(reference_states, problem.stream);
    controls.upload(reference_controls, problem.stream);
    state_positions.upload(maps.state, problem.stream);
    control_positions.upload(maps.control, problem.stream);
    next_positions.upload(maps.next, problem.stream);
    virtual_positions.upload(maps.virtual_control, problem.stream);
    const auto rw64 = [](auto& buffer) {
        return test::view(
            buffer.get(), buffer.size(), false, SPACEPDHCG_SCALAR_FLOAT64,
            SPACEPDHCG_ACCESS_READ_WRITE
        );
    };
    const auto ro64 = [](auto& buffer) {
        return test::view(
            buffer.get(), buffer.size(), false, SPACEPDHCG_SCALAR_FLOAT64,
            SPACEPDHCG_ACCESS_READ_ONLY
        );
    };
    const auto ro32 = [](auto& buffer) {
        return test::view(
            buffer.get(), buffer.size(), false, SPACEPDHCG_SCALAR_INT32,
            SPACEPDHCG_ACCESS_READ_ONLY
        );
    };
    const spacepdhcg_cuda_variational_request linearise{
        SPACEPDHCG_CUDA_WORKSPACE_ABI_VERSION,
        intervals,
        ro64(states),
        ro64(controls),
        rw64(propagated),
        rw64(transition),
        rw64(sensitivity),
        rw64(offset),
    };
    const spacepdhcg_cuda_csc_dynamics_fill fill{
        SPACEPDHCG_CUDA_WORKSPACE_ABI_VERSION,
        intervals,
        StateDimension,
        ControlDimension,
        ro64(transition),
        ro64(sensitivity),
        ro64(offset),
        ro32(state_positions),
        ro32(control_positions),
        ro32(next_positions),
        maps.virtual_control.empty() ? null_int_view() : ro32(virtual_positions),
        problem.numeric_views().scalar_constraint,
        problem.numeric_views().scalar_lower,
        problem.numeric_views().scalar_upper,
        dynamics_row_start,
    };
    auto* workspace = test::create_workspace(problem);
    spacepdhcg_cuda_pointer_snapshot pointers_before{};
    test::status_require(
        spacepdhcg_cuda_workspace_pointer_snapshot(workspace, &pointers_before),
        "pointer snapshot before resident sequence"
    );
    spacepdhcg_cuda_diagnostics diagnostics{};
    const int sequence_iterations =
        sanitizer_mode || diagnostic_mode || dump_mode ? 1 : 2;
    for (int iteration = 0; iteration < sequence_iterations; ++iteration) {
        test::status_require(
            spacepdhcg_cuda_variational_rk4_async(
                &dynamics_config, &linearise, problem.exchange.consumer_stream
            ),
            "resident coefficient generation"
        );
        test::status_require(
            spacepdhcg_cuda_fill_dynamics_csc_async(
                &fill, problem.exchange.consumer_stream
            ),
            "resident direct CSC fill"
        );
        auto numeric = problem.numeric_views();
        test::status_require(
            spacepdhcg_cuda_workspace_update_async(
                workspace, problem.fingerprint, &numeric, problem.exchange.consumer_stream
            ),
            "resident values-only update"
        );
        test::status_require(spacepdhcg_cuda_workspace_wait(workspace), "resident update wait");
        if (iteration > 0) {
            if (refresh_before_tight_mode) {
                test::status_require(
                    spacepdhcg_cuda_workspace_refresh_scaling_async(
                        workspace,
                        problem.exchange.consumer_stream
                    ),
                    "forced scaling refresh before tight solve"
                );
                test::status_require(
                    spacepdhcg_cuda_workspace_wait(workspace),
                    "forced scaling refresh wait"
                );
            }
            test::status_require(
                spacepdhcg_cuda_workspace_warm_start_async(
                    workspace,
                    SPACEPDHCG_CUDA_WARM_START_FULL_RETAINED,
                    nullptr,
                    problem.exchange.consumer_stream
                ),
                "retained full-state warm start"
            );
            test::status_require(
                spacepdhcg_cuda_workspace_wait(workspace), "retained warm-start wait"
            );
        }
        const auto solve = sanitizer_mode
            ? test::solve_options(1.0e-2, 5'000U)
            : (dump_mode
                   ? test::solve_options(1.0e-6, 1U)
            : ((tight_residual_mode || diagnostic_mode || repeated_tight_mode
                || (tight_after_loose_mode && iteration > 0))
                   ? test::solve_options(1.0e-6, tight_iteration_limit)
                   : (tight_after_loose_mode
                          ? test::solve_options(1.0e-2, 200'000U)
                          : test::solve_options(1.0e-2, 200'000U))));
        if (diagnostic_mode || dump_mode) {
            run_upstream_diagnostic(problem);
        }
        if (dump_mode) {
            test::destroy_workspace(workspace);
            return {};
        }
        diagnostics = test::solve_and_wait(workspace, problem, solve);
        if (diagnostics.termination != SPACEPDHCG_CUDA_TERMINATION_OPTIMAL) {
            if (tight_residual_mode || diagnostic_mode || dump_mode
                || tight_after_loose_mode || repeated_tight_mode) {
                std::fprintf(
                    stderr,
                    "{\"case\":\"tight_final_residual\",\"termination\":%d,"
                    "\"iterations\":%llu,\"requested\":1e-6,"
                    "\"relative_primal\":%.9g,\"relative_dual\":%.9g,"
                    "\"natural_residual\":%.9g}\n",
                    static_cast<int>(diagnostics.termination),
                    static_cast<unsigned long long>(diagnostics.iterations),
                    diagnostics.relative_primal_residual,
                    diagnostics.relative_dual_residual,
                    diagnostics.natural_residual_inf
                );
                if (repeated_tight_mode && iteration == 0) {
                    continue;
                }
                break;
            }
            std::fprintf(
                stderr,
                "resident solve failed: variables=%d rows=%d+%d termination=%d "
                "iterations=%llu primal=%.6g dual=%.6g natural=%.6g\n",
                problem.variables,
                problem.scalar_rows,
                problem.affine_rows,
                static_cast<int>(diagnostics.termination),
                static_cast<unsigned long long>(diagnostics.iterations),
                diagnostics.relative_primal_residual,
                diagnostics.relative_dual_residual,
                diagnostics.natural_residual_inf
            );
            std::exit(6);
        }
        test::status_require(
            spacepdhcg_cuda_workspace_residuals_async(
                workspace, problem.exchange.consumer_stream
            ),
            "independent resident residual"
        );
        test::status_require(
            spacepdhcg_cuda_workspace_wait(workspace), "independent residual wait"
        );
        test::status_require(
            spacepdhcg_cuda_workspace_diagnostics(workspace, &diagnostics),
            "resident diagnostics"
        );
        if (diagnostics.natural_residual_inf > 1.0e-2) {
            test::status_require(
                spacepdhcg_cuda_workspace_warm_start_async(
                    workspace,
                    SPACEPDHCG_CUDA_WARM_START_FULL_RETAINED,
                    nullptr,
                    problem.exchange.consumer_stream
                ),
                "under-solved identical-CQP retained warm start"
            );
            test::status_require(
                spacepdhcg_cuda_workspace_wait(workspace),
                "under-solved retained warm-start wait"
            );
            const auto refined_solve = test::solve_options(1.0e-5, 1'000'000U);
            diagnostics = test::solve_and_wait(workspace, problem, refined_solve);
            test::status_require(
                spacepdhcg_cuda_workspace_residuals_async(
                    workspace, problem.exchange.consumer_stream
                ),
                "refined independent resident residual"
            );
            test::status_require(
                spacepdhcg_cuda_workspace_wait(workspace),
                "refined independent residual wait"
            );
            test::status_require(
                spacepdhcg_cuda_workspace_diagnostics(workspace, &diagnostics),
                "refined resident diagnostics"
            );
        }
    }
    spacepdhcg_cuda_pointer_snapshot pointers_after{};
    test::status_require(
        spacepdhcg_cuda_workspace_pointer_snapshot(workspace, &pointers_after),
        "pointer snapshot after resident sequence"
    );
    test::require(
        pointers_before.scalar_values == pointers_after.scalar_values
            && pointers_before.primal == pointers_after.primal
            && pointers_before.dual == pointers_after.dual,
        "persistent workspace pointers changed"
    );
    test::require(diagnostics.hidden_cpu_fallback == 0, "hidden CPU fallback detected");
    test::require(
        diagnostics.topology_allocation_delta_last_update == 0U,
        "post-create topology allocation detected"
    );
    test::require(
        diagnostics.topology_index_copy_delta_last_update == 0U,
        "steady-state topology index copy detected"
    );
    test::require(
        diagnostics.allocation_delta_last_update == 0U,
        "steady-state allocation detected"
    );
    const auto solution = problem.primal.download(problem.stream);
    if (diagnostic_mode) {
        print_diagnostic_vector("persistent_primal", solution);
        print_diagnostic_vector(
            "persistent_dual",
            problem.dual.download(problem.stream)
        );
    }
    double maximum_solution = 0.0;
    for (const auto value : solution) {
        test::require(std::isfinite(value), "resident solution is non-finite");
        maximum_solution = std::max(maximum_solution, std::abs(value));
    }
    const IntegrationResult result{
        diagnostics,
        diagnostics.topology_allocation_delta_last_update,
        diagnostics.topology_index_copy_delta_last_update,
        diagnostics.allocation_delta_last_update,
        maximum_solution,
    };
    test::destroy_workspace(workspace);
    return result;
}

spacepdhcg_cuda_dynamics_config model_config(
    const spacepdhcg_cuda_dynamics_model model,
    const double step
) {
    return {
        SPACEPDHCG_CUDA_WORKSPACE_ABI_VERSION,
        model,
        step,
        1.13e-3,
        {0.0, 0.0, -3.711},
        398'600.4418,
        1.0e-3,
        model == SPACEPDHCG_CUDA_DYNAMICS_LOW_THRUST ? 3.4e-5 : 4.6e-4,
        {2'500.0, 2'200.0, 1'800.0},
    };
}

template <typename State>
std::vector<double> flatten_states(const std::vector<State>& states, const std::size_t intervals) {
    std::vector<double> result;
    result.reserve(intervals * states.front().size());
    for (std::size_t interval = 0; interval < intervals; ++interval) {
        result.insert(result.end(), states[interval].begin(), states[interval].end());
    }
    return result;
}

template <typename Control>
std::vector<double> flatten_controls(const std::vector<Control>& controls) {
    std::vector<double> result;
    for (const auto& control : controls) {
        result.insert(result.end(), control.begin(), control.end());
    }
    return result;
}

IntegrationResult run_hcw() {
    transcription::HcwRendezvousConfig config;
    config.intervals = 2U;
    config.step_seconds = 10.0;
    transcription::HcwRendezvousCqp subproblem(config);
    const dynamics::HcwState initial{};
    const dynamics::HcwState target{};
    const auto values = subproblem.values(initial, target);
    std::vector<dynamics::HcwState> states(2U, initial);
    std::vector<dynamics::HcwControl> controls(2U);
    const auto& layout = subproblem.layout();
    const auto maps = make_maps(
        subproblem.structure().scalar_constraint,
        2U,
        6U,
        3U,
        layout.dynamics_row(),
        [&layout](std::size_t node) {
            return transcription::IndexRange{layout.state_index(node, 0U), 6U};
        },
        [&layout](std::size_t interval) {
            return transcription::IndexRange{layout.control_index(interval, 0U), 3U};
        },
        [](std::size_t) { return transcription::IndexRange{}; },
        false
    );
    return run_resident_sequence<6U, 3U>(
        subproblem.structure(),
        values,
        flatten_states(states, 2U),
        flatten_controls(controls),
        maps,
        2U,
        layout.dynamics_row(),
        model_config(SPACEPDHCG_CUDA_DYNAMICS_HCW, config.step_seconds)
    );
}

IntegrationResult run_pd3() {
    transcription::PoweredDescentScvxConfig config;
    config.intervals = 2U;
    config.step_seconds = 0.25;
    config.discretisation = transcription::DiscretisationMethod::rk4_variational;
    config.virtual_l1_weight = 10.0;
    config.virtual_quadratic_weight = 1.0e-3;
    config.virtual_epigraph_regularisation = 1.0e-3;
    transcription::PoweredDescent3DofSubproblem subproblem(
        dynamics::PoweredDescent3DofModel{}, config
    );
    const dynamics::PoweredDescentState initial{
        0.0, 0.0, 100.0, 0.0, 0.0, 0.0, 2'000.0
    };
    const std::vector<dynamics::PoweredDescentControl> controls{
        {0.0, 0.0, 7'422.0, 7'422.0},
        {0.0, 0.0, 7'421.5, 7'421.5},
    };
    const auto states = subproblem.model().rollout(initial, controls, config.step_seconds);
    const std::array<double, 3U> target_position{
        states.back()[0], states.back()[1], states.back()[2]
    };
    const std::array<double, 3U> target_velocity{
        states.back()[3], states.back()[4], states.back()[5]
    };
    const auto values = subproblem.values(
        states, controls, initial, target_position, target_velocity
    );
    const auto& layout = subproblem.layout();
    const auto maps = make_maps(
        subproblem.structure().scalar_constraint,
        config.intervals,
        7U,
        4U,
        layout.dynamics_rows().start,
        [&layout](std::size_t node) { return layout.state(node); },
        [&layout](std::size_t interval) { return layout.control(interval); },
        [&layout](std::size_t interval) { return layout.virtual_control(interval); },
        true
    );
    return run_resident_sequence<7U, 4U>(
        subproblem.structure(),
        values,
        flatten_states(states, config.intervals),
        flatten_controls(controls),
        maps,
        config.intervals,
        layout.dynamics_rows().start,
        model_config(
            SPACEPDHCG_CUDA_DYNAMICS_POWERED_DESCENT_3DOF, config.step_seconds
        )
    );
}

IntegrationResult run_low_thrust() {
    transcription::LowThrustScvxConfig config;
    config.intervals = 2U;
    config.step_seconds = 1.0;
    config.discretisation = transcription::DiscretisationMethod::rk4_variational;
    config.virtual_l1_weight = 10.0;
    config.virtual_quadratic_weight = 1.0e-3;
    config.virtual_epigraph_regularisation = 1.0e-3;
    transcription::LowThrustSubproblem subproblem(
        dynamics::LowThrustTwoBodyModel{}, config
    );
    const dynamics::LowThrustState initial{
        7'000.0, 0.0, 0.0, 0.0, 7.546, 0.0, 500.0
    };
    const std::vector<dynamics::LowThrustControl> controls{
        {0.0, 0.0, 0.0, 0.0}, {0.0, 0.0, 0.0, 0.0}
    };
    const auto states = subproblem.model().rollout(initial, controls, config.step_seconds);
    const auto values = subproblem.values(states, controls, initial, states.back());
    const auto& layout = subproblem.layout();
    const auto maps = make_maps(
        subproblem.structure().scalar_constraint,
        config.intervals,
        7U,
        4U,
        layout.dynamics_rows().start,
        [&layout](std::size_t node) { return layout.state(node); },
        [&layout](std::size_t interval) { return layout.control(interval); },
        [&layout](std::size_t interval) { return layout.virtual_control(interval); },
        true
    );
    return run_resident_sequence<7U, 4U>(
        subproblem.structure(),
        values,
        flatten_states(states, config.intervals),
        flatten_controls(controls),
        maps,
        config.intervals,
        layout.dynamics_rows().start,
        model_config(SPACEPDHCG_CUDA_DYNAMICS_LOW_THRUST, config.step_seconds)
    );
}

IntegrationResult run_pd6() {
    transcription::PoweredDescent6DofScvxConfig config;
    config.intervals = 2U;
    config.step_seconds = 0.05;
    config.discretisation = transcription::DiscretisationMethod::rk4_variational;
    config.virtual_l1_weight = 10.0;
    config.virtual_quadratic_weight = 1.0e-3;
    config.virtual_epigraph_regularisation = 1.0e-3;
    transcription::PoweredDescent6DofSubproblem subproblem(
        dynamics::PoweredDescent6DofModel{}, config
    );
    const dynamics::PoweredDescent6DofState initial{
        0.0, 0.0, 100.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 2'000.0,
    };
    const std::vector<dynamics::PoweredDescent6DofControl> controls{
        {0.0, 0.0, 7'422.0, 0.0, 0.0, 0.0, 7'422.0},
        {0.0, 0.0, 7'421.8, 0.0, 0.0, 0.0, 7'421.8},
    };
    const auto states = subproblem.model().rollout(initial, controls, config.step_seconds);
    const auto values = subproblem.values(states, controls, initial, states.back());
    const auto& layout = subproblem.layout();
    const auto maps = make_maps(
        subproblem.structure().scalar_constraint,
        config.intervals,
        14U,
        7U,
        layout.dynamics_rows().start,
        [&layout](std::size_t node) { return layout.state(node); },
        [&layout](std::size_t interval) { return layout.control(interval); },
        [&layout](std::size_t interval) { return layout.virtual_control(interval); },
        true
    );
    return run_resident_sequence<14U, 7U>(
        subproblem.structure(),
        values,
        flatten_states(states, config.intervals),
        flatten_controls(controls),
        maps,
        config.intervals,
        layout.dynamics_rows().start,
        model_config(
            SPACEPDHCG_CUDA_DYNAMICS_POWERED_DESCENT_6DOF, config.step_seconds
        )
    );
}

}  // namespace

int main(const int argc, char** argv) {
    const auto mode = argc > 1 ? std::string_view(argv[1]) : std::string_view{};
    sanitizer_mode = mode == "--sanitizer";
    tight_residual_mode =
        mode == "--tight-pd3" || mode == "--tight-pd3-default-stream"
        || mode == "--tight-pd3-1k" || mode == "--tight-pd3-10k"
        || mode == "--tight-pd3-100k" || mode == "--tight-pd3-300k";
    diagnostic_mode = mode == "--diagnose-pd3";
    dump_mode = mode == "--dump-pd3";
    tight_after_loose_mode =
        mode == "--tight-after-loose-pd3"
        || mode == "--tight-after-loose-refresh-pd3";
    default_stream_mode = mode == "--tight-pd3-default-stream";
    refresh_before_tight_mode = mode == "--tight-after-loose-refresh-pd3";
    repeated_tight_mode = mode == "--tight-twice-pd3";
    if (mode == "--tight-pd3-1k") {
        tight_iteration_limit = 1'000U;
    } else if (mode == "--tight-pd3-10k") {
        tight_iteration_limit = 10'000U;
    } else if (mode == "--tight-pd3-100k") {
        tight_iteration_limit = 100'000U;
    } else if (mode == "--tight-pd3-300k") {
        tight_iteration_limit = 300'000U;
    }
    if (sanitizer_mode) {
        const auto hcw = run_hcw();
        std::printf(
            "{\"case\":\"device_scvx_integration_sanitizer\",\"family\":\"hcw\","
            "\"values_only_updates\":true,\"hidden_cpu_fallback\":false,"
            "\"natural_residual\":%.9g}\n",
            hcw.diagnostics.natural_residual_inf
        );
        return 0;
    }
    if (tight_residual_mode) {
        const auto pd3 = run_pd3();
        std::printf(
            "{\"case\":\"tight_final_residual\",\"termination\":%d,"
            "\"iterations\":%llu,\"requested\":1e-6,"
            "\"relative_primal\":%.9g,\"relative_dual\":%.9g,"
            "\"natural_residual\":%.9g}\n",
            static_cast<int>(pd3.diagnostics.termination),
            static_cast<unsigned long long>(pd3.diagnostics.iterations),
            pd3.diagnostics.relative_primal_residual,
            pd3.diagnostics.relative_dual_residual,
            pd3.diagnostics.natural_residual_inf
        );
        return pd3.diagnostics.natural_residual_inf <= 1.0e-6 ? 0 : 9;
    }
    if (diagnostic_mode) {
        const auto pd3 = run_pd3();
        return pd3.diagnostics.natural_residual_inf <= 1.0e-6 ? 0 : 9;
    }
    if (dump_mode) {
        static_cast<void>(run_pd3());
        return 0;
    }
    if (tight_after_loose_mode) {
        const auto pd3 = run_pd3();
        return pd3.diagnostics.natural_residual_inf <= 1.0e-6 ? 0 : 9;
    }
    if (repeated_tight_mode) {
        const auto pd3 = run_pd3();
        return pd3.diagnostics.natural_residual_inf <= 1.0e-6 ? 0 : 9;
    }
    const auto hcw = run_hcw();
    const auto pd3 = run_pd3();
    const auto low_thrust = run_low_thrust();
    const auto pd6 = run_pd6();
    const auto maximum_residual = std::max({
        hcw.diagnostics.natural_residual_inf,
        pd3.diagnostics.natural_residual_inf,
        low_thrust.diagnostics.natural_residual_inf,
        pd6.diagnostics.natural_residual_inf,
    });
    if (maximum_residual > 1.0e-2) {
        std::fprintf(
            stderr,
            "family residuals: hcw=%.9g pd3=%.9g low_thrust=%.9g pd6=%.9g\n",
            hcw.diagnostics.natural_residual_inf,
            pd3.diagnostics.natural_residual_inf,
            low_thrust.diagnostics.natural_residual_inf,
            pd6.diagnostics.natural_residual_inf
        );
        std::exit(7);
    }
    std::printf(
        "{\"case\":\"device_scvx_integration\",\"families\":4,"
        "\"warm_mode\":\"full_retained\",\"values_only_updates\":true,"
        "\"topology_allocation_delta\":0,\"topology_index_copy_delta\":0,"
        "\"update_allocation_delta\":0,\"hidden_cpu_fallback\":false,"
        "\"maximum_natural_residual\":%.9g}\n",
        maximum_residual
    );
    return 0;
}
