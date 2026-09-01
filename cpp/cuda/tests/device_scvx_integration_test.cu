#include "cuda_test_support.hpp"
#include "spacepdhcg/cuda/device_scvx_c_api.h"
#include "spacepdhcg/cuda/device_scvx_driver_c_api.h"
#include "spacepdhcg/scvx/low_thrust_driver.hpp"
#include "spacepdhcg/scvx/powered_descent_3dof_driver.hpp"
#include "spacepdhcg/transcription/hcw_rendezvous.hpp"
#include "spacepdhcg/transcription/low_thrust.hpp"
#include "spacepdhcg/transcription/powered_descent_3dof.hpp"
#include "spacepdhcg/transcription/powered_descent_6dof.hpp"

#include <pdhcg.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <map>
#include <string>
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
bool tight_all_mode = false;
bool tight_pd6_mode = false;
bool production_driver_mode = false;
bool g4_sample_mode = false;
std::size_t h1_intervals = 0U;
std::size_t g4_intervals = 0U;
std::string g4_family;
std::string g4_policy{"adaptive"};
spacepdhcg_cuda_warm_start_mode g4_warm_start =
    SPACEPDHCG_CUDA_WARM_START_FULL_RETAINED;
double g4_quality_tolerance = 1.0e-6;
double g4_dispersion = 0.0;
std::uint32_t production_outer_iterations = 1U;
double cuda_startup_seconds = 0.0;
int benchmark_variables = 0;
int benchmark_scalar_rows = 0;
int benchmark_affine_rows = 0;
std::size_t benchmark_q_nonzeros = 0U;
std::size_t benchmark_a_nonzeros = 0U;
std::size_t benchmark_f_nonzeros = 0U;
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
    std::vector<int> state_variables;
    std::vector<int> control_variables;
    std::vector<int> virtual_variables;
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
        maps.virtual_variables.reserve(intervals * state_dimension);
    }
    maps.state_variables.reserve((intervals + 1U) * state_dimension);
    maps.control_variables.reserve(intervals * control_dimension);
    for (std::size_t node = 0; node <= intervals; ++node) {
        const auto range = state_range(node);
        for (std::size_t index = 0; index < state_dimension; ++index) {
            maps.state_variables.push_back(
                static_cast<int>(range.start + index)
            );
        }
    }
    for (std::size_t interval = 0; interval < intervals; ++interval) {
        const auto control = control_range(interval);
        for (std::size_t index = 0; index < control_dimension; ++index) {
            maps.control_variables.push_back(
                static_cast<int>(control.start + index)
            );
        }
        if (has_virtual) {
            const auto virtual_control = virtual_range(interval);
            for (std::size_t index = 0; index < state_dimension; ++index) {
                maps.virtual_variables.push_back(
                    static_cast<int>(virtual_control.start + index)
                );
            }
        }
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
    spacepdhcg_cuda_scvx_result outer{};
    std::uint64_t topology_allocations{0U};
    std::uint64_t topology_copies{0U};
    std::uint64_t update_allocations{0U};
    double maximum_solution_magnitude{0.0};
    double coefficient_parity_max{0.0};
    double cpu_gpu_trajectory_max{0.0};
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
    const std::vector<double>& initial_state,
    const std::vector<double>& target_state,
    const DynamicsMaps& maps,
    const std::size_t intervals,
    const std::size_t dynamics_row_start,
    const spacepdhcg_cuda_dynamics_config& dynamics_config,
    const std::vector<double>& state_trust_scales,
    const std::vector<double>& control_trust_scales,
    const double fuel_weight,
    const double virtual_l1_weight
) {
    test::ProblemStorage problem(false, !default_stream_mode);
    const auto topology_started = std::chrono::steady_clock::now();
    materialise(problem, structure, values);
    const double topology_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - topology_started
    ).count();
    benchmark_variables = problem.variables;
    benchmark_scalar_rows = problem.scalar_rows;
    benchmark_affine_rows = problem.affine_rows;
    benchmark_q_nonzeros = problem.h_q.size();
    benchmark_a_nonzeros = problem.h_a.size();
    benchmark_f_nonzeros = problem.h_f.size();
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
    test::CudaBuffer<int> state_variables(maps.state_variables.size(), false);
    test::CudaBuffer<int> control_variables(maps.control_variables.size(), false);
    test::CudaBuffer<int> virtual_variables(maps.virtual_variables.size(), false);
    const auto q_lookup = positions(structure.quadratic);
    std::vector<int> q_diagonal_positions;
    q_diagonal_positions.reserve(
        maps.state_variables.size() + maps.control_variables.size()
    );
    for (const int variable : maps.state_variables) {
        q_diagonal_positions.push_back(q_lookup.at({
            static_cast<std::size_t>(variable),
            static_cast<std::size_t>(variable),
        }));
    }
    for (const int variable : maps.control_variables) {
        q_diagonal_positions.push_back(q_lookup.at({
            static_cast<std::size_t>(variable),
            static_cast<std::size_t>(variable),
        }));
    }
    std::vector<int> radial_positions;
    std::vector<int> quaternion_positions;
    std::size_t radial_row_start{0U};
    std::size_t quaternion_row_start{0U};
    std::size_t stage_trust_row_start{0U};
    std::size_t stage_trust_stride{0U};
    std::size_t terminal_trust_row_start{0U};
    if (dynamics_config.model == SPACEPDHCG_CUDA_DYNAMICS_LOW_THRUST) {
        radial_row_start = 7U + 7U * intervals + 6U + 14U * intervals;
        stage_trust_row_start = 4U * intervals;
        stage_trust_stride = 12U;
        terminal_trust_row_start =
            stage_trust_row_start + stage_trust_stride * intervals;
        const auto scalar_lookup = positions(structure.scalar_constraint);
        for (std::size_t node = 0U; node <= intervals; ++node) {
            for (std::size_t component = 0U; component < 3U; ++component) {
                radial_positions.push_back(scalar_lookup.at({
                    radial_row_start + node,
                    static_cast<std::size_t>(
                        maps.state_variables[node * StateDimension + component]
                    ),
                }));
            }
        }
    } else if (
        dynamics_config.model == SPACEPDHCG_CUDA_DYNAMICS_POWERED_DESCENT_3DOF
    ) {
        stage_trust_row_start = 4U * intervals + 3U * (intervals + 1U);
        stage_trust_stride = 12U;
        terminal_trust_row_start =
            stage_trust_row_start + stage_trust_stride * intervals;
    } else if (
        dynamics_config.model == SPACEPDHCG_CUDA_DYNAMICS_POWERED_DESCENT_6DOF
    ) {
        quaternion_row_start =
            14U + 14U * intervals + 13U + 28U * intervals + intervals;
        stage_trust_row_start =
            8U * intervals + 7U * (intervals + 1U);
        stage_trust_stride = 22U;
        terminal_trust_row_start =
            stage_trust_row_start + stage_trust_stride * intervals;
        const auto scalar_lookup = positions(structure.scalar_constraint);
        for (std::size_t node = 0U; node <= intervals; ++node) {
            for (std::size_t component = 0U; component < 4U; ++component) {
                quaternion_positions.push_back(scalar_lookup.at({
                    quaternion_row_start + node,
                    static_cast<std::size_t>(
                        maps.state_variables[
                            node * StateDimension + 6U + component
                        ]
                    ),
                }));
            }
        }
    }
    test::CudaBuffer<int> q_positions(q_diagonal_positions.size(), false);
    test::CudaBuffer<int> radial_position_buffer(radial_positions.size(), false);
    test::CudaBuffer<int> quaternion_position_buffer(
        quaternion_positions.size(), false
    );
    test::CudaBuffer<double> initial(StateDimension, false);
    test::CudaBuffer<double> target(StateDimension, false);
    states.upload(reference_states, problem.stream);
    controls.upload(reference_controls, problem.stream);
    state_positions.upload(maps.state, problem.stream);
    control_positions.upload(maps.control, problem.stream);
    next_positions.upload(maps.next, problem.stream);
    virtual_positions.upload(maps.virtual_control, problem.stream);
    state_variables.upload(maps.state_variables, problem.stream);
    control_variables.upload(maps.control_variables, problem.stream);
    virtual_variables.upload(maps.virtual_variables, problem.stream);
    q_positions.upload(q_diagonal_positions, problem.stream);
    radial_position_buffer.upload(radial_positions, problem.stream);
    quaternion_position_buffer.upload(quaternion_positions, problem.stream);
    initial.upload(initial_state, problem.stream);
    target.upload(target_state, problem.stream);
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
        test::view(
            states.get(),
            intervals * StateDimension,
            false,
            SPACEPDHCG_SCALAR_FLOAT64,
            SPACEPDHCG_ACCESS_READ_ONLY
        ),
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
    const auto workspace_create_started = std::chrono::steady_clock::now();
    auto* workspace = test::create_workspace(problem);
    const double workspace_create_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - workspace_create_started
    ).count();
    spacepdhcg_cuda_pointer_snapshot pointers_before{};
    test::status_require(
        spacepdhcg_cuda_workspace_pointer_snapshot(workspace, &pointers_before),
        "pointer snapshot before resident sequence"
    );
    spacepdhcg_cuda_diagnostics diagnostics{};
    double coefficient_parity_max{0.0};
    if (production_driver_mode) {
        auto numeric = problem.numeric_views();
        spacepdhcg_cuda_scvx_numeric_update numeric_update{};
        numeric_update.abi_version = SPACEPDHCG_CUDA_WORKSPACE_ABI_VERSION;
        numeric_update.quadratic_diagonal_positions = ro32(q_positions);
        numeric_update.radial_positions = radial_positions.empty()
            ? null_int_view()
            : ro32(radial_position_buffer);
        numeric_update.quaternion_positions = quaternion_positions.empty()
            ? null_int_view()
            : ro32(quaternion_position_buffer);
        numeric_update.terminal_row_start =
            dynamics_row_start + intervals * StateDimension;
        numeric_update.radial_row_start = radial_row_start;
        numeric_update.quaternion_row_start = quaternion_row_start;
        numeric_update.stage_trust_row_start = stage_trust_row_start;
        numeric_update.stage_trust_stride = stage_trust_stride;
        numeric_update.terminal_trust_row_start = terminal_trust_row_start;
        numeric_update.virtual_variable_offset =
            maps.virtual_variables.empty()
            ? 0U
            : static_cast<std::size_t>(maps.virtual_variables.front());
        numeric_update.epigraph_variable_offset =
            numeric_update.virtual_variable_offset
            + maps.virtual_variables.size();
        std::copy(
            state_trust_scales.begin(),
            state_trust_scales.end(),
            numeric_update.state_trust_scales
        );
        std::copy(
            control_trust_scales.begin(),
            control_trust_scales.end(),
            numeric_update.control_trust_scales
        );
        numeric_update.fuel_weight = fuel_weight;
        numeric_update.virtual_l1_weight = virtual_l1_weight;
        const spacepdhcg_cuda_scvx_problem outer_problem{
            SPACEPDHCG_CUDA_WORKSPACE_ABI_VERSION,
            workspace,
            problem.fingerprint,
            intervals,
            StateDimension,
            ControlDimension,
            dynamics_config,
            numeric,
            linearise,
            fill,
            ro32(state_variables),
            ro32(control_variables),
            maps.virtual_variables.empty()
                ? null_int_view()
                : ro32(virtual_variables),
            rw64(states),
            rw64(controls),
            ro64(initial),
            ro64(target),
            numeric_update,
        };
        test::status_require(
            spacepdhcg_cuda_variational_rk4_async(
                &dynamics_config, &linearise, problem.exchange.consumer_stream
            ),
            "coefficient parity variational update"
        );
        test::status_require(
            spacepdhcg_cuda_fill_dynamics_csc_async(
                &fill, problem.exchange.consumer_stream
            ),
            "coefficient parity dynamics fill"
        );
        test::status_require(
            spacepdhcg_cuda_scvx_update_numeric_async(
                &outer_problem,
                1.0,
                virtual_l1_weight,
                problem.exchange.consumer_stream
            ),
            "coefficient parity numeric update"
        );
        test::cuda_require(
            cudaStreamSynchronize(problem.stream),
            "coefficient parity synchronization"
        );
        auto expected = values;
        const auto compare = [&coefficient_parity_max](
            const std::vector<double>& actual,
            const std::vector<double>& wanted
        ) {
            test::require(
                actual.size() == wanted.size(),
                "coefficient parity vector size mismatch"
            );
            for (std::size_t index = 0U; index < actual.size(); ++index) {
                if (std::isinf(actual[index]) || std::isinf(wanted[index])) {
                    test::require(
                        actual[index] == wanted[index],
                        "coefficient parity infinity mismatch"
                    );
                } else {
                    coefficient_parity_max = std::max(
                        coefficient_parity_max,
                        std::abs(actual[index] - wanted[index])
                    );
                }
            }
        };
        compare(problem.q.download(problem.stream), expected.quadratic);
        compare(problem.a.download(problem.stream), expected.scalar_constraint);
        compare(problem.f.download(problem.stream), expected.affine_cone);
        compare(problem.c.download(problem.stream), expected.linear_objective);
        compare(
            problem.scalar_lower.download(problem.stream), expected.scalar_lower
        );
        compare(
            problem.scalar_upper.download(problem.stream), expected.scalar_upper
        );
        compare(
            problem.affine_offset.download(problem.stream), expected.affine_offset
        );
        compare(
            problem.variable_lower.download(problem.stream), expected.variable_lower
        );
        compare(
            problem.variable_upper.download(problem.stream), expected.variable_upper
        );
        if (coefficient_parity_max > 5.0e-12) {
            std::fprintf(
                stderr,
                "coefficient parity maximum %.17g exceeds 5e-12\n",
                coefficient_parity_max
            );
            test::require(
                false,
                "device coefficient update diverged from CPU transcription"
            );
        }
        if (!maps.virtual_variables.empty()) {
            test::status_require(
                spacepdhcg_cuda_scvx_update_numeric_async(
                    &outer_problem,
                    0.37,
                    2.0 * virtual_l1_weight,
                    problem.exchange.consumer_stream
                ),
                "trust and exact-penalty mutation"
            );
            test::cuda_require(
                cudaStreamSynchronize(problem.stream),
                "trust and exact-penalty synchronization"
            );
            const auto changed_c = problem.c.download(problem.stream);
            for (std::size_t variable = numeric_update.epigraph_variable_offset;
                 variable < changed_c.size();
                 ++variable) {
                test::require(
                    std::abs(changed_c[variable] - 2.0 * virtual_l1_weight)
                        <= 1.0e-12,
                    "exact-penalty coefficient failed to update"
                );
            }
            const auto changed_offset =
                problem.affine_offset.download(problem.stream);
            for (std::size_t interval = 0U; interval < intervals; ++interval) {
                test::require(
                    std::abs(
                        changed_offset[
                            numeric_update.stage_trust_row_start
                            + (interval + 1U)
                                * numeric_update.stage_trust_stride
                            - 1U
                        ] - 0.37
                    ) <= 1.0e-12,
                    "stage trust radius failed to update"
                );
            }
            test::require(
                std::abs(
                    changed_offset[
                        numeric_update.terminal_trust_row_start + StateDimension
                    ] - 0.37
                ) <= 1.0e-12,
                "terminal trust radius failed to update"
            );
            test::status_require(
                spacepdhcg_cuda_scvx_update_numeric_async(
                    &outer_problem,
                    1.0,
                    virtual_l1_weight,
                    problem.exchange.consumer_stream
                ),
                "restore initial numeric coefficients"
            );
        }
        spacepdhcg_cuda_scvx_options outer_options{
            SPACEPDHCG_CUDA_WORKSPACE_ABI_VERSION,
            production_outer_iterations,
            production_outer_iterations,
            1U,
            sanitizer_mode ? 1.0e-2 : 1.0e-6,
            2.0e-2,
            0.05,
            0.90,
            100.0,
            100.0,
            1.0,
            1.0e-4,
            8.0,
            0.5,
            1.8,
            sanitizer_mode ? 1.0e-2 : 0.0,
            sanitizer_mode ? 5'000U : 0U,
            SPACEPDHCG_CUDA_SCVX_ADAPTIVE,
            SPACEPDHCG_CUDA_WARM_START_FULL_RETAINED,
            1.0e-3,
            g4_sample_mode ? 1.0e-8 : 1.0e-6,
            1.0e-3,
            0.2,
            0.5,
            0.6,
            1.0e-8,
            1'000'000U,
        };
        if (g4_sample_mode) {
            outer_options.warm_start_mode = g4_warm_start;
            outer_options.convergence_tolerance = g4_quality_tolerance;
            if (g4_policy == "fixed-tight") {
                outer_options.policy = SPACEPDHCG_CUDA_SCVX_FIXED_TIGHT;
            } else if (g4_policy == "fixed-loose") {
                outer_options.policy = SPACEPDHCG_CUDA_SCVX_FIXED_LOOSE;
            } else if (g4_policy == "adaptive+polish") {
                outer_options.policy = SPACEPDHCG_CUDA_SCVX_ADAPTIVE_POLISH;
            } else {
                outer_options.policy = SPACEPDHCG_CUDA_SCVX_ADAPTIVE;
            }
        }
        spacepdhcg_cuda_scvx_driver* driver = nullptr;
        test::status_require(
            spacepdhcg_cuda_scvx_driver_create(
                &outer_problem,
                &outer_options,
                &driver
            ),
            "production outer driver create"
        );
        std::vector<spacepdhcg_cuda_scvx_iteration> records(
            production_outer_iterations
        );
        spacepdhcg_cuda_scvx_result outer{};
        test::status_require(
            spacepdhcg_cuda_scvx_driver_solve(
                driver,
                problem.exchange.consumer_stream,
                records.data(),
                records.size(),
                &outer
            ),
            "production outer driver solve"
        );
        outer.topology_seconds = topology_seconds;
        outer.workspace_create_seconds = workspace_create_seconds;
        const double quality_tolerance = sanitizer_mode ? 1.0e-2 : 1.0e-6;
        if (!g4_sample_mode) {
            test::require(
                outer.status == SPACEPDHCG_CUDA_SCVX_CONVERGED,
                "production outer driver did not converge"
            );
            test::require(
                outer.canonical_residual <= quality_tolerance
                    && outer.dynamics_defect <= quality_tolerance
                    && outer.path_violation <= quality_tolerance
                    && outer.terminal_residual <= quality_tolerance,
                "production outer driver failed final quality"
            );
        }
        test::require(
            outer.topology_allocation_count_after_create == 0U
                && outer.topology_index_copy_count_after_create == 0U
                && outer.hidden_cpu_fallback == 0,
            "production outer driver violated residency"
        );
        const auto final_states = states.download(problem.stream);
        const auto final_controls = controls.download(problem.stream);
        double parity_maximum = 0.0;
        for (std::size_t index = 0; index < final_states.size(); ++index) {
            parity_maximum = std::max(
                parity_maximum,
                std::abs(final_states[index] - reference_states[index])
            );
        }
        for (std::size_t index = 0; index < final_controls.size(); ++index) {
            parity_maximum = std::max(
                parity_maximum,
                std::abs(final_controls[index] - reference_controls[index])
            );
        }
        if (!g4_sample_mode) {
            test::require(
                parity_maximum <= 1.0e-9,
                "production CPU/GPU trajectory parity failed"
            );
        }
        test::status_require(
            spacepdhcg_cuda_workspace_diagnostics(workspace, &diagnostics),
            "production outer diagnostics"
        );
        std::printf(
            "{\"case\":\"production_outer\",\"model\":%d,"
            "\"outer_iterations\":%u,\"accepted\":%u,\"rejected\":%u,"
            "\"trust_radius\":%.9g,\"requested\":%.9g,\"achieved\":%.9g,"
            "\"ratio\":%.9g,\"objective\":%.9g,\"virtual\":%.9g,"
            "\"dynamics\":%.9g,\"path\":%.9g,\"terminal\":%.9g,"
            "\"cpu_gpu_trajectory\":%.9g,"
            "\"t_cqp\":%.9g,\"t_scvx\":%.9g,"
            "\"recovery_seconds\":%.9g,\"recovery_iterations\":%llu,"
            "\"inner_iterations\":%llu,\"d2h_bytes\":%llu,"
            "\"topology_allocations\":%llu,\"topology_copies\":%llu}\n",
            static_cast<int>(dynamics_config.model),
            outer.outer_iterations,
            outer.accepted_steps,
            outer.rejected_steps,
            outer.final_trust_radius,
            records[0].requested_tolerance,
            records[0].achieved_residual,
            records[0].reduction_ratio,
            outer.objective,
            outer.virtual_control,
            outer.dynamics_defect,
            outer.path_violation,
            outer.terminal_residual,
            parity_maximum,
            outer.cqp_total_seconds,
            outer.scvx_total_seconds,
            outer.recovery_seconds,
            static_cast<unsigned long long>(outer.recovery_iterations),
            static_cast<unsigned long long>(outer.inner_iterations),
            static_cast<unsigned long long>(outer.d2h_bytes),
            static_cast<unsigned long long>(
                outer.topology_allocation_count_after_create
            ),
            static_cast<unsigned long long>(
                outer.topology_index_copy_count_after_create
            )
        );
        if (g4_sample_mode) {
            for (std::size_t index = 0U; index < outer.outer_iterations; ++index) {
                const auto& record = records[index];
                std::printf(
                    "{\"case\":\"g4_iteration\",\"family\":\"%s\","
                    "\"policy\":\"%s\",\"intervals\":%zu,\"outer\":%u,"
                    "\"phase\":%d,\"requested\":%.17g,\"achieved\":%.17g,"
                    "\"native_primal\":%.17g,\"native_dual\":%.17g,"
                    "\"complementarity\":%.17g,\"inner_iterations\":%llu,"
                    "\"matvecs\":%llu,\"cone_projections\":%llu,"
                    "\"re_solved\":%d,\"cqp_fingerprint\":\"%016llx\","
                    "\"resolve_fingerprint\":\"%016llx\","
                    "\"resolve_fingerprint_match\":%d,\"trust_action\":%d,"
                    "\"trust_before\":%.17g,\"trust_after\":%.17g,"
                    "\"predicted\":%.17g,\"actual\":%.17g,\"ratio\":%.17g,"
                    "\"candidate_dynamics\":%.17g,\"candidate_path\":%.17g,"
                    "\"candidate_terminal\":%.17g,\"candidate_virtual\":%.17g,"
                    "\"scaling_refreshed\":%d,\"scaling_min\":%.17g,"
                    "\"scaling_max\":%.17g,\"warm_start\":%d,"
                    "\"recovery_mode\":%d,\"forcing_satisfied\":%d,"
                    "\"final_polish_handoff\":%d,\"accepted\":%d}\n",
                    g4_family.c_str(),
                    g4_policy.c_str(),
                    intervals,
                    record.outer_iteration,
                    static_cast<int>(record.phase),
                    record.requested_tolerance,
                    record.achieved_residual,
                    record.native_primal_residual,
                    record.native_dual_residual,
                    record.complementarity_residual,
                    static_cast<unsigned long long>(record.inner_iterations),
                    static_cast<unsigned long long>(record.matvecs),
                    static_cast<unsigned long long>(record.cone_projections),
                    record.re_solved,
                    static_cast<unsigned long long>(
                        record.cqp_numeric_fingerprint
                    ),
                    static_cast<unsigned long long>(
                        record.resolve_numeric_fingerprint
                    ),
                    record.resolve_fingerprint_match,
                    static_cast<int>(record.trust_action),
                    record.trust_radius_before,
                    record.trust_radius_after,
                    record.predicted_reduction,
                    record.actual_reduction,
                    record.reduction_ratio,
                    record.dynamics_defect,
                    record.path_violation,
                    record.terminal_residual,
                    record.virtual_control,
                    record.scaling_refreshed,
                    record.scaling_min,
                    record.scaling_max,
                    static_cast<int>(record.warm_start_mode),
                    static_cast<int>(record.recovery_reason),
                    record.forcing_satisfied,
                    record.final_polish_handoff,
                    record.accepted
                );
            }
            const bool qualified =
                outer.canonical_residual <= g4_quality_tolerance
                && outer.dynamics_defect <= g4_quality_tolerance
                && outer.path_violation <= g4_quality_tolerance
                && outer.terminal_residual <= g4_quality_tolerance
                && outer.virtual_control <= g4_quality_tolerance;
            std::printf(
                "{\"case\":\"g4_sample\",\"family\":\"%s\","
                "\"policy\":\"%s\",\"intervals\":%zu,\"status\":%d,"
                "\"qualified\":%s,\"quality_tolerance\":%.17g,"
                "\"canonical_residual\":%.17g,\"objective\":%.17g,"
                "\"dynamics\":%.17g,\"path\":%.17g,\"terminal\":%.17g,"
                "\"virtual\":%.17g,\"trajectory_difference\":%.17g,"
                "\"cqp_seconds\":%.17g,\"scvx_seconds\":%.17g,"
                "\"recovery_seconds\":%.17g,\"recovery_iterations\":%llu,"
                "\"inner_iterations\":%llu,\"h2d_bytes\":%llu,"
                "\"d2h_bytes\":%llu,\"peak_device_bytes\":%llu,"
                "\"topology_allocations_after_create\":%llu,"
                "\"hidden_cpu_fallback\":%d}\n",
                g4_family.c_str(),
                g4_policy.c_str(),
                intervals,
                static_cast<int>(outer.status),
                qualified ? "true" : "false",
                g4_quality_tolerance,
                outer.canonical_residual,
                outer.objective,
                outer.dynamics_defect,
                outer.path_violation,
                outer.terminal_residual,
                outer.virtual_control,
                parity_maximum,
                outer.cqp_total_seconds,
                outer.scvx_total_seconds,
                outer.recovery_seconds,
                static_cast<unsigned long long>(outer.recovery_iterations),
                static_cast<unsigned long long>(outer.inner_iterations),
                static_cast<unsigned long long>(outer.h2d_bytes),
                static_cast<unsigned long long>(outer.d2h_bytes),
                static_cast<unsigned long long>(diagnostics.peak_active_bytes),
                static_cast<unsigned long long>(
                    outer.topology_allocation_count_after_create
                ),
                outer.hidden_cpu_fallback
            );
        }
        test::status_require(
            spacepdhcg_cuda_scvx_driver_destroy(&driver),
            "production outer driver destroy"
        );
        const IntegrationResult result{
            diagnostics,
            outer,
            outer.topology_allocation_count_after_create,
            outer.topology_index_copy_count_after_create,
            0U,
            0.0,
            coefficient_parity_max,
            parity_maximum,
        };
        test::destroy_workspace(workspace);
        return result;
    }
    const int sequence_iterations =
        sanitizer_mode || tight_residual_mode || tight_all_mode || tight_pd6_mode
            || diagnostic_mode || dump_mode
        ? 1
        : 2;
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
            : ((tight_residual_mode || tight_all_mode || tight_pd6_mode
                || diagnostic_mode || repeated_tight_mode
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
            if (tight_residual_mode || tight_all_mode || tight_pd6_mode
                || diagnostic_mode || dump_mode
                || tight_after_loose_mode || repeated_tight_mode) {
                std::fprintf(
                    stderr,
                    "{\"case\":\"tight_final_residual\",\"termination\":%d,"
                    "\"iterations\":%llu,\"requested\":1e-6,"
                    "\"relative_primal\":%.9g,\"relative_dual\":%.9g,"
                    "\"natural_residual\":%.9g,\"recovery_attempts\":%llu,"
                    "\"recovery_accepted\":%llu,\"recovery_rejected\":%llu,"
                    "\"recovery_outcome\":%d,\"recovery_initial\":%.9g,"
                    "\"recovery_final\":%.9g}\n",
                    static_cast<int>(diagnostics.termination),
                    static_cast<unsigned long long>(diagnostics.iterations),
                    diagnostics.relative_primal_residual,
                    diagnostics.relative_dual_residual,
                    diagnostics.natural_residual_inf,
                    static_cast<unsigned long long>(
                        diagnostics.recovery_attempt_count
                    ),
                    static_cast<unsigned long long>(diagnostics.recovery_count),
                    static_cast<unsigned long long>(
                        diagnostics.recovery_rejected_count
                    ),
                    static_cast<int>(diagnostics.recovery_outcome_reason),
                    diagnostics.recovery_initial_residual,
                    diagnostics.recovery_final_residual
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
    if (tight_residual_mode && tight_iteration_limit >= 350'000U) {
        test::require(
            diagnostics.recovery_count > 0U,
            "tight solve did not record GPU recovery"
        );
        test::require(
            diagnostics.recovery_rejected_count == 0U,
            "tight solve rejected GPU recovery"
        );
    }
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
    if (diagnostic_mode || tight_residual_mode) {
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
        {},
        diagnostics.topology_allocation_delta_last_update,
        diagnostics.topology_index_copy_delta_last_update,
        diagnostics.allocation_delta_last_update,
        maximum_solution,
        0.0,
        0.0,
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
    result.reserve((intervals + 1U) * states.front().size());
    for (std::size_t node = 0; node <= intervals; ++node) {
        result.insert(result.end(), states[node].begin(), states[node].end());
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
    config.intervals = h1_intervals > 0U ? h1_intervals : 2U;
    config.step_seconds = 10.0;
    transcription::HcwRendezvousCqp subproblem(config);
    const dynamics::HcwState initial{};
    const dynamics::HcwState target{};
    const auto values = subproblem.values(initial, target);
    std::vector<dynamics::HcwState> states(config.intervals + 1U, initial);
    std::vector<dynamics::HcwControl> controls(config.intervals);
    const auto& layout = subproblem.layout();
    const auto maps = make_maps(
        subproblem.structure().scalar_constraint,
        config.intervals,
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
        flatten_states(states, config.intervals),
        flatten_controls(controls),
        std::vector<double>(initial.begin(), initial.end()),
        std::vector<double>(target.begin(), target.end()),
        maps,
        config.intervals,
        layout.dynamics_row(),
        model_config(SPACEPDHCG_CUDA_DYNAMICS_HCW, config.step_seconds),
        {},
        {},
        0.0,
        0.0
    );
}

IntegrationResult run_pd3() {
    transcription::PoweredDescentScvxConfig config;
    config.intervals = g4_intervals > 0U ? g4_intervals : 2U;
    config.step_seconds = 0.25;
    config.discretisation = transcription::DiscretisationMethod::rk4_variational;
    config.virtual_l1_weight = 10.0;
    config.virtual_quadratic_weight = 1.0e-3;
    config.virtual_epigraph_regularisation = 1.0e-3;
    transcription::PoweredDescent3DofSubproblem subproblem(
        dynamics::PoweredDescent3DofModel{}, config
    );
    dynamics::PoweredDescentState initial{
        0.0, 0.0, 100.0, 0.0, 0.0, 0.0, 2'000.0
    };
    std::vector<dynamics::PoweredDescentControl> controls(config.intervals);
    for (std::size_t interval = 0U; interval < config.intervals; ++interval) {
        const double thrust = 7'422.0 - 0.5 * static_cast<double>(interval);
        controls[interval] = {0.0, 0.0, thrust, thrust};
    }
    const auto nominal_states =
        subproblem.model().rollout(initial, controls, config.step_seconds);
    if (g4_sample_mode) {
        initial[0] += 10.0 * g4_dispersion;
        initial[2] += 100.0 * g4_dispersion;
        initial[3] -= 5.0 * g4_dispersion;
    }
    const auto states = subproblem.model().rollout(initial, controls, config.step_seconds);
    const std::array<double, 3U> target_position{
        nominal_states.back()[0], nominal_states.back()[1], nominal_states.back()[2]
    };
    const std::array<double, 3U> target_velocity{
        nominal_states.back()[3], nominal_states.back()[4], nominal_states.back()[5]
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
        std::vector<double>(initial.begin(), initial.end()),
        std::vector<double>(nominal_states.back().begin(), nominal_states.back().end()),
        maps,
        config.intervals,
        layout.dynamics_rows().start,
        model_config(
            SPACEPDHCG_CUDA_DYNAMICS_POWERED_DESCENT_3DOF, config.step_seconds
        ),
        std::vector<double>(
            config.state_trust_scales.begin(), config.state_trust_scales.end()
        ),
        std::vector<double>(
            config.control_trust_scales.begin(), config.control_trust_scales.end()
        ),
        config.fuel_weight,
        config.virtual_l1_weight
    );
}

IntegrationResult run_low_thrust() {
    transcription::LowThrustScvxConfig config;
    config.intervals = g4_intervals > 0U ? g4_intervals : 2U;
    config.step_seconds = 1.0;
    config.discretisation = transcription::DiscretisationMethod::rk4_variational;
    config.virtual_l1_weight = 10.0;
    config.virtual_quadratic_weight = 1.0e-3;
    config.virtual_epigraph_regularisation = 1.0e-3;
    transcription::LowThrustSubproblem subproblem(
        dynamics::LowThrustTwoBodyModel{}, config
    );
    dynamics::LowThrustState initial{
        7'000.0, 0.0, 0.0, 0.0, 7.546, 0.0, 500.0
    };
    const std::vector<dynamics::LowThrustControl> controls(
        config.intervals,
        {0.0, 0.0, 0.0, 0.0}
    );
    const auto nominal_states =
        subproblem.model().rollout(initial, controls, config.step_seconds);
    if (g4_sample_mode) {
        initial[0] *= 1.0 + g4_dispersion;
        initial[4] *= 1.0 - 0.5 * g4_dispersion;
    }
    const auto states = subproblem.model().rollout(initial, controls, config.step_seconds);
    const auto values = subproblem.values(
        states, controls, initial, nominal_states.back()
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
        std::vector<double>(initial.begin(), initial.end()),
        std::vector<double>(nominal_states.back().begin(), nominal_states.back().end()),
        maps,
        config.intervals,
        layout.dynamics_rows().start,
        model_config(SPACEPDHCG_CUDA_DYNAMICS_LOW_THRUST, config.step_seconds),
        std::vector<double>(
            config.state_trust_scales.begin(), config.state_trust_scales.end()
        ),
        std::vector<double>(
            config.control_trust_scales.begin(), config.control_trust_scales.end()
        ),
        config.fuel_weight,
        config.virtual_l1_weight
    );
}

IntegrationResult run_pd6() {
    transcription::PoweredDescent6DofScvxConfig config;
    config.intervals = g4_intervals > 0U ? g4_intervals : 2U;
    config.step_seconds = 0.05;
    config.discretisation = transcription::DiscretisationMethod::rk4_variational;
    config.virtual_l1_weight = 10.0;
    config.virtual_quadratic_weight = 1.0e-3;
    config.virtual_epigraph_regularisation = 1.0e-3;
    transcription::PoweredDescent6DofSubproblem subproblem(
        dynamics::PoweredDescent6DofModel{}, config
    );
    dynamics::PoweredDescent6DofState initial{
        0.0, 0.0, 100.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 2'000.0,
    };
    std::vector<dynamics::PoweredDescent6DofControl> controls(config.intervals);
    for (std::size_t interval = 0U; interval < config.intervals; ++interval) {
        const double thrust = 7'422.0 - 0.2 * static_cast<double>(interval);
        controls[interval] = {
            0.0, 0.0, thrust, 0.0, 0.0, 0.0, thrust
        };
    }
    const auto nominal_states =
        subproblem.model().rollout(initial, controls, config.step_seconds);
    if (g4_sample_mode) {
        initial[0] += 10.0 * g4_dispersion;
        initial[2] += 100.0 * g4_dispersion;
        initial[7] += g4_dispersion;
        initial[10] += 0.1 * g4_dispersion;
    }
    const auto states = subproblem.model().rollout(initial, controls, config.step_seconds);
    const auto values = subproblem.values(
        states, controls, initial, nominal_states.back()
    );
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
        std::vector<double>(initial.begin(), initial.end()),
        std::vector<double>(nominal_states.back().begin(), nominal_states.back().end()),
        maps,
        config.intervals,
        layout.dynamics_rows().start,
        model_config(
            SPACEPDHCG_CUDA_DYNAMICS_POWERED_DESCENT_6DOF, config.step_seconds
        ),
        std::vector<double>(
            config.state_trust_scales.begin(), config.state_trust_scales.end()
        ),
        std::vector<double>(
            config.control_trust_scales.begin(), config.control_trust_scales.end()
        ),
        config.fuel_weight,
        config.virtual_l1_weight
    );
}

}  // namespace

int main(const int argc, char** argv) {
    const auto mode = argc > 1 ? std::string_view(argv[1]) : std::string_view{};
    sanitizer_mode =
        mode == "--sanitizer"
        || mode == "--production-outer-sanitizer";
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
    tight_all_mode = mode == "--tight-all";
    tight_pd6_mode = mode == "--tight-pd6";
    production_driver_mode =
        mode == "--production-outer"
        || mode == "--production-outer-sanitizer"
        || mode == "--h1-hcw"
        || mode == "--g4-sample";
    if (mode == "--g4-sample") {
        test::require(
            argc == 9,
            "G4 mode requires family intervals policy warm quality outer-iterations dispersion"
        );
        g4_sample_mode = true;
        g4_family = argv[2];
        g4_intervals = std::stoull(argv[3]);
        g4_policy = argv[4];
        const std::string_view warm = argv[5];
        if (warm == "cold") {
            g4_warm_start = SPACEPDHCG_CUDA_WARM_START_NONE;
        } else if (warm == "primal") {
            g4_warm_start = SPACEPDHCG_CUDA_WARM_START_PRIMAL;
        } else if (warm == "primal_dual") {
            g4_warm_start = SPACEPDHCG_CUDA_WARM_START_PRIMAL_DUAL;
        } else {
            test::require(false, "unknown G4 warm-start mode");
        }
        g4_quality_tolerance = std::stod(argv[6]);
        production_outer_iterations =
            static_cast<std::uint32_t>(std::stoul(argv[7]));
        g4_dispersion = std::stod(argv[8]);
    }
    if (mode == "--h1-hcw") {
        test::require(argc == 4, "H1 mode requires intervals and repeats");
        h1_intervals = std::stoull(argv[2]);
        production_outer_iterations =
            static_cast<std::uint32_t>(std::stoul(argv[3]));
        const auto startup_begin = std::chrono::steady_clock::now();
        test::cuda_require(cudaFree(nullptr), "CUDA startup");
        cuda_startup_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - startup_begin
        ).count();
    }
    if (mode == "--tight-pd3-1k") {
        tight_iteration_limit = 1'000U;
    } else if (mode == "--tight-pd3-10k") {
        tight_iteration_limit = 10'000U;
    } else if (mode == "--tight-pd3-100k") {
        tight_iteration_limit = 100'000U;
    } else if (mode == "--tight-pd3-300k") {
        tight_iteration_limit = 300'000U;
    }
    if (mode == "--sanitizer") {
        const auto hcw = run_hcw();
        std::printf(
            "{\"case\":\"device_scvx_integration_sanitizer\",\"family\":\"hcw\","
            "\"values_only_updates\":true,\"hidden_cpu_fallback\":false,"
            "\"natural_residual\":%.9g}\n",
            hcw.diagnostics.natural_residual_inf
        );
        return 0;
    }
    if (production_driver_mode) {
        if (g4_sample_mode) {
            if (g4_family == "P1-C-pd3") {
                static_cast<void>(run_pd3());
            } else if (g4_family == "P1-D-pd6") {
                static_cast<void>(run_pd6());
            } else if (g4_family == "P1-E-low-thrust") {
                static_cast<void>(run_low_thrust());
            } else {
                test::require(false, "unknown G4 family");
            }
            return 0;
        }
        const auto hcw = run_hcw();
        if (mode == "--h1-hcw") {
            const auto& timing = hcw.outer;
            std::printf(
                "{\"case\":\"h1_hcw\",\"intervals\":%zu,\"repeats\":%u,"
                "\"variables\":%d,\"scalar_rows\":%d,\"affine_rows\":%d,"
                "\"q_nonzeros\":%zu,\"a_nonzeros\":%zu,\"f_nonzeros\":%zu,"
                "\"cuda_startup_seconds\":%.9g,\"topology_seconds\":%.9g,"
                "\"coefficient_seconds\":%.9g,"
                "\"workspace_create_seconds\":%.9g,\"update_seconds\":%.9g,"
                "\"scaling_seconds\":%.9g,\"h2d_seconds\":%.9g,"
                "\"solve_seconds\":%.9g,\"recovery_seconds\":%.9g,"
                "\"residual_seconds\":%.9g,\"replay_seconds\":%.9g,"
                "\"acceptance_seconds\":%.9g,\"d2h_seconds\":%.9g,"
                "\"cqp_total_seconds\":%.9g,\"scvx_total_seconds\":%.9g,"
                "\"allocation_count\":%llu,\"allocation_bytes\":%llu,"
                "\"h2d_copy_count\":%llu,\"h2d_bytes\":%llu,"
                "\"d2h_copy_count\":%llu,\"d2h_bytes\":%llu,"
                "\"device_copy_count\":%llu,\"device_copy_bytes\":%llu,"
                "\"topology_allocations_after_create\":%llu,"
                "\"topology_copies_after_create\":%llu,"
                "\"recovery_iterations\":%llu,"
                "\"canonical_residual\":%.9g,\"nonlinear_residual\":%.9g,"
                "\"cpu_gpu_trajectory\":%.9g,"
                "\"omega_persist\":0.0,"
                "\"modes\":[\"cold\",\"first-persistent\",\"warm\","
                "\"repeated\",\"primal\",\"primal-dual\",\"full-state\"]}\n",
                h1_intervals,
                production_outer_iterations,
                benchmark_variables,
                benchmark_scalar_rows,
                benchmark_affine_rows,
                benchmark_q_nonzeros,
                benchmark_a_nonzeros,
                benchmark_f_nonzeros,
                cuda_startup_seconds,
                timing.topology_seconds,
                timing.coefficient_seconds,
                timing.workspace_create_seconds,
                timing.update_seconds,
                timing.scaling_seconds,
                timing.h2d_seconds,
                timing.solve_seconds,
                timing.recovery_seconds,
                timing.residual_seconds,
                timing.replay_seconds,
                timing.acceptance_seconds,
                timing.d2h_seconds,
                timing.cqp_total_seconds,
                timing.scvx_total_seconds,
                static_cast<unsigned long long>(timing.allocation_count),
                static_cast<unsigned long long>(timing.allocation_bytes),
                static_cast<unsigned long long>(timing.h2d_copy_count),
                static_cast<unsigned long long>(timing.h2d_bytes),
                static_cast<unsigned long long>(timing.d2h_copy_count),
                static_cast<unsigned long long>(timing.d2h_bytes),
                static_cast<unsigned long long>(timing.device_copy_count),
                static_cast<unsigned long long>(timing.device_copy_bytes),
                static_cast<unsigned long long>(
                    timing.topology_allocation_count_after_create
                ),
                static_cast<unsigned long long>(
                    timing.topology_index_copy_count_after_create
                ),
                static_cast<unsigned long long>(timing.recovery_iterations),
                timing.canonical_residual,
                std::max({
                    timing.dynamics_defect,
                    timing.path_violation,
                    timing.terminal_residual,
                }),
                hcw.cpu_gpu_trajectory_max
            );
            return 0;
        }
        const auto pd3 = run_pd3();
        const auto low_thrust = run_low_thrust();
        const auto pd6 = run_pd6();
        const double maximum_canonical = std::max({
            hcw.outer.canonical_residual,
            pd3.outer.canonical_residual,
            low_thrust.outer.canonical_residual,
            pd6.outer.canonical_residual,
        });
        const double maximum_nonlinear = std::max({
            hcw.outer.dynamics_defect,
            hcw.outer.path_violation,
            hcw.outer.terminal_residual,
            pd3.outer.dynamics_defect,
            pd3.outer.path_violation,
            pd3.outer.terminal_residual,
            low_thrust.outer.dynamics_defect,
            low_thrust.outer.path_violation,
            low_thrust.outer.terminal_residual,
            pd6.outer.dynamics_defect,
            pd6.outer.path_violation,
            pd6.outer.terminal_residual,
        });
        const double maximum_trajectory_difference = std::max({
            hcw.cpu_gpu_trajectory_max,
            pd3.cpu_gpu_trajectory_max,
            low_thrust.cpu_gpu_trajectory_max,
            pd6.cpu_gpu_trajectory_max,
        });
        std::printf(
            "{\"case\":\"production_outer_all\",\"families\":4,"
            "\"maximum_canonical\":%.9g,\"maximum_nonlinear\":%.9g,"
            "\"maximum_trajectory_difference\":%.9g,"
            "\"hidden_cpu_fallback\":false,"
            "\"topology_allocations_after_create\":0,"
            "\"topology_copies_after_create\":0}\n",
            maximum_canonical,
            maximum_nonlinear,
            maximum_trajectory_difference
        );
        const double quality_tolerance =
            mode == "--production-outer-sanitizer" ? 1.0e-2 : 1.0e-6;
        return maximum_canonical <= quality_tolerance
                && maximum_nonlinear <= quality_tolerance
            ? 0
            : 11;
    }
    if (tight_residual_mode) {
        const auto pd3 = run_pd3();
        std::printf(
            "{\"case\":\"tight_final_residual\",\"termination\":%d,"
            "\"iterations\":%llu,\"requested\":1e-6,"
            "\"relative_primal\":%.9g,\"relative_dual\":%.9g,"
            "\"recovery_count\":%llu,\"recovery_iterations\":%llu,"
            "\"natural_residual\":%.9g}\n",
            static_cast<int>(pd3.diagnostics.termination),
            static_cast<unsigned long long>(pd3.diagnostics.iterations),
            pd3.diagnostics.relative_primal_residual,
            pd3.diagnostics.relative_dual_residual,
            static_cast<unsigned long long>(pd3.diagnostics.recovery_count),
            static_cast<unsigned long long>(pd3.diagnostics.recovery_iterations),
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
    if (tight_all_mode) {
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
        std::printf(
            "{\"case\":\"tight_all\",\"hcw\":%.9g,\"pd3\":%.9g,"
            "\"low_thrust\":%.9g,\"pd6\":%.9g}\n",
            hcw.diagnostics.natural_residual_inf,
            pd3.diagnostics.natural_residual_inf,
            low_thrust.diagnostics.natural_residual_inf,
            pd6.diagnostics.natural_residual_inf
        );
        return maximum_residual <= 1.0e-6 ? 0 : 10;
    }
    if (tight_pd6_mode) {
        const auto pd6 = run_pd6();
        std::printf(
            "{\"case\":\"tight_pd6\",\"natural\":%.9g,\"stationarity\":%.9g,"
            "\"recovery_attempts\":%llu,\"recovery_accepted\":%llu,"
            "\"recovery_rejected\":%llu,\"recovery_outcome\":%d,"
            "\"recovery_initial\":%.9g,\"recovery_final\":%.9g,"
            "\"recovery_primal\":%.9g,\"recovery_stationarity\":%.9g,"
            "\"recovery_complementarity\":%.9g,"
            "\"recovery_stationarity_index\":%d,"
            "\"recovery_stationarity_value\":%.9g}\n",
            pd6.diagnostics.natural_residual_inf,
            pd6.diagnostics.stationarity_inf,
            static_cast<unsigned long long>(
                pd6.diagnostics.recovery_attempt_count
            ),
            static_cast<unsigned long long>(pd6.diagnostics.recovery_count),
            static_cast<unsigned long long>(
                pd6.diagnostics.recovery_rejected_count
            ),
            static_cast<int>(pd6.diagnostics.recovery_outcome_reason),
            pd6.diagnostics.recovery_initial_residual,
            pd6.diagnostics.recovery_final_residual,
            pd6.diagnostics.recovery_final_primal_residual,
            pd6.diagnostics.recovery_final_stationarity,
            pd6.diagnostics.recovery_final_complementarity,
            pd6.diagnostics.recovery_stationarity_index,
            pd6.diagnostics.recovery_stationarity_value
        );
        return pd6.diagnostics.natural_residual_inf <= 1.0e-6 ? 0 : 10;
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
