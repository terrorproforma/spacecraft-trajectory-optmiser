#include "cuda_test_support.hpp"
#include "spacepdhcg/cuda/device_scvx_c_api.h"
#include "spacepdhcg/cuda/device_scvx_driver_c_api.h"
#include "spacepdhcg/scvx/g4_policy.generated.hpp"
#include "spacepdhcg/scvx/low_thrust_benchmark.hpp"
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
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

namespace test = spacepdhcg::cuda::test;
namespace core = spacepdhcg::core;
namespace dynamics = spacepdhcg::dynamics;
namespace transcription = spacepdhcg::transcription;
namespace frozen_g4 = spacepdhcg::scvx::g4_policy;

namespace {

thread_local bool sanitizer_mode = false;
thread_local bool tight_residual_mode = false;
thread_local bool diagnostic_mode = false;
thread_local bool dump_mode = false;
thread_local bool tight_after_loose_mode = false;
thread_local bool default_stream_mode = false;
thread_local bool refresh_before_tight_mode = false;
thread_local bool repeated_tight_mode = false;
thread_local bool tight_all_mode = false;
thread_local bool tight_pd6_mode = false;
thread_local bool production_driver_mode = false;
thread_local bool g4_sample_mode = false;
thread_local bool g4_probe_mode = false;
thread_local bool g4_diagnostic_mode = false;
thread_local bool p1d_path_audit_mode = false;
thread_local bool p1d_diagnostic_mode = false;
thread_local bool qoco_handback_mode = false;
thread_local bool qoco_unavailable_mode = false;
thread_local bool p1c_qoco_repeatability_mode = false;
thread_local std::size_t h1_intervals = 0U;
thread_local std::size_t g4_intervals = 0U;
thread_local std::string g4_family;
thread_local std::string g4_policy{"adaptive"};
thread_local std::string g4_quality_tier{"tight"};
thread_local std::string g4_scaling_mode{"refresh_if_needed"};
thread_local std::string g4_warm_mode{"primal_dual"};
thread_local spacepdhcg_cuda_warm_start_mode g4_warm_start =
    SPACEPDHCG_CUDA_WARM_START_FULL_RETAINED;
thread_local double g4_quality_tolerance = 1.0e-6;
thread_local double g4_family_class = 0.0;
thread_local std::string g4_transfer_class{"not_applicable"};
thread_local double g4_dispersion = 0.0;
thread_local double g4_secondary_dispersion = 0.0;
thread_local double g4_conditioning_log10_span = 0.0;
thread_local std::uint64_t g4_evaluation_seed = 0U;
thread_local std::string g4_repeat_kind{"not_applicable"};
thread_local std::uint32_t g4_repeat_index = 0U;
thread_local std::uint32_t g4_solver_order = 0U;
thread_local std::string g4_coordinate_id;
thread_local std::string g4_matrix_sha256;
thread_local std::string g4_capability_sha256;
thread_local std::uint64_t g4_instance_hash = 0U;
thread_local std::uint64_t g4_problem_hash = 0U;
thread_local std::uint64_t g4_coefficient_hash = 0U;
thread_local double g4_condition_factor_min = 1.0;
thread_local double g4_condition_factor_max = 1.0;
thread_local double g4_conditioned_coefficient_ratio = 1.0;
thread_local std::uint32_t production_outer_iterations = 1U;
thread_local double cuda_startup_seconds = 0.0;
thread_local double g4_deadline_seconds = 0.0;
thread_local std::chrono::steady_clock::time_point g4_deadline{};
thread_local bool g4_server_request = false;
thread_local std::size_t g4_lane_index = 0U;
thread_local int benchmark_variables = 0;
thread_local int benchmark_scalar_rows = 0;
thread_local int benchmark_affine_rows = 0;
thread_local std::size_t benchmark_q_nonzeros = 0U;
thread_local std::size_t benchmark_a_nonzeros = 0U;
thread_local std::size_t benchmark_f_nonzeros = 0U;
thread_local std::uint64_t tight_iteration_limit = 1'000'000U;
constexpr std::array<std::uint64_t, 20U> g4_evaluation_seeds{
    59U, 71U, 89U, 101U, 127U, 149U, 173U, 197U, 223U, 251U,
    281U, 313U, 349U, 389U, 431U, 479U, 521U, 569U, 617U, 659U,
};

std::uint64_t hash_bytes(
    std::uint64_t hash,
    const void* data,
    const std::size_t size
) {
    const auto* bytes = static_cast<const unsigned char*>(data);
    for (std::size_t index = 0U; index < size; ++index) {
        hash ^= bytes[index];
        hash *= 1'099'511'628'211ULL;
    }
    return hash;
}

template <typename Value>
std::uint64_t hash_value(std::uint64_t hash, const Value& value) {
    return hash_bytes(hash, &value, sizeof(value));
}

template <typename Value>
std::uint64_t hash_vector(std::uint64_t hash, const std::vector<Value>& values) {
    return hash_bytes(hash, values.data(), values.size() * sizeof(Value));
}

std::uint64_t splitmix64(std::uint64_t value) {
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31U);
}

double seeded_signed(const std::uint64_t stream) {
    const auto bits = splitmix64(g4_evaluation_seed ^ stream) >> 11U;
    return 2.0 * static_cast<double>(bits)
        / static_cast<double>(1ULL << 53U) - 1.0;
}

double row_condition_factor(
    const std::size_t row,
    const std::size_t rows
) {
    if (g4_conditioning_log10_span == 0.0 || rows <= 1U) {
        return 1.0;
    }
    const double fraction =
        static_cast<double>(row) / static_cast<double>(rows - 1U);
    return std::pow(
        10.0,
        g4_conditioning_log10_span * (fraction - 0.5)
    );
}

core::NumericValues condition_values(
    const core::FixedStructure& structure,
    const core::NumericValues& source,
    const std::size_t dynamics_row_start,
    const std::size_t dynamics_rows
) {
    auto result = source;
    if (!g4_sample_mode || g4_conditioning_log10_span == 0.0) {
        return result;
    }
    const auto& scalar = structure.scalar_constraint;
    for (std::size_t column = 0U;
         column < static_cast<std::size_t>(scalar.columns);
         ++column) {
        for (int position = scalar.offsets[column];
             position < scalar.offsets[column + 1U];
             ++position) {
            const auto row = static_cast<std::size_t>(scalar.indices[position]);
            if (row >= dynamics_row_start
                && row < dynamics_row_start + dynamics_rows) {
                result.scalar_constraint[position] *= row_condition_factor(
                    row - dynamics_row_start,
                    dynamics_rows
                );
            }
        }
    }
    for (std::size_t row = 0U; row < dynamics_rows; ++row) {
        const double factor = row_condition_factor(row, dynamics_rows);
        result.scalar_lower[dynamics_row_start + row] *= factor;
        result.scalar_upper[dynamics_row_start + row] *= factor;
    }
    return result;
}

std::uint64_t numeric_hash(const core::NumericValues& values) {
    std::uint64_t hash = 1'469'598'103'934'665'603ULL;
    hash = hash_vector(hash, values.quadratic);
    hash = hash_vector(hash, values.scalar_constraint);
    hash = hash_vector(hash, values.affine_cone);
    hash = hash_vector(hash, values.linear_objective);
    hash = hash_vector(hash, values.scalar_lower);
    hash = hash_vector(hash, values.scalar_upper);
    hash = hash_vector(hash, values.affine_offset);
    hash = hash_vector(hash, values.variable_lower);
    return hash_vector(hash, values.variable_upper);
}

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

struct G4WorkspaceCacheEntry {
    std::string key;
    std::unique_ptr<test::ProblemStorage> problem;
    spacepdhcg_cuda_workspace* workspace{nullptr};

    ~G4WorkspaceCacheEntry() {
        if (workspace != nullptr) {
            static_cast<void>(spacepdhcg_cuda_workspace_destroy(&workspace));
        }
    }
};

std::mutex g4_workspace_cache_mutex;
std::array<std::shared_ptr<G4WorkspaceCacheEntry>, 1024U> g4_workspace_cache{};

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
    const double virtual_l1_weight,
    const core::NumericValues* displaced_values = nullptr,
    const std::vector<double>* displaced_states = nullptr,
    const std::vector<double>* displaced_controls = nullptr,
    const std::vector<double>* displaced_initial = nullptr,
    const std::vector<double>* displaced_target = nullptr
) {
    const auto topology_started = std::chrono::steady_clock::now();
    const auto conditioned_values = condition_values(
        structure,
        values,
        dynamics_row_start,
        intervals * StateDimension
    );
    std::shared_ptr<G4WorkspaceCacheEntry> cached_entry;
    bool workspace_cache_hit = false;
    std::unique_ptr<test::ProblemStorage> ephemeral_problem;
    if (g4_sample_mode && g4_server_request) {
        const std::string cache_key =
            std::to_string(structure.fingerprint()) + ":" + g4_scaling_mode;
        std::lock_guard lock(g4_workspace_cache_mutex);
        auto& slot = g4_workspace_cache.at(g4_lane_index);
        if (slot == nullptr || slot->key != cache_key) {
            slot = std::make_shared<G4WorkspaceCacheEntry>();
            slot->key = cache_key;
            slot->problem = std::make_unique<test::ProblemStorage>(
                false, !default_stream_mode
            );
            materialise(*slot->problem, structure, conditioned_values);
        } else {
            workspace_cache_hit = true;
        }
        cached_entry = slot;
    } else {
        ephemeral_problem = std::make_unique<test::ProblemStorage>(
            false, !default_stream_mode
        );
        materialise(*ephemeral_problem, structure, conditioned_values);
    }
    auto& problem = cached_entry != nullptr
        ? *cached_entry->problem
        : *ephemeral_problem;
    if (workspace_cache_hit) {
        problem.h_q = conditioned_values.quadratic;
        problem.h_a = conditioned_values.scalar_constraint;
        problem.h_f = conditioned_values.affine_cone;
        problem.h_c = conditioned_values.linear_objective;
        problem.h_scalar_lower = conditioned_values.scalar_lower;
        problem.h_scalar_upper = conditioned_values.scalar_upper;
        problem.h_affine_offset = conditioned_values.affine_offset;
        problem.h_variable_lower = conditioned_values.variable_lower;
        problem.h_variable_upper = conditioned_values.variable_upper;
        problem.upload_numeric();
        problem.primal.upload(
            std::vector<double>(static_cast<std::size_t>(problem.variables), 0.0),
            problem.stream
        );
        problem.dual.upload(
            std::vector<double>(
                static_cast<std::size_t>(problem.scalar_rows + problem.affine_rows),
                0.0
            ),
            problem.stream
        );
        test::cuda_require(
            cudaStreamSynchronize(problem.stream),
            "cached G4 problem reset"
        );
    }
    g4_coefficient_hash = numeric_hash(conditioned_values);
    g4_problem_hash = hash_value(
        hash_value(g4_instance_hash, structure.fingerprint()),
        g4_coefficient_hash
    );
    g4_condition_factor_min = row_condition_factor(0U, intervals * StateDimension);
    g4_condition_factor_max = row_condition_factor(
        intervals * StateDimension - 1U,
        intervals * StateDimension
    );
    double coefficient_min = std::numeric_limits<double>::infinity();
    double coefficient_max = 0.0;
    for (const double coefficient : conditioned_values.scalar_constraint) {
        const double magnitude = std::abs(coefficient);
        if (magnitude > 0.0 && std::isfinite(magnitude)) {
            coefficient_min = std::min(coefficient_min, magnitude);
            coefficient_max = std::max(coefficient_max, magnitude);
        }
    }
    g4_conditioned_coefficient_ratio =
        coefficient_min < std::numeric_limits<double>::infinity()
        ? coefficient_max / coefficient_min
        : 1.0;
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
    auto create_options = test::create_options();
    if (g4_sample_mode) {
        if (g4_scaling_mode == "always_refresh") {
            create_options.scaling_mode = SPACEPDHCG_CUDA_SCALING_ALWAYS_REFRESH;
        } else if (g4_scaling_mode == "reuse") {
            create_options.scaling_mode = SPACEPDHCG_CUDA_SCALING_REUSE;
        } else {
            create_options.scaling_mode =
                SPACEPDHCG_CUDA_SCALING_REFRESH_IF_NEEDED;
        }
    }
    auto* workspace = cached_entry != nullptr ? cached_entry->workspace : nullptr;
    if (workspace == nullptr) {
        workspace = test::create_workspace(problem, create_options);
        if (cached_entry != nullptr) {
            cached_entry->workspace = workspace;
        }
    } else {
        test::status_require(
            spacepdhcg_cuda_workspace_reset_async(
                workspace,
                SPACEPDHCG_CUDA_RESET_FULL,
                problem.exchange.consumer_stream
            ),
            "cached G4 workspace reset"
        );
        test::status_require(
            spacepdhcg_cuda_workspace_wait(workspace),
            "cached G4 workspace reset wait"
        );
    }
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
    double coefficient_parity_relative{0.0};
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
        numeric_update.conditioning_log10_span =
            g4_sample_mode ? g4_conditioning_log10_span : 0.0;
        if (dynamics_config.model
            == SPACEPDHCG_CUDA_DYNAMICS_POWERED_DESCENT_3DOF) {
            const dynamics::PoweredDescent3DofModel physical_model{};
            const auto& physical = physical_model.config();
            numeric_update.maximum_thrust = physical.maximum_thrust;
            numeric_update.glide_slope_tangent =
                physical.glide_slope_tangent();
        } else if (dynamics_config.model
                   == SPACEPDHCG_CUDA_DYNAMICS_LOW_THRUST) {
            const dynamics::LowThrustTwoBodyModel physical_model{};
            const auto& physical = physical_model.config();
            numeric_update.maximum_thrust = physical.maximum_thrust;
            numeric_update.minimum_radius = physical.minimum_radius;
        } else if (dynamics_config.model
                   == SPACEPDHCG_CUDA_DYNAMICS_POWERED_DESCENT_6DOF) {
            const dynamics::PoweredDescent6DofModel physical_model{};
            const auto& physical =
                physical_model.config();
            numeric_update.maximum_thrust = physical.maximum_thrust;
            numeric_update.maximum_torque = physical.maximum_torque;
            numeric_update.maximum_angular_rate =
                physical.maximum_angular_rate;
            numeric_update.tilt_cosine = physical.tilt_cosine();
            numeric_update.glide_slope_tangent =
                physical.glide_slope_tangent();
        }
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
            problem.structure,
            problem.exchange.topology,
        };
        const double initial_numeric_trust =
            g4_sample_mode
                    && dynamics_config.model == SPACEPDHCG_CUDA_DYNAMICS_LOW_THRUST
                ? g4_family_class
                : 1.0;
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
                initial_numeric_trust,
                virtual_l1_weight,
                problem.exchange.consumer_stream
            ),
            "coefficient parity numeric update"
        );
        test::cuda_require(
            cudaStreamSynchronize(problem.stream),
            "coefficient parity synchronization"
        );
        auto expected = conditioned_values;
        const auto compare = [
            &coefficient_parity_max,
            &coefficient_parity_relative
        ](
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
                    const double difference =
                        std::abs(actual[index] - wanted[index]);
                    coefficient_parity_max = std::max(
                        coefficient_parity_max, difference
                    );
                    coefficient_parity_relative = std::max(
                        coefficient_parity_relative,
                        difference / std::max({
                            1.0,
                            std::abs(wanted[index]),
                            g4_condition_factor_max,
                        })
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
        if (displaced_values != nullptr
            || displaced_states != nullptr
            || displaced_controls != nullptr
            || displaced_initial != nullptr
            || displaced_target != nullptr) {
            test::require(
                displaced_values != nullptr
                    && displaced_states != nullptr
                    && displaced_controls != nullptr
                    && displaced_initial != nullptr
                    && displaced_target != nullptr,
                "displaced coefficient audit requires complete inputs"
            );
            states.upload(*displaced_states, problem.stream);
            controls.upload(*displaced_controls, problem.stream);
            initial.upload(*displaced_initial, problem.stream);
            target.upload(*displaced_target, problem.stream);
            test::status_require(
                spacepdhcg_cuda_variational_rk4_async(
                    &dynamics_config, &linearise, problem.exchange.consumer_stream
                ),
                "displaced coefficient variational update"
            );
            test::status_require(
                spacepdhcg_cuda_fill_dynamics_csc_async(
                    &fill, problem.exchange.consumer_stream
                ),
                "displaced coefficient dynamics fill"
            );
            test::status_require(
                spacepdhcg_cuda_scvx_update_numeric_async(
                    &outer_problem,
                    0.37,
                    virtual_l1_weight,
                    problem.exchange.consumer_stream
                ),
                "displaced full numeric update"
            );
            test::cuda_require(
                cudaStreamSynchronize(problem.stream),
                "displaced coefficient synchronization"
            );
            const auto conditioned_displaced = condition_values(
                structure,
                *displaced_values,
                dynamics_row_start,
                intervals * StateDimension
            );
            compare(problem.q.download(problem.stream), conditioned_displaced.quadratic);
            compare(
                problem.a.download(problem.stream),
                conditioned_displaced.scalar_constraint
            );
            compare(problem.f.download(problem.stream), conditioned_displaced.affine_cone);
            compare(
                problem.c.download(problem.stream),
                conditioned_displaced.linear_objective
            );
            compare(
                problem.scalar_lower.download(problem.stream),
                conditioned_displaced.scalar_lower
            );
            compare(
                problem.scalar_upper.download(problem.stream),
                conditioned_displaced.scalar_upper
            );
            compare(
                problem.affine_offset.download(problem.stream),
                conditioned_displaced.affine_offset
            );
            compare(
                problem.variable_lower.download(problem.stream),
                conditioned_displaced.variable_lower
            );
            compare(
                problem.variable_upper.download(problem.stream),
                conditioned_displaced.variable_upper
            );
            states.upload(reference_states, problem.stream);
            controls.upload(reference_controls, problem.stream);
            initial.upload(initial_state, problem.stream);
            target.upload(target_state, problem.stream);
            test::status_require(
                spacepdhcg_cuda_variational_rk4_async(
                    &dynamics_config, &linearise, problem.exchange.consumer_stream
                ),
                "restore reference coefficient variational update"
            );
            test::status_require(
                spacepdhcg_cuda_fill_dynamics_csc_async(
                    &fill, problem.exchange.consumer_stream
                ),
                "restore reference coefficient dynamics fill"
            );
            test::status_require(
                spacepdhcg_cuda_scvx_update_numeric_async(
                    &outer_problem,
                    initial_numeric_trust,
                    virtual_l1_weight,
                    problem.exchange.consumer_stream
                ),
                "restore reference full numeric update"
            );
        }
        if (coefficient_parity_relative > 5.0e-12) {
            std::fprintf(
                stderr,
                "relative coefficient parity maximum %.17g exceeds 5e-12\n",
                coefficient_parity_relative
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
                    initial_numeric_trust,
                    virtual_l1_weight,
                    problem.exchange.consumer_stream
                ),
                "restore initial numeric coefficients"
            );
        }
        if (dump_mode) {
            run_upstream_diagnostic(problem);
            test::destroy_workspace(workspace);
            return {};
        }
        if (p1d_diagnostic_mode) {
            auto updated = problem.numeric_views();
            test::status_require(
                spacepdhcg_cuda_workspace_update_async(
                    workspace,
                    problem.fingerprint,
                    &updated,
                    problem.exchange.consumer_stream
                ),
                "P1-D diagnostic numeric update"
            );
            const auto diagnostic = test::solve_and_wait(
                workspace,
                problem,
                test::solve_options(1.0e-6, 300'000U)
            );
            std::printf(
                "{\"case\":\"p1d_recovery_diagnostic\","
                "\"termination\":%d,\"iterations\":%llu,"
                "\"natural\":%.17g,\"stationarity\":%.17g,"
                "\"scalar\":%.17g,\"box\":%.17g,\"cone\":%.17g,"
                "\"complementarity\":%.17g}\n",
                static_cast<int>(diagnostic.termination),
                static_cast<unsigned long long>(diagnostic.iterations),
                diagnostic.natural_residual_inf,
                diagnostic.stationarity_inf,
                diagnostic.scalar_primal_violation_inf,
                diagnostic.box_violation_inf,
                diagnostic.affine_cone_distance_inf,
                diagnostic.complementarity_inf
            );
            print_diagnostic_vector(
                "persistent_primal",
                problem.primal.download(problem.stream)
            );
            print_diagnostic_vector(
                "persistent_dual",
                problem.dual.download(problem.stream)
            );
            test::destroy_workspace(workspace);
            return {};
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
            virtual_l1_weight,
            1.0,
            1.0e-4,
            8.0,
            0.5,
            1.8,
            frozen_g4::trust_strong_agreement,
            frozen_g4::trust_boundary_fraction,
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
            frozen_g4::repair_ceiling,
            frozen_g4::progress_ceiling,
            frozen_g4::refinement_ceiling,
            frozen_g4::polish_ceiling,
            frozen_g4::repair_iterations,
            frozen_g4::progress_iterations,
            frozen_g4::refinement_iterations,
            frozen_g4::polish_iterations,
            frozen_g4::resolve_trigger_multiple,
            frozen_g4::resolve_refinement_factor,
            frozen_g4::resolve_minimum_tolerance,
            1.0e-8,
            1'000'000U,
        };
        if (g4_sample_mode) {
            if (dynamics_config.model == SPACEPDHCG_CUDA_DYNAMICS_LOW_THRUST) {
                outer_options.initial_trust_radius = g4_family_class;
            }
            outer_options.warm_start_mode = g4_warm_start;
            outer_options.convergence_tolerance = g4_quality_tolerance;
            if (g4_policy == "fixed-tight") {
                outer_options.policy = SPACEPDHCG_CUDA_SCVX_FIXED_TIGHT;
                outer_options.fixed_inner_tolerance = g4_quality_tolerance;
                outer_options.fixed_inner_iteration_limit =
                    frozen_g4::polish_iterations;
            } else if (g4_policy == "fixed-loose") {
                outer_options.policy = SPACEPDHCG_CUDA_SCVX_FIXED_LOOSE;
                outer_options.fixed_inner_tolerance =
                    frozen_g4::epsilon_max;
                outer_options.fixed_inner_iteration_limit =
                    frozen_g4::progress_iterations;
            } else if (g4_policy == "adaptive+polish") {
                outer_options.policy = SPACEPDHCG_CUDA_SCVX_ADAPTIVE_POLISH;
            } else if (g4_policy == "pure-gpu-ipm") {
                outer_options.policy = SPACEPDHCG_CUDA_SCVX_PURE_QOCO;
                outer_options.fixed_inner_tolerance = 1.0e-8;
                outer_options.fixed_inner_iteration_limit = 200U;
            } else if (g4_policy == "hybrid-pdhcg-ipm") {
                outer_options.policy = SPACEPDHCG_CUDA_SCVX_HYBRID_QOCO;
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
        if (g4_probe_mode) {
            std::printf(
                "{\"case\":\"g4_axis_probe\",\"coordinate_id\":\"%s\","
                "\"policy_sha256\":\"%.*s\",\"matrix_sha256\":\"%s\","
                "\"capability_sha256\":\"%s\",\"family\":\"%s\","
                "\"intervals\":%zu,\"policy\":\"%s\",\"policy_code\":%d,"
                "\"quality_tier\":\"%s\",\"quality_tolerance\":%.17g,"
                "\"conditioning_log10_span\":%.17g,"
                "\"scaling_mode\":\"%s\",\"scaling_code\":%d,"
                "\"warm_start_mode\":\"%s\",\"warm_start_code\":%d,"
                "\"dispersion_class\":%.17g,\"attitude_class\":%.17g,"
                "\"rate_class\":%.17g,\"trust_class\":%.17g,"
                "\"transfer_class\":\"%s\",\"evaluation_seed\":%llu,"
                "\"instance\":\"%s-seed-%llu\",\"repeat_kind\":\"%s\","
                "\"repeat\":%u,\"solver_order\":%u,"
                "\"instance_hash\":\"%016llx\",\"problem_hash\":\"%016llx\","
                "\"coefficient_hash\":\"%016llx\","
                "\"condition_factor_min\":%.17g,"
                "\"condition_factor_max\":%.17g,"
                "\"conditioned_coefficient_ratio\":%.17g,"
                "\"coefficient_parity_maximum\":%.17g,"
                "\"coefficient_parity_relative\":%.17g}\n",
                g4_coordinate_id.c_str(),
                static_cast<int>(frozen_g4::sha256.size()),
                frozen_g4::sha256.data(),
                g4_matrix_sha256.c_str(),
                g4_capability_sha256.c_str(),
                g4_family.c_str(),
                intervals,
                g4_policy.c_str(),
                static_cast<int>(outer_options.policy),
                g4_quality_tier.c_str(),
                g4_quality_tolerance,
                g4_conditioning_log10_span,
                g4_scaling_mode.c_str(),
                static_cast<int>(create_options.scaling_mode),
                g4_warm_mode.c_str(),
                static_cast<int>(outer_options.warm_start_mode),
                g4_family == "P1-C-pd3" ? g4_family_class : 0.0,
                g4_family == "P1-D-pd6" ? g4_dispersion : 0.0,
                g4_family == "P1-D-pd6" ? g4_secondary_dispersion : 0.0,
                g4_family == "P1-E-low-thrust" ? g4_family_class : 0.0,
                g4_transfer_class.c_str(),
                static_cast<unsigned long long>(g4_evaluation_seed),
                g4_family.c_str(),
                static_cast<unsigned long long>(g4_evaluation_seed),
                g4_repeat_kind.c_str(),
                g4_repeat_index,
                g4_solver_order,
                static_cast<unsigned long long>(g4_instance_hash),
                static_cast<unsigned long long>(g4_problem_hash),
                static_cast<unsigned long long>(g4_coefficient_hash),
                g4_condition_factor_min,
                g4_condition_factor_max,
                g4_conditioned_coefficient_ratio,
                coefficient_parity_max,
                coefficient_parity_relative
            );
            test::status_require(
                spacepdhcg_cuda_scvx_driver_destroy(&driver),
                "G4 axis-probe driver destroy"
            );
            test::destroy_workspace(workspace);
            return {};
        }
        std::vector<spacepdhcg_cuda_scvx_iteration> records(
            production_outer_iterations
        );
        spacepdhcg_cuda_scvx_result outer{};
        std::mutex deadline_mutex;
        std::condition_variable deadline_condition;
        bool solve_finished = false;
        std::thread deadline_thread;
        if (g4_deadline_seconds > 0.0) {
            deadline_thread = std::thread([&]() {
                std::unique_lock lock(deadline_mutex);
                if (!deadline_condition.wait_until(
                        lock,
                        g4_deadline,
                        [&]() { return solve_finished; }
                    )) {
                    static_cast<void>(
                        spacepdhcg_cuda_scvx_driver_cancel(driver)
                    );
                }
            });
        }
        const auto outer_status = spacepdhcg_cuda_scvx_driver_solve(
            driver,
            problem.exchange.consumer_stream,
            records.data(),
            records.size(),
            &outer
        );
        if (deadline_thread.joinable()) {
            {
                std::lock_guard lock(deadline_mutex);
                solve_finished = true;
            }
            deadline_condition.notify_one();
            deadline_thread.join();
        }
        if (qoco_unavailable_mode) {
            test::require(
                outer_status == SPACEPDHCG_CUDA_UNSUPPORTED,
                "missing native QOCO must propagate unsupported"
            );
            test::require(
                outer.status == SPACEPDHCG_CUDA_SCVX_INNER_FAILURE
                    && outer.qoco_failure
                        == SPACEPDHCG_CUDA_QOCO_FAILURE_UNAVAILABLE
                    && outer.hidden_cpu_fallback == 0,
                "missing native QOCO must classify failure without fallback"
            );
            test::status_require(
                spacepdhcg_cuda_scvx_driver_destroy(&driver),
                "unavailable QOCO driver destroy"
            );
            test::destroy_workspace(workspace);
            return {};
        }
        test::status_require(outer_status, "production outer driver solve");
        if (g4_policy == "pure-gpu-ipm") {
            test::require(
                outer.qoco_workspace_creations == 1U,
                "pure QOCO must reuse one native workspace"
            );
            test::require(
                outer.qoco_numeric_updates + 1U == outer.outer_iterations,
                "pure QOCO numeric updates must match persistent outer reuse"
            );
            test::require(
                outer.qoco_failure == SPACEPDHCG_CUDA_QOCO_FAILURE_NONE,
                "pure QOCO reported a native CUDA/cuDSS failure"
            );
            test::require(
                outer.hidden_cpu_fallback == 0,
                "pure QOCO must not use a CPU fallback"
            );
            test::require(
                outer.qoco_conversion_seconds >= 0.0
                    && outer.qoco_setup_seconds >= 0.0
                    && outer.qoco_update_seconds >= 0.0
                    && outer.qoco_solve_seconds > 0.0,
                "pure QOCO timing accounting is incomplete"
            );
        }
        spacepdhcg_cuda_scvx_path_inventory path_inventory{};
        test::status_require(
            spacepdhcg_cuda_scvx_driver_path_inventory(
                driver, &path_inventory
            ),
            "production outer path inventory"
        );
        test::require(
            path_inventory.abi_version == SPACEPDHCG_CUDA_WORKSPACE_ABI_VERSION,
            "production outer path inventory ABI mismatch"
        );
        if (qoco_handback_mode) {
            const auto canonical_primal = problem.primal.download(problem.stream);
            const auto canonical_dual = problem.dual.download(problem.stream);
            const auto& final_iteration =
                records[outer.outer_iterations - 1U];
            const double outer_residual = std::max({
                outer.dynamics_defect,
                outer.path_violation,
                outer.terminal_residual,
                outer.virtual_control,
            });
            const spacepdhcg_cuda_qoco_candidate candidate{
                SPACEPDHCG_CUDA_WORKSPACE_ABI_VERSION,
                SPACEPDHCG_CUDA_QOCO_PURE_GPU_IPM,
                canonical_primal.data(),
                canonical_primal.size(),
                canonical_dual.data(),
                canonical_dual.size(),
                problem.fingerprint,
                final_iteration.cqp_numeric_fingerprint,
                outer.canonical_residual,
                outer.canonical_residual,
                std::max(1.0e-8, 1.01 * outer.canonical_residual),
                outer.final_trust_radius,
                outer.objective
                    + 100.0
                        * (outer.path_violation + outer.terminal_residual),
                outer_residual,
                1,
                1,
                0,
                1.0e-4,
                2.0e-4,
                3.0e-4,
            };
            spacepdhcg_cuda_qoco_handback_result handback{};
            test::status_require(
                spacepdhcg_cuda_scvx_driver_handback_qoco(
                    driver,
                    &candidate,
                    problem.exchange.consumer_stream,
                    &handback
                ),
                "QOCO candidate nonlinear handback"
            );
            test::require(
                handback.device_replay == 1
                    && handback.hidden_cpu_fallback == 0,
                "QOCO handback did not use the native device replay"
            );
            test::require(
                handback.fingerprint_match == 1
                    && handback.permutation_match == 1,
                "QOCO handback fingerprint/permutation mismatch"
            );
            test::require(
                handback.disposition
                        == SPACEPDHCG_CUDA_QOCO_HANDBACK_ACCEPTED
                    || handback.disposition
                        == SPACEPDHCG_CUDA_QOCO_HANDBACK_NONLINEAR_REJECTED,
                "QOCO handback did not make an outer acceptance decision"
            );
            test::require(
                std::isfinite(handback.quaternion_violation)
                    && std::isfinite(handback.terminal_residual)
                    && std::isfinite(handback.reduction_ratio),
                "QOCO handback omitted P1-D nonlinear quality"
            );
            auto ineligible = candidate;
            ineligible.mode =
                SPACEPDHCG_CUDA_QOCO_HYBRID_PDHCG_IPM;
            ineligible.hybrid_handoff_eligible = 0;
            spacepdhcg_cuda_qoco_handback_result hybrid{};
            test::status_require(
                spacepdhcg_cuda_scvx_driver_handback_qoco(
                    driver,
                    &ineligible,
                    problem.exchange.consumer_stream,
                    &hybrid
                ),
                "ineligible hybrid handback"
            );
            test::require(
                hybrid.disposition
                        == SPACEPDHCG_CUDA_QOCO_HANDBACK_HYBRID_INELIGIBLE
                    && hybrid.device_replay == 0,
                "ineligible hybrid reached QOCO nonlinear replay"
            );
            std::printf(
                "{\"case\":\"qoco_handback\",\"family\":\"P1-D-pd6\","
                "\"mode\":\"pure-gpu-ipm\",\"accepted\":%d,"
                "\"fingerprint_match\":%d,\"permutation_match\":%d,"
                "\"device_replay\":%d,\"hidden_cpu_fallback\":%d,"
                "\"quaternion\":%.17g,\"terminal\":%.17g,"
                "\"predicted\":%.17g,\"actual\":%.17g,\"ratio\":%.17g}\n",
                handback.accepted,
                handback.fingerprint_match,
                handback.permutation_match,
                handback.device_replay,
                handback.hidden_cpu_fallback,
                handback.quaternion_violation,
                handback.terminal_residual,
                handback.predicted_reduction,
                handback.actual_reduction,
                handback.reduction_ratio
            );
        }
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
        double retained_change_maximum = 0.0;
        for (std::size_t index = 0; index < final_states.size(); ++index) {
            retained_change_maximum = std::max(
                retained_change_maximum,
                std::abs(final_states[index] - reference_states[index])
            );
        }
        for (std::size_t index = 0; index < final_controls.size(); ++index) {
            retained_change_maximum = std::max(
                retained_change_maximum,
                std::abs(final_controls[index] - reference_controls[index])
            );
        }
        double replay_parity_maximum = 0.0;
        double independent_path = outer.path_violation;
        double independent_terminal = outer.terminal_residual;
        std::array<double, 8U> independent_path_inventory{};
        if constexpr (StateDimension == 14U && ControlDimension == 7U) {
            dynamics::PoweredDescent6DofState cpu_initial{};
            std::copy(
                initial_state.begin(),
                initial_state.end(),
                cpu_initial.begin()
            );
            std::vector<dynamics::PoweredDescent6DofControl> cpu_controls(
                intervals
            );
            for (std::size_t interval = 0U; interval < intervals; ++interval) {
                std::copy_n(
                    final_controls.begin()
                        + static_cast<std::ptrdiff_t>(
                            interval * ControlDimension
                        ),
                    ControlDimension,
                    cpu_controls[interval].begin()
                );
            }
            const dynamics::PoweredDescent6DofModel cpu_model{};
            const auto cpu_replay = cpu_model.rollout(
                cpu_initial,
                cpu_controls,
                dynamics_config.step_seconds
            );
            for (std::size_t node = 0U; node <= intervals; ++node) {
                for (std::size_t component = 0U;
                     component < StateDimension;
                     ++component) {
                    replay_parity_maximum = std::max(
                        replay_parity_maximum,
                        std::abs(
                            cpu_replay[node][component]
                            - final_states[
                                node * StateDimension + component
                            ]
                        )
                    );
                }
            }
            const auto path = cpu_model.path_diagnostics(
                cpu_replay,
                cpu_controls
            );
            const auto& physical = cpu_model.config();
            const double position_scale = std::max({
                state_trust_scales[0U],
                state_trust_scales[1U],
                state_trust_scales[2U],
            });
            const double angular_rate_scale = std::max({
                state_trust_scales[10U],
                state_trust_scales[11U],
                state_trust_scales[12U],
            });
            independent_path_inventory = {
                std::max({
                    path.thrust_epigraph,
                    path.throttle_lower,
                    path.throttle_upper,
                }) / physical.maximum_thrust,
                path.torque / physical.maximum_torque,
                path.pointing / physical.maximum_thrust,
                path.minimum_mass * state_trust_scales[13U],
                path.altitude * position_scale,
                path.glide_slope * position_scale,
                path.angular_rate * angular_rate_scale,
                path.quaternion_norm_error,
            };
            independent_path = *std::max_element(
                independent_path_inventory.begin(),
                independent_path_inventory.end()
            );
            independent_terminal = 0.0;
            for (std::size_t component = 0U; component < 13U; ++component) {
                independent_terminal = std::max(
                    independent_terminal,
                    std::abs(
                        cpu_replay.back()[component] - target_state[component]
                    ) * state_trust_scales[component]
                );
            }
            test::status_require(
                spacepdhcg_cuda_variational_rk4_async(
                    &dynamics_config,
                    &linearise,
                    problem.exchange.consumer_stream
                ),
                "final P1-D variational coefficient update"
            );
            test::status_require(
                spacepdhcg_cuda_fill_dynamics_csc_async(
                    &fill,
                    problem.exchange.consumer_stream
                ),
                "final P1-D dynamics coefficient fill"
            );
            test::status_require(
                spacepdhcg_cuda_scvx_update_numeric_async(
                    &outer_problem,
                    outer.final_trust_radius,
                    virtual_l1_weight,
                    problem.exchange.consumer_stream
                ),
                "final P1-D numeric coefficient update"
            );
            test::cuda_require(
                cudaStreamSynchronize(problem.stream),
                "final P1-D coefficient synchronization"
            );
            std::vector<dynamics::PoweredDescent6DofState> cpu_reference(
                intervals + 1U
            );
            for (std::size_t node = 0U; node <= intervals; ++node) {
                std::copy_n(
                    final_states.begin()
                        + static_cast<std::ptrdiff_t>(node * StateDimension),
                    StateDimension,
                    cpu_reference[node].begin()
                );
            }
            transcription::PoweredDescent6DofScvxConfig cpu_config;
            cpu_config.intervals = intervals;
            cpu_config.step_seconds = dynamics_config.step_seconds;
            cpu_config.discretisation =
                transcription::DiscretisationMethod::rk4_variational;
            cpu_config.virtual_l1_weight = virtual_l1_weight;
            cpu_config.virtual_quadratic_weight = 1.0e-3;
            cpu_config.virtual_epigraph_regularisation = 1.0e-3;
            cpu_config.fuel_weight = fuel_weight;
            std::copy(
                state_trust_scales.begin(),
                state_trust_scales.end(),
                cpu_config.state_trust_scales.begin()
            );
            std::copy(
                control_trust_scales.begin(),
                control_trust_scales.end(),
                cpu_config.control_trust_scales.begin()
            );
            const transcription::PoweredDescent6DofSubproblem cpu_subproblem(
                cpu_model,
                cpu_config
            );
            dynamics::PoweredDescent6DofState cpu_target{};
            std::copy(target_state.begin(), target_state.end(), cpu_target.begin());
            expected = cpu_subproblem.values(
                cpu_reference,
                cpu_controls,
                cpu_initial,
                cpu_target,
                outer.final_trust_radius
            );
            compare(problem.q.download(problem.stream), expected.quadratic);
            compare(problem.a.download(problem.stream), expected.scalar_constraint);
            compare(problem.f.download(problem.stream), expected.affine_cone);
            compare(problem.c.download(problem.stream), expected.linear_objective);
            compare(
                problem.scalar_lower.download(problem.stream),
                expected.scalar_lower
            );
            compare(
                problem.scalar_upper.download(problem.stream),
                expected.scalar_upper
            );
            compare(
                problem.affine_offset.download(problem.stream),
                expected.affine_offset
            );
            compare(
                problem.variable_lower.download(problem.stream),
                expected.variable_lower
            );
            compare(
                problem.variable_upper.download(problem.stream),
                expected.variable_upper
            );
            test::require(
                coefficient_parity_relative <= 5.0e-12,
                "final P1-D CPU/device coefficients diverged"
            );
            if (p1d_path_audit_mode) {
                test::require(
                    independent_path_inventory[2U] > 5.0e-2,
                    "injected P1-D pointing violation was not detected"
                );
                test::require(
                    std::abs(outer.path_violation - independent_path)
                        <= 5.0e-12,
                    "CUDA and independent P1-D path checks diverged"
                );
                test::require(
                    outer.outer_iterations == 2U
                        && records[1U].warm_start_mode
                            == SPACEPDHCG_CUDA_WARM_START_PRIMAL_DUAL,
                    "P1-D multi-iteration warm state was not retained"
                );
            }
        }
        if (!g4_sample_mode) {
            test::require(
                replay_parity_maximum <= 1.0e-9,
                "production CPU/GPU trajectory parity failed"
            );
        }
        test::status_require(
            spacepdhcg_cuda_workspace_diagnostics(workspace, &diagnostics),
            "production outer diagnostics"
        );
        if (g4_diagnostic_mode) {
            dump_mode = true;
            run_upstream_diagnostic(problem);
            print_diagnostic_vector(
                "persistent_primal",
                problem.primal.download(problem.stream)
            );
            print_diagnostic_vector(
                "persistent_dual",
                problem.dual.download(problem.stream)
            );
            print_diagnostic_vector("retained_states", final_states);
            print_diagnostic_vector("retained_controls", final_controls);
            std::printf(
                "{\"case\":\"g4_recovery_diagnostic\","
                "\"attempts\":%llu,\"accepted\":%llu,\"rejected\":%llu,"
                "\"trigger\":%d,\"outcome\":%d,\"seconds\":%.17g,"
                "\"iterations\":%llu,\"initial\":%.17g,\"final\":%.17g,"
                "\"primal\":%.17g,\"stationarity\":%.17g,"
                "\"complementarity\":%.17g,\"stationarity_index\":%d,"
                "\"stationarity_value\":%.17g,"
                "\"scalar\":%.17g,\"box\":%.17g,\"cone\":%.17g,"
                "\"natural\":%.17g}\n",
                static_cast<unsigned long long>(
                    diagnostics.recovery_attempt_count
                ),
                static_cast<unsigned long long>(diagnostics.recovery_count),
                static_cast<unsigned long long>(
                    diagnostics.recovery_rejected_count
                ),
                static_cast<int>(diagnostics.recovery_trigger_reason),
                static_cast<int>(diagnostics.recovery_outcome_reason),
                diagnostics.recovery_seconds,
                static_cast<unsigned long long>(
                    diagnostics.recovery_iterations
                ),
                diagnostics.recovery_initial_residual,
                diagnostics.recovery_final_residual,
                diagnostics.recovery_final_primal_residual,
                diagnostics.recovery_final_stationarity,
                diagnostics.recovery_final_complementarity,
                diagnostics.recovery_stationarity_index,
                diagnostics.recovery_stationarity_value,
                diagnostics.scalar_primal_violation_inf,
                diagnostics.box_violation_inf,
                diagnostics.affine_cone_distance_inf,
                diagnostics.natural_residual_inf
            );
        }
        std::printf(
            "{\"case\":\"production_outer\",\"model\":%d,"
            "\"outer_iterations\":%u,\"accepted\":%u,\"rejected\":%u,"
            "\"trust_radius\":%.9g,\"requested\":%.9g,\"achieved\":%.9g,"
            "\"ratio\":%.9g,\"objective\":%.9g,\"virtual\":%.9g,"
            "\"dynamics\":%.9g,\"path\":%.9g,\"terminal\":%.9g,"
            "\"path_thrust\":%.9g,\"path_mass\":%.9g,"
            "\"path_altitude\":%.9g,"
            "\"cpu_gpu_trajectory\":%.9g,"
            "\"cpu_gpu_replay\":%.9g,\"retained_change\":%.9g,"
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
            independent_path,
            independent_terminal,
            path_inventory.thrust_violation,
            path_inventory.mass_violation,
            path_inventory.altitude_violation,
            replay_parity_maximum,
            replay_parity_maximum,
            retained_change_maximum,
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
            const double primary_coordinate =
                g4_family == "P1-D-pd6" ? g4_dispersion : g4_family_class;
            std::printf(
                "{\"case\":\"g4_coordinate\",\"coordinate_id\":\"%s\","
                "\"family\":\"%s\","
                "\"primary_dispersion\":%.17g,"
                "\"secondary_dispersion\":%.17g,"
                "\"attitude_dispersion_radians\":%.17g,"
                "\"angular_rate_dispersion\":%.17g}\n",
                g4_coordinate_id.c_str(),
                g4_family.c_str(),
                primary_coordinate,
                g4_family == "P1-D-pd6" ? g4_secondary_dispersion : 0.0,
                g4_family == "P1-D-pd6" ? g4_dispersion : 0.0,
                g4_family == "P1-D-pd6" ? g4_secondary_dispersion : 0.0
            );
            std::printf(
                "{\"case\":\"g4_runtime\",\"coordinate_id\":\"%s\","
                "\"policy_sha256\":\"%.*s\","
                "\"requested\":{\"policy\":\"%s\",\"quality_tier\":\"%s\","
                "\"quality_tolerance\":%.17g,\"scaling_mode\":\"%s\","
                "\"warm_start_mode\":\"%s\"},"
                "\"actual\":{\"policy\":\"%s\",\"quality_tier\":\"%s\","
                "\"quality_tolerance\":%.17g,\"scaling_mode\":\"%s\","
                "\"warm_start_mode\":\"%s\","
                "\"resolve_trigger_multiple\":%.17g,"
                "\"resolve_refinement_factor\":%.17g,"
                "\"resolve_minimum_tolerance\":%.17g,"
                "\"maximum_resolves\":%u,"
                "\"polish_tolerance_ceiling\":%.17g}}\n",
                g4_coordinate_id.c_str(),
                static_cast<int>(frozen_g4::sha256.size()),
                frozen_g4::sha256.data(),
                g4_policy.c_str(),
                g4_quality_tier.c_str(),
                g4_quality_tolerance,
                g4_scaling_mode.c_str(),
                g4_warm_mode.c_str(),
                g4_policy.c_str(),
                g4_quality_tier.c_str(),
                g4_quality_tolerance,
                g4_scaling_mode.c_str(),
                g4_warm_mode.c_str(),
                outer_options.resolve_trigger_multiple,
                outer_options.resolve_refinement_factor,
                outer_options.resolve_minimum_tolerance,
                outer_options.maximum_resolves_per_iteration,
                outer_options.polish_tolerance_ceiling
            );
            std::printf(
                "{\"case\":\"g4_axis_application\","
                "\"coordinate_id\":\"%s\",\"policy_sha256\":\"%.*s\","
                "\"matrix_sha256\":\"%s\",\"capability_sha256\":\"%s\","
                "\"family\":\"%s\",\"intervals\":%zu,\"policy\":\"%s\","
                "\"quality_tier\":\"%s\",\"quality_tolerance\":%.17g,"
                "\"conditioning_log10_span\":%.17g,"
                "\"scaling_mode\":\"%s\",\"warm_start_mode\":\"%s\","
                "\"dispersion_class\":%.17g,\"attitude_class\":%.17g,"
                "\"rate_class\":%.17g,\"trust_class\":%.17g,"
                "\"transfer_class\":\"%s\",\"evaluation_seed\":%llu,"
                "\"instance\":\"%s-seed-%llu\",\"repeat_kind\":\"%s\","
                "\"repeat\":%u,\"solver_order\":%u,"
                "\"instance_hash\":\"%016llx\",\"problem_hash\":\"%016llx\","
                "\"coefficient_hash\":\"%016llx\","
                "\"condition_factor_min\":%.17g,"
                "\"condition_factor_max\":%.17g,"
                "\"conditioned_coefficient_ratio\":%.17g,"
                "\"coefficient_parity_maximum\":%.17g,"
                "\"coefficient_parity_relative\":%.17g}\n",
                g4_coordinate_id.c_str(),
                static_cast<int>(frozen_g4::sha256.size()),
                frozen_g4::sha256.data(),
                g4_matrix_sha256.c_str(),
                g4_capability_sha256.c_str(),
                g4_family.c_str(),
                intervals,
                g4_policy.c_str(),
                g4_quality_tier.c_str(),
                g4_quality_tolerance,
                g4_conditioning_log10_span,
                g4_scaling_mode.c_str(),
                g4_warm_mode.c_str(),
                g4_family == "P1-C-pd3" ? g4_family_class : 0.0,
                g4_family == "P1-D-pd6" ? g4_dispersion : 0.0,
                g4_family == "P1-D-pd6" ? g4_secondary_dispersion : 0.0,
                g4_family == "P1-E-low-thrust" ? g4_family_class : 0.0,
                g4_transfer_class.c_str(),
                static_cast<unsigned long long>(g4_evaluation_seed),
                g4_family.c_str(),
                static_cast<unsigned long long>(g4_evaluation_seed),
                g4_repeat_kind.c_str(),
                g4_repeat_index,
                g4_solver_order,
                static_cast<unsigned long long>(g4_instance_hash),
                static_cast<unsigned long long>(g4_problem_hash),
                static_cast<unsigned long long>(g4_coefficient_hash),
                g4_condition_factor_min,
                g4_condition_factor_max,
                g4_conditioned_coefficient_ratio,
                coefficient_parity_max,
                coefficient_parity_relative
            );
            if constexpr (StateDimension == 14U && ControlDimension == 7U) {
                std::printf(
                    "{\"case\":\"g4_path_inventory\","
                    "\"family\":\"P1-D-pd6\",\"independent\":true,"
                    "\"thrust\":%.17g,\"torque\":%.17g,"
                    "\"pointing\":%.17g,\"mass\":%.17g,"
                    "\"altitude\":%.17g,\"glide_slope\":%.17g,"
                    "\"angular_rate\":%.17g,\"quaternion\":%.17g,"
                    "\"complete\":true,\"cpu_gpu_replay\":%.17g}\n",
                    independent_path_inventory[0U],
                    independent_path_inventory[1U],
                    independent_path_inventory[2U],
                    independent_path_inventory[3U],
                    independent_path_inventory[4U],
                    independent_path_inventory[5U],
                    independent_path_inventory[6U],
                    independent_path_inventory[7U],
                    replay_parity_maximum
                );
            }
            for (std::size_t index = 0U; index < outer.outer_iterations; ++index) {
                const auto& record = records[index];
                std::printf(
                    "{\"case\":\"g4_iteration\",\"coordinate_id\":\"%s\","
                    "\"family\":\"%s\","
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
                    "\"step_fraction\":%.17g,"
                    "\"maximum_stage_trust_distance\":%.17g,"
                    "\"terminal_trust_distance\":%.17g,"
                    "\"candidate_dynamics\":%.17g,\"candidate_path\":%.17g,"
                    "\"candidate_terminal\":%.17g,\"candidate_virtual\":%.17g,"
                    "\"current_merit\":%.17g,\"candidate_merit\":%.17g,"
                    "\"candidate_model_merit\":%.17g,"
                    "\"current_dynamics\":%.17g,\"current_path\":%.17g,"
                    "\"current_terminal\":%.17g,"
                    "\"scalar_primal\":%.17g,\"box_primal\":%.17g,"
                    "\"cone_primal\":%.17g,\"stationarity\":%.17g,"
                    "\"natural\":%.17g,"
                    "\"recovery_attempts\":%llu,"
                    "\"recovery_accepted\":%llu,\"recovery_rejected\":%llu,"
                    "\"recovery_seconds\":%.17g,\"recovery_iterations\":%llu,"
                    "\"recovery_initial\":%.17g,\"recovery_final\":%.17g,"
                    "\"recovery_primal\":%.17g,"
                    "\"recovery_stationarity\":%.17g,"
                    "\"recovery_complementarity\":%.17g,"
                    "\"scaling_refreshed\":%d,\"scaling_min\":%.17g,"
                    "\"scaling_max\":%.17g,\"warm_start\":%d,"
                    "\"recovery_mode\":%d,\"forcing_satisfied\":%d,"
                    "\"final_polish_handoff\":%d,\"accepted\":%d}\n",
                    g4_coordinate_id.c_str(),
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
                    record.step_fraction,
                    record.maximum_stage_trust_distance,
                    record.terminal_trust_distance,
                    record.dynamics_defect,
                    record.path_violation,
                    record.terminal_residual,
                    record.virtual_control,
                    record.current_merit,
                    record.candidate_merit,
                    record.candidate_model_merit,
                    record.current_dynamics_defect,
                    record.current_path_violation,
                    record.current_terminal_residual,
                    record.scalar_primal_residual,
                    record.box_primal_residual,
                    record.cone_primal_residual,
                    record.stationarity_residual,
                    record.natural_residual,
                    static_cast<unsigned long long>(
                        record.recovery_attempt_count
                    ),
                    static_cast<unsigned long long>(
                        record.recovery_accepted_count
                    ),
                    static_cast<unsigned long long>(
                        record.recovery_rejected_count
                    ),
                    record.recovery_seconds,
                    static_cast<unsigned long long>(
                        record.recovery_iterations
                    ),
                    record.recovery_initial_residual,
                    record.recovery_final_residual,
                    record.recovery_final_primal_residual,
                    record.recovery_final_stationarity,
                    record.recovery_final_complementarity,
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
                && independent_path <= g4_quality_tolerance
                && independent_terminal <= g4_quality_tolerance
                && outer.virtual_control <= g4_quality_tolerance
                && (g4_policy != "hybrid-pdhcg-ipm"
                    || outer.hybrid_handoff_eligible != 0);
            std::printf(
                "{\"case\":\"g4_sample\",\"coordinate_id\":\"%s\","
                "\"family\":\"%s\","
                "\"policy\":\"%s\",\"intervals\":%zu,\"status\":%d,"
                "\"trust_class\":%.17g,\"transfer_class\":\"%s\","
                "\"qualified\":%s,\"quality_tolerance\":%.17g,"
                "\"canonical_residual\":%.17g,\"objective\":%.17g,"
                "\"dynamics\":%.17g,\"path\":%.17g,\"terminal\":%.17g,"
                "\"path_inventory\":{\"thrust\":%.17g,\"mass\":%.17g,"
                "\"altitude\":%.17g},"
                "\"virtual\":%.17g,\"trajectory_difference\":%.17g,"
                "\"cqp_seconds\":%.17g,\"scvx_seconds\":%.17g,"
                "\"topology_seconds\":%.17g,"
                "\"coefficient_seconds\":%.17g,"
                "\"workspace_create_seconds\":%.17g,"
                "\"update_seconds\":%.17g,\"scaling_seconds\":%.17g,"
                "\"solve_seconds\":%.17g,\"residual_seconds\":%.17g,"
                "\"replay_seconds\":%.17g,\"acceptance_seconds\":%.17g,"
                "\"h2d_seconds\":%.17g,\"d2h_seconds\":%.17g,"
                "\"recovery_seconds\":%.17g,\"recovery_iterations\":%llu,"
                "\"inner_iterations\":%llu,\"h2d_bytes\":%llu,"
                "\"d2h_bytes\":%llu,\"peak_device_bytes\":%llu,"
                "\"topology_allocations_after_create\":%llu,"
                "\"workspace_cache_hit\":%s,"
                "\"hidden_cpu_fallback\":%d,"
                "\"qoco_conversion_seconds\":%.17g,"
                "\"qoco_setup_seconds\":%.17g,"
                "\"qoco_update_seconds\":%.17g,"
                "\"qoco_solve_seconds\":%.17g,"
                "\"qoco_workspace_creations\":%llu,"
                "\"qoco_numeric_updates\":%llu,"
                "\"qoco_dual_discarded\":%d,"
                "\"hybrid_handoff_eligible\":%d,"
                "\"qoco_failure\":%d}\n",
                g4_coordinate_id.c_str(),
                g4_family.c_str(),
                g4_policy.c_str(),
                intervals,
                static_cast<int>(outer.status),
                dynamics_config.model == SPACEPDHCG_CUDA_DYNAMICS_LOW_THRUST
                    ? g4_family_class
                    : 0.0,
                g4_transfer_class.c_str(),
                qualified ? "true" : "false",
                g4_quality_tolerance,
                outer.canonical_residual,
                outer.objective,
                outer.dynamics_defect,
                independent_path,
                independent_terminal,
                path_inventory.thrust_violation,
                path_inventory.mass_violation,
                path_inventory.altitude_violation,
                outer.virtual_control,
                retained_change_maximum,
                outer.cqp_total_seconds,
                outer.scvx_total_seconds,
                outer.topology_seconds,
                outer.coefficient_seconds,
                outer.workspace_create_seconds,
                outer.update_seconds,
                outer.scaling_seconds,
                outer.solve_seconds,
                outer.residual_seconds,
                outer.replay_seconds,
                outer.acceptance_seconds,
                outer.h2d_seconds,
                outer.d2h_seconds,
                outer.recovery_seconds,
                static_cast<unsigned long long>(outer.recovery_iterations),
                static_cast<unsigned long long>(outer.inner_iterations),
                static_cast<unsigned long long>(outer.h2d_bytes),
                static_cast<unsigned long long>(outer.d2h_bytes),
                static_cast<unsigned long long>(diagnostics.peak_active_bytes),
                static_cast<unsigned long long>(
                    outer.topology_allocation_count_after_create
                ),
                workspace_cache_hit ? "true" : "false",
                outer.hidden_cpu_fallback,
                outer.qoco_conversion_seconds,
                outer.qoco_setup_seconds,
                outer.qoco_update_seconds,
                outer.qoco_solve_seconds,
                static_cast<unsigned long long>(
                    outer.qoco_workspace_creations
                ),
                static_cast<unsigned long long>(
                    outer.qoco_numeric_updates
                ),
                outer.qoco_dual_discarded,
                outer.hybrid_handoff_eligible,
                static_cast<int>(outer.qoco_failure)
            );
        }
        test::status_require(
            spacepdhcg_cuda_scvx_driver_destroy(&driver),
            "production outer driver destroy"
        );
        if (g4_policy == "pure-gpu-ipm"
            && g4_family == "P1-C-pd3"
            && production_outer_iterations >= 2U) {
            std::fprintf(
                stderr,
                "P1-C native QOCO diagnostic: status=%d outer=%u "
                "accepted=%u rejected=%u terminal=%.17g canonical=%.17g "
                "qoco_failure=%d ratios=%.17g/%.17g\n",
                static_cast<int>(outer.status),
                outer.outer_iterations,
                outer.accepted_steps,
                outer.rejected_steps,
                outer.terminal_residual,
                outer.canonical_residual,
                static_cast<int>(outer.qoco_failure),
                records[0].reduction_ratio,
                records[1].reduction_ratio
            );
            test::require(
                outer.accepted_steps == 2U,
                "P1-C pure QOCO must reproduce two accepted steps"
            );
            test::require(
                outer.terminal_residual <= g4_quality_tolerance,
                "P1-C pure QOCO terminal residual missed frozen quality"
            );
            test::require(
                std::abs(records[0].reduction_ratio - 0.999897) <= 5.0e-3
                    && std::abs(records[1].reduction_ratio - 0.999998)
                        <= 5.0e-3,
                "P1-C native ratios differ from the Python oracle"
            );
            test::require(
                outer.qoco_dual_discarded == 1,
                "P1-C second solve must report primal-only dual discard"
            );
        }
        const IntegrationResult result{
            diagnostics,
            outer,
            outer.topology_allocation_count_after_create,
            outer.topology_index_copy_count_after_create,
            0U,
            0.0,
            coefficient_parity_max,
            replay_parity_maximum,
        };
        if (cached_entry == nullptr) {
            test::destroy_workspace(workspace);
        }
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
    if (g4_sample_mode) {
        initial[0U] = 1.0e-6 * seeded_signed(50U);
        initial[1U] = 1.0e-6 * seeded_signed(51U);
    }
    std::vector<dynamics::PoweredDescentControl> controls(config.intervals);
    for (std::size_t interval = 0U; interval < config.intervals; ++interval) {
        const double thrust = 7'422.0 - 0.5 * static_cast<double>(interval);
        controls[interval] = {0.0, 0.0, thrust, thrust};
    }
    const auto nominal_states =
        subproblem.model().rollout(initial, controls, config.step_seconds);
    if (g4_sample_mode) {
        initial[0U] += 10.0 * g4_family_class;
        initial[2U] += 100.0 * g4_family_class;
        initial[3U] -= 5.0 * g4_family_class;
    }
    const auto states = subproblem.model().rollout(initial, controls, config.step_seconds);
    const std::array<double, 3U> target_position{
        nominal_states.back()[0], nominal_states.back()[1], nominal_states.back()[2]
    };
    const std::array<double, 3U> target_velocity{
        nominal_states.back()[3], nominal_states.back()[4], nominal_states.back()[5]
    };
    g4_instance_hash = hash_vector(
        hash_bytes(
            1'469'598'103'934'665'603ULL,
            initial.data(),
            initial.size() * sizeof(double)
        ),
        flatten_controls(controls)
    );
    g4_instance_hash = hash_bytes(
        g4_instance_hash,
        nominal_states.back().data(),
        nominal_states.back().size() * sizeof(double)
    );
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
    if (g4_sample_mode) {
        config.trust_radius = g4_family_class;
    }
    transcription::LowThrustSubproblem subproblem(
        dynamics::LowThrustTwoBodyModel{}, config
    );
    dynamics::LowThrustState initial{
        7'000.0, 0.0, 0.0, 0.0, 7.546, 0.0, 500.0
    };
    if (g4_sample_mode) {
        const double angle = std::acos(-1.0) * seeded_signed(300U);
        const double cosine = std::cos(angle);
        const double sine = std::sin(angle);
        initial[0U] = 7'000.0 * cosine;
        initial[1U] = 7'000.0 * sine;
        initial[3U] = -7.546 * sine;
        initial[4U] = 7.546 * cosine;
    }
    const std::vector<dynamics::LowThrustControl> controls(
        config.intervals,
        {0.0, 0.0, 0.0, 0.0}
    );
    const auto states = subproblem.model().rollout(initial, controls, config.step_seconds);
    auto target = states.back();
    if (g4_sample_mode) {
        const auto transfer = spacepdhcg::scvx::make_low_thrust_transfer_target(
            subproblem.model(),
            initial,
            config.intervals,
            config.step_seconds,
            spacepdhcg::scvx::low_thrust_transfer_class(g4_transfer_class)
        );
        target = transfer.first.back();
    }
    g4_instance_hash = hash_bytes(
        hash_bytes(
            1'469'598'103'934'665'603ULL,
            initial.data(),
            initial.size() * sizeof(double)
        ),
        target.data(),
        target.size() * sizeof(double)
    );
    const auto values = subproblem.values(
        states, controls, initial, target
    );
    auto displaced_initial = initial;
    displaced_initial[0U] += 2.0;
    displaced_initial[4U] -= 1.0e-3;
    const auto displaced_reference =
        spacepdhcg::scvx::make_low_thrust_transfer_target(
            subproblem.model(),
            displaced_initial,
            config.intervals,
            config.step_seconds,
            spacepdhcg::scvx::LowThrustTransferClass::combined
        );
    const auto displaced_target = displaced_reference.first.back();
    const auto displaced_values = subproblem.values(
        displaced_reference.first,
        displaced_reference.second,
        displaced_initial,
        displaced_target,
        0.37
    );
    const auto displaced_flat_states =
        flatten_states(displaced_reference.first, config.intervals);
    const auto displaced_flat_controls =
        flatten_controls(displaced_reference.second);
    const std::vector<double> displaced_initial_vector(
        displaced_initial.begin(), displaced_initial.end()
    );
    const std::vector<double> displaced_target_vector(
        displaced_target.begin(), displaced_target.end()
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
        std::vector<double>(target.begin(), target.end()),
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
        config.virtual_l1_weight,
        &displaced_values,
        &displaced_flat_states,
        &displaced_flat_controls,
        &displaced_initial_vector,
        &displaced_target_vector
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
    if (g4_sample_mode) {
        initial[0U] = 1.0e-6 * seeded_signed(350U);
        initial[1U] = 1.0e-6 * seeded_signed(351U);
    }
    std::vector<dynamics::PoweredDescent6DofControl> controls(config.intervals);
    for (std::size_t interval = 0U; interval < config.intervals; ++interval) {
        const double thrust = 7'422.0 - 0.2 * static_cast<double>(interval);
        controls[interval] = {
            0.0, 0.0, thrust, 0.0, 0.0, 0.0, thrust
        };
    }
    if (p1d_path_audit_mode) {
        controls.front()[2U] = 0.0;
        controls.front()[6U] = 1'000.0;
    }
    const auto nominal_states =
        subproblem.model().rollout(initial, controls, config.step_seconds);
    if (g4_sample_mode) {
        const double direction = seeded_signed(400U) < 0.0 ? -1.0 : 1.0;
        const double half_angle = 0.5 * g4_dispersion;
        initial[6] = std::cos(half_angle);
        initial[7U] = direction * std::sin(half_angle);
        initial[8U] = 0.0;
        initial[9U] = 0.0;
        initial[10U] += direction * g4_secondary_dispersion;
    }
    const auto states = subproblem.model().rollout(initial, controls, config.step_seconds);
    if (g4_sample_mode) {
        for (std::size_t component = 0U; component < initial.size(); ++component) {
            test::require(
                std::abs(states.front()[component] - initial[component])
                    <= 1.0e-14,
                "P1-D initial boundary differs from the normalised reference"
            );
        }
    }
    const auto values = subproblem.values(
        states, controls, initial, nominal_states.back()
    );
    g4_instance_hash = hash_vector(
        hash_bytes(
            hash_bytes(
                1'469'598'103'934'665'603ULL,
                initial.data(),
                initial.size() * sizeof(double)
            ),
            nominal_states.back().data(),
            nominal_states.back().size() * sizeof(double)
        ),
        flatten_controls(controls)
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

int run_invocation(const int argc, char** argv) {
    const auto mode = argc > 1 ? std::string_view(argv[1]) : std::string_view{};
    if (mode == "--g4-sample" && g4_deadline_seconds > 0.0) {
        g4_deadline = std::chrono::steady_clock::now()
            + std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                std::chrono::duration<double>(g4_deadline_seconds)
            );
    }
    if (mode == "--g4-capabilities") {
        std::printf(
            "{\"schema_version\":1,\"executor_semantics_version\":1,"
            "\"axes\":{"
            "\"family\":{\"status\":\"applied\",\"mechanism\":\"model selection\"},"
            "\"intervals\":{\"status\":\"applied\",\"mechanism\":\"transcription size\"},"
            "\"policy\":{\"status\":\"applied\",\"mechanism\":\"solver policy\"},"
            "\"quality_tier\":{\"status\":\"applied\",\"mechanism\":\"frozen tolerance\"},"
            "\"conditioning\":{\"status\":\"applied\","
            "\"mechanism\":\"equivalent dynamics-row scaling\"},"
            "\"scaling_mode\":{\"status\":\"applied\","
            "\"mechanism\":\"workspace scaling policy\"},"
            "\"warm_start_mode\":{\"status\":\"applied\","
            "\"mechanism\":\"resident iterate retention\"},"
            "\"family_classes\":{\"status\":\"applied\","
            "\"mechanism\":\"seeded physical initial/reference data\"},"
            "\"evaluation_seed\":{\"status\":\"applied\","
            "\"mechanism\":\"SplitMix64 physical instance generation\"},"
            "\"repeat\":{\"status\":\"execution_only\","
            "\"mechanism\":\"warmup/measured sample identity\"},"
            "\"solver_order\":{\"status\":\"execution_only\","
            "\"mechanism\":\"frozen external launch rotation\"}},"
            "\"independent_replay\":true,"
            "\"timing_boundary\":\"coefficient-generation-through-independent-"
            "replay-and-acceptance;cuda-startup-excluded\"}\n"
        );
        return 0;
    }
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
        || mode == "--g4-sample"
        || mode == "--g4-axis-probe"
        || mode == "--g4-diagnose"
        || mode == "--p1d-path-audit"
        || mode == "--dump-p1d"
        || mode == "--dump-p1e"
        || mode == "--diagnose-p1d"
        || mode == "--p1c-qoco-repeatability"
        || mode == "--qoco-unavailable";
    if (mode == "--p1c-qoco-repeatability") {
        p1c_qoco_repeatability_mode = true;
        g4_sample_mode = true;
        g4_family = "P1-C-pd3";
        g4_intervals = 20U;
        g4_policy = "pure-gpu-ipm";
        g4_quality_tier = "ipm";
        g4_quality_tolerance = 1.0e-8;
        g4_family_class = 0.01;
        production_outer_iterations = 2U;
    }
    if (mode == "--qoco-unavailable") {
        qoco_unavailable_mode = true;
        g4_sample_mode = true;
        g4_family = "P1-C-pd3";
        g4_intervals = 2U;
        g4_policy = "pure-gpu-ipm";
        g4_quality_tier = "ipm";
        g4_quality_tolerance = 1.0e-8;
        g4_dispersion = 0.01;
        production_outer_iterations = 1U;
    }
    if (mode == "--qoco-handback") {
        production_driver_mode = true;
        qoco_handback_mode = true;
        g4_sample_mode = true;
        g4_family = "P1-D-pd6";
        g4_intervals = 2U;
        g4_policy = "fixed-tight";
        g4_quality_tier = "ipm";
        g4_quality_tolerance = 1.0e-8;
        g4_dispersion = 0.05;
        g4_secondary_dispersion = 0.05;
        production_outer_iterations = 1U;
    }
    if (mode == "--diagnose-p1d") {
        p1d_diagnostic_mode = true;
        g4_sample_mode = true;
        g4_family = "P1-D-pd6";
        g4_intervals = 20U;
        g4_dispersion = 0.05;
        g4_secondary_dispersion = 0.05;
    }
    if (mode == "--dump-p1d") {
        dump_mode = true;
        g4_sample_mode = true;
        g4_family = "P1-D-pd6";
        g4_intervals = 20U;
        g4_policy = "fixed-tight";
        g4_quality_tier = "tight";
        g4_scaling_mode = "refresh_if_needed";
        g4_warm_mode = "primal_dual";
        g4_warm_start = SPACEPDHCG_CUDA_WARM_START_PRIMAL_DUAL;
        g4_quality_tolerance = 1.0e-6;
        g4_dispersion = 0.05;
        g4_secondary_dispersion = 0.05;
        production_outer_iterations = 1U;
    }
    if (mode == "--dump-p1e") {
        dump_mode = true;
        g4_sample_mode = true;
        g4_family = "P1-E-low-thrust";
        g4_intervals = 100U;
        g4_policy = "pure-gpu-ipm";
        g4_quality_tier = "ipm";
        g4_scaling_mode = "refresh_if_needed";
        g4_warm_mode = "primal_dual";
        g4_warm_start = SPACEPDHCG_CUDA_WARM_START_PRIMAL_DUAL;
        g4_quality_tolerance = 1.0e-8;
        g4_family_class = 0.25;
        g4_transfer_class = "radius_raise";
        production_outer_iterations = 1U;
    }
    if (mode == "--p1d-path-audit") {
        p1d_path_audit_mode = true;
        g4_sample_mode = true;
        g4_family = "P1-D-pd6";
        g4_intervals = 2U;
        g4_policy = "fixed-loose";
        g4_quality_tier = "coarse";
        g4_scaling_mode = "reuse";
        g4_warm_mode = "primal_dual";
        g4_warm_start = SPACEPDHCG_CUDA_WARM_START_PRIMAL_DUAL;
        g4_quality_tolerance = 1.0e-3;
        g4_dispersion = 0.01;
        g4_secondary_dispersion = 0.01;
        production_outer_iterations = 2U;
    }
    if (mode == "--g4-sample" || mode == "--g4-axis-probe"
        || mode == "--g4-diagnose") {
        test::require(
            argc == 21,
            "G4 mode requires family intervals policy warm quality "
            "outer-iterations family-coordinate-1 family-coordinate-2 "
            "quality-tier scaling-mode policy-sha256 conditioning seed "
            "repeat-kind repeat-index solver-order coordinate-id matrix-sha256 "
            "capability-sha256"
        );
        g4_sample_mode = true;
        g4_probe_mode = mode == "--g4-axis-probe";
        g4_diagnostic_mode = mode == "--g4-diagnose";
        g4_family = argv[2];
        g4_intervals = std::stoull(argv[3]);
        g4_policy = argv[4];
        const std::string_view warm = argv[5];
        g4_warm_mode = argv[5];
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
        if (g4_family == "P1-D-pd6") {
            g4_dispersion = std::stod(argv[8]);
            g4_secondary_dispersion = std::stod(argv[9]);
        } else {
            g4_family_class = std::stod(argv[8]);
            g4_transfer_class = argv[9];
        }
        g4_quality_tier = argv[10];
        g4_scaling_mode = argv[11];
        test::require(
            std::string_view(argv[12]) == frozen_g4::sha256,
            "G4 runtime policy SHA-256 differs from generated policy"
        );
        g4_conditioning_log10_span = std::stod(argv[13]);
        g4_evaluation_seed = std::stoull(argv[14]);
        g4_repeat_kind = argv[15];
        g4_repeat_index = static_cast<std::uint32_t>(std::stoul(argv[16]));
        g4_solver_order = static_cast<std::uint32_t>(std::stoul(argv[17]));
        g4_coordinate_id = argv[18];
        g4_matrix_sha256 = argv[19];
        g4_capability_sha256 = argv[20];
        test::require(
            g4_conditioning_log10_span == 0.0
                || g4_conditioning_log10_span == 2.0
                || g4_conditioning_log10_span == 4.0
                || g4_conditioning_log10_span == 8.0,
            "conditioning span is outside the frozen matrix"
        );
        test::require(
            g4_repeat_kind == "warmup" || g4_repeat_kind == "measured",
            "unknown G4 repeat kind"
        );
        test::require(
            std::find(
                g4_evaluation_seeds.begin(),
                g4_evaluation_seeds.end(),
                g4_evaluation_seed
            ) != g4_evaluation_seeds.end(),
            "seed is not in the frozen evaluation set"
        );
        test::require(
            (g4_repeat_kind == "warmup" && g4_repeat_index < 2U)
                || (g4_repeat_kind == "measured" && g4_repeat_index < 7U),
            "G4 repeat index is outside the frozen matrix"
        );
        test::require(g4_solver_order < 6U, "invalid G4 solver order");
        test::require(
            g4_coordinate_id.size() == 64U
                && g4_matrix_sha256.size() == 64U
                && g4_capability_sha256.size() == 64U,
            "G4 hash identity has invalid length"
        );
        test::require(
            g4_policy == "fixed-tight" || g4_policy == "fixed-loose"
                || g4_policy == "adaptive" || g4_policy == "adaptive+polish"
                || g4_policy == "pure-gpu-ipm"
                || g4_policy == "hybrid-pdhcg-ipm",
            "unknown G4 policy"
        );
        test::require(
            (g4_quality_tier == "coarse" && g4_quality_tolerance == 1.0e-3)
                || (g4_quality_tier == "medium"
                    && g4_quality_tolerance == 1.0e-4)
                || (g4_quality_tier == "tight"
                    && g4_quality_tolerance == 1.0e-6)
                || (g4_quality_tier == "ipm"
                    && g4_quality_tolerance == 1.0e-8),
            "G4 quality tier and tolerance disagree"
        );
        if (g4_family == "P1-E-low-thrust") {
            static_cast<void>(
                spacepdhcg::scvx::low_thrust_transfer_class(g4_transfer_class)
            );
            test::require(
                g4_family_class == 0.25 || g4_family_class == 0.5
                    || g4_family_class == 1.0 || g4_family_class == 2.0,
                "P1-E trust radius is outside the frozen matrix"
            );
            test::require(
                g4_intervals == 100U || g4_intervals == 500U
                    || g4_intervals == 2'000U
                    || g4_intervals == 10'000U
                    || g4_intervals == 50'000U,
                "P1-E interval count is outside the frozen matrix"
            );
        } else if (g4_family == "P1-C-pd3") {
            test::require(
                g4_family_class == 0.0 || g4_family_class == 0.01
                    || g4_family_class == 0.05 || g4_family_class == 0.1,
                "P1-C dispersion is outside the frozen matrix"
            );
            test::require(
                g4_intervals == 20U || g4_intervals == 50U
                    || g4_intervals == 100U || g4_intervals == 500U
                    || g4_intervals == 2'000U,
                "P1-C interval count is outside the frozen matrix"
            );
        } else if (g4_family == "P1-D-pd6") {
            test::require(
                (g4_dispersion == 0.0 || g4_dispersion == 0.05
                 || g4_dispersion == 0.2 || g4_dispersion == 0.5)
                    && (g4_secondary_dispersion == 0.0
                        || g4_secondary_dispersion == 0.05
                        || g4_secondary_dispersion == 0.2),
                "P1-D dispersion is outside the frozen matrix"
            );
            test::require(
                g4_intervals == 20U || g4_intervals == 50U
                    || g4_intervals == 100U || g4_intervals == 500U
                    || g4_intervals == 2'000U,
                "P1-D interval count is outside the frozen matrix"
            );
        } else {
            test::require(false, "unknown G4 family");
        }
        test::require(
            g4_family == "P1-E-low-thrust"
                || g4_transfer_class == "not_applicable",
            "non-P1-E family may not report a transfer class"
        );
        test::require(
            g4_family == "P1-D-pd6" || g4_secondary_dispersion == 0.0,
            "secondary dispersion is only defined for P1-D-pd6"
        );
        test::require(
            g4_scaling_mode == "always_refresh"
                || g4_scaling_mode == "reuse"
                || g4_scaling_mode == "refresh_if_needed",
            "unknown G4 scaling mode"
        );
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
                if (p1c_qoco_repeatability_mode) {
                    constexpr std::uint32_t repeats = 7U;
                    for (std::uint32_t repeat = 0U; repeat < repeats; ++repeat) {
                        const auto result = run_pd3();
                        test::require(
                            result.outer.status
                                    == SPACEPDHCG_CUDA_SCVX_CONVERGED
                                && result.outer.accepted_steps == 2U
                                && result.outer.terminal_residual
                                    <= g4_quality_tolerance,
                            "P1-C pure QOCO repeat changed qualification"
                        );
                    }
                    std::printf(
                        "{\"case\":\"p1c_qoco_repeatability\","
                        "\"repeats\":%u,\"qualified\":%u,"
                        "\"outer_attempt_budget\":%u}\n",
                        repeats,
                        repeats,
                        production_outer_iterations
                    );
                    return 0;
                }
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
        const double maximum_coefficient_difference = std::max({
            hcw.coefficient_parity_max,
            pd3.coefficient_parity_max,
            low_thrust.coefficient_parity_max,
            pd6.coefficient_parity_max,
        });
        std::printf(
            "{\"case\":\"production_outer_all\",\"families\":4,"
            "\"maximum_canonical\":%.9g,\"maximum_nonlinear\":%.9g,"
            "\"maximum_trajectory_difference\":%.9g,"
            "\"maximum_coefficient_difference\":%.17g,"
            "\"hidden_cpu_fallback\":false,"
            "\"topology_allocations_after_create\":0,"
            "\"topology_copies_after_create\":0}\n",
            maximum_canonical,
            maximum_nonlinear,
            maximum_trajectory_difference,
            maximum_coefficient_difference
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

int main(const int argc, char** argv) {
    const auto mode = argc > 1 ? std::string_view(argv[1]) : std::string_view{};
    if (mode != "--g4-server") {
        return run_invocation(argc, argv);
    }
    test::require(argc == 3, "G4 server requires a row deadline");
    const double server_deadline_seconds = std::stod(argv[2]);
    test::require(server_deadline_seconds > 0.0, "G4 server deadline must be positive");
    g4_deadline_seconds = server_deadline_seconds;

    const auto startup_begin = std::chrono::steady_clock::now();
    test::cuda_require(cudaFree(nullptr), "persistent G4 CUDA startup");
    const double startup_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - startup_begin
    ).count();
    std::printf(
        "{\"case\":\"g4_server_ready\",\"protocol_version\":2,"
        "\"cuda_startup_seconds\":%.17g}\n",
        startup_seconds
    );
    std::fflush(stdout);

    const auto parse_request = [](const std::string& line) {
        std::vector<std::string> arguments{"device_scvx_integration_test"};
        std::size_t begin = 0U;
        while (begin <= line.size()) {
            const auto end = line.find('\t', begin);
            arguments.emplace_back(
                line.substr(
                    begin,
                    end == std::string::npos ? std::string::npos : end - begin
                )
            );
            if (end == std::string::npos) {
                break;
            }
            begin = end + 1U;
        }
        return arguments;
    };
    const auto valid_request = [](const std::vector<std::string>& arguments) {
        return arguments.size() == 21U && arguments[1] == "--g4-sample";
    };
    const auto execute_request = [&](std::vector<std::string>& arguments,
                                     const std::size_t lane_index) {
        g4_deadline_seconds = server_deadline_seconds;
        g4_server_request = true;
        g4_lane_index = lane_index;
        std::vector<char*> pointers;
        pointers.reserve(arguments.size());
        for (auto& argument : arguments) {
            pointers.push_back(argument.data());
        }
        return run_invocation(
            static_cast<int>(pointers.size()),
            pointers.data()
        );
    };
    const auto emit_result = [](const std::vector<std::string>& arguments,
                                const int returncode,
                                const double elapsed_seconds) {
        std::printf(
            "{\"case\":\"g4_server_result\",\"protocol_version\":2,"
            "\"coordinate_id\":\"%s\",\"returncode\":%d,"
            "\"elapsed_seconds\":%.17g}\n",
            arguments[18].c_str(),
            returncode,
            elapsed_seconds
        );
    };
    std::mutex server_output_mutex;

    std::string line;
    while (std::getline(std::cin, line)) {
        if (line == "cancel") {
            break;
        }
        if (line.starts_with("batch\t")) {
            const auto count = static_cast<std::size_t>(
                std::stoull(line.substr(6U))
            );
            test::require(count > 0U && count <= 1024U, "invalid G4 batch size");
            std::vector<std::vector<std::string>> requests;
            requests.reserve(count);
            for (std::size_t index = 0U; index < count; ++index) {
                test::require(
                    static_cast<bool>(std::getline(std::cin, line)),
                    "truncated G4 batch"
                );
                requests.push_back(parse_request(line));
                test::require(valid_request(requests.back()), "invalid G4 batch request");
            }
            std::vector<int> returncodes(count, 0);
            std::vector<std::thread> lanes;
            lanes.reserve(count);
            for (std::size_t index = 0U; index < count; ++index) {
                lanes.emplace_back([&, index]() {
                    const auto row_started = std::chrono::steady_clock::now();
                    returncodes[index] = execute_request(requests[index], index);
                    const double row_elapsed = std::chrono::duration<double>(
                        std::chrono::steady_clock::now() - row_started
                    ).count();
                    std::lock_guard lock(server_output_mutex);
                    emit_result(requests[index], returncodes[index], row_elapsed);
                    std::fflush(stdout);
                });
            }
            for (auto& lane : lanes) {
                lane.join();
            }
            std::printf(
                "{\"case\":\"g4_batch_result\",\"protocol_version\":2,"
                "\"batch_size\":%zu}\n",
                count
            );
            std::fflush(stdout);
            continue;
        }

        auto arguments = parse_request(line);
        if (!valid_request(arguments)) {
            std::printf(
                "{\"case\":\"g4_server_error\",\"protocol_version\":1,"
                "\"reason\":\"invalid request\"}\n"
            );
            std::fflush(stdout);
            continue;
        }
        const auto row_started = std::chrono::steady_clock::now();
        const int returncode = execute_request(arguments, 0U);
        const double row_elapsed = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - row_started
        ).count();
        emit_result(arguments, returncode, row_elapsed);
        std::fflush(stdout);
    }
    return 0;
}
