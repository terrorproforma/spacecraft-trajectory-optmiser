/*
 * Production device-resident deterministic SCvx outer driver.
 *
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "spacepdhcg/cuda/device_scvx_c_api.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct spacepdhcg_cuda_scvx_driver spacepdhcg_cuda_scvx_driver;

typedef enum spacepdhcg_cuda_scvx_status {
    SPACEPDHCG_CUDA_SCVX_CONVERGED = 0,
    SPACEPDHCG_CUDA_SCVX_MAXIMUM_ITERATIONS = 1,
    SPACEPDHCG_CUDA_SCVX_TRUST_REGION_EXHAUSTED = 2,
    SPACEPDHCG_CUDA_SCVX_INNER_FAILURE = 3,
    SPACEPDHCG_CUDA_SCVX_CANCELLED = 4,
    SPACEPDHCG_CUDA_SCVX_INVALID = 5
} spacepdhcg_cuda_scvx_status;

typedef enum spacepdhcg_cuda_scvx_phase {
    SPACEPDHCG_CUDA_SCVX_REPAIR = 0,
    SPACEPDHCG_CUDA_SCVX_PROGRESS = 1,
    SPACEPDHCG_CUDA_SCVX_REFINEMENT = 2,
    SPACEPDHCG_CUDA_SCVX_POLISH = 3
} spacepdhcg_cuda_scvx_phase;

typedef enum spacepdhcg_cuda_scvx_policy {
    SPACEPDHCG_CUDA_SCVX_ADAPTIVE = 0,
    SPACEPDHCG_CUDA_SCVX_FIXED_TIGHT = 1,
    SPACEPDHCG_CUDA_SCVX_FIXED_LOOSE = 2,
    SPACEPDHCG_CUDA_SCVX_ADAPTIVE_POLISH = 3
} spacepdhcg_cuda_scvx_policy;

typedef enum spacepdhcg_cuda_scvx_trust_action {
    SPACEPDHCG_CUDA_SCVX_TRUST_RETAIN = 0,
    SPACEPDHCG_CUDA_SCVX_TRUST_SHRINK = 1,
    SPACEPDHCG_CUDA_SCVX_TRUST_EXPAND = 2
} spacepdhcg_cuda_scvx_trust_action;

typedef struct spacepdhcg_cuda_scvx_numeric_update {
    uint32_t abi_version;
    spacepdhcg_accelerator_buffer_view quadratic_diagonal_positions;
    spacepdhcg_accelerator_buffer_view radial_positions;
    spacepdhcg_accelerator_buffer_view quaternion_positions;
    size_t terminal_row_start;
    size_t radial_row_start;
    size_t quaternion_row_start;
    size_t stage_trust_row_start;
    size_t stage_trust_stride;
    size_t terminal_trust_row_start;
    size_t virtual_variable_offset;
    size_t epigraph_variable_offset;
    double state_trust_scales[14];
    double control_trust_scales[7];
    double fuel_weight;
    double virtual_l1_weight;
} spacepdhcg_cuda_scvx_numeric_update;

typedef struct spacepdhcg_cuda_scvx_problem {
    uint32_t abi_version;
    spacepdhcg_cuda_workspace* workspace;
    uint64_t topology_fingerprint;
    size_t intervals;
    size_t state_dimension;
    size_t control_dimension;
    spacepdhcg_cuda_dynamics_config dynamics;
    spacepdhcg_cqp_numeric_accelerator_views numeric;
    spacepdhcg_cuda_variational_request variational;
    spacepdhcg_cuda_csc_dynamics_fill dynamics_fill;
    spacepdhcg_accelerator_buffer_view state_variable_indices;
    spacepdhcg_accelerator_buffer_view control_variable_indices;
    spacepdhcg_accelerator_buffer_view virtual_variable_indices;
    spacepdhcg_accelerator_buffer_view reference_states;
    spacepdhcg_accelerator_buffer_view reference_controls;
    spacepdhcg_accelerator_buffer_view initial_state;
    spacepdhcg_accelerator_buffer_view target_state;
    spacepdhcg_cuda_scvx_numeric_update numeric_update;
} spacepdhcg_cuda_scvx_problem;

typedef struct spacepdhcg_cuda_scvx_options {
    uint32_t abi_version;
    uint32_t maximum_outer_iterations;
    uint32_t minimum_outer_iterations;
    uint32_t maximum_resolves_per_iteration;
    double convergence_tolerance;
    double step_tolerance;
    double acceptance_threshold;
    double restoration_reduction;
    double feasibility_penalty;
    double virtual_penalty;
    double initial_trust_radius;
    double minimum_trust_radius;
    double maximum_trust_radius;
    double shrink_factor;
    double expansion_factor;
    double strong_agreement_threshold;
    double near_boundary_fraction;
    double fixed_inner_tolerance;
    uint64_t fixed_inner_iteration_limit;
    spacepdhcg_cuda_scvx_policy policy;
    spacepdhcg_cuda_warm_start_mode warm_start_mode;
    double adaptive_epsilon_max;
    double adaptive_epsilon_floor;
    double adaptive_epsilon_0;
    double adaptive_coefficient;
    double adaptive_alpha;
    double adaptive_gamma;
    double repair_tolerance_ceiling;
    double progress_tolerance_ceiling;
    double refinement_tolerance_ceiling;
    double polish_tolerance_ceiling;
    uint64_t repair_iteration_limit;
    uint64_t progress_iteration_limit;
    uint64_t refinement_iteration_limit;
    uint64_t polish_iteration_limit;
    double resolve_trigger_multiple;
    double resolve_refinement_factor;
    double resolve_minimum_tolerance;
    double final_polish_tolerance;
    uint64_t final_polish_iteration_limit;
} spacepdhcg_cuda_scvx_options;

typedef struct spacepdhcg_cuda_scvx_iteration {
    uint32_t outer_iteration;
    spacepdhcg_cuda_scvx_phase phase;
    double requested_tolerance;
    double achieved_residual;
    uint64_t inner_iterations;
    double trust_radius_before;
    double trust_radius_after;
    double predicted_reduction;
    double actual_reduction;
    double reduction_ratio;
    double step_fraction;
    double objective;
    double virtual_control;
    double dynamics_defect;
    double path_violation;
    double terminal_residual;
    int32_t accepted;
    int32_t restoration_accepted;
    int32_t re_solved;
    int32_t scaling_refreshed;
    int32_t recovery_used;
    double native_primal_residual;
    double native_dual_residual;
    double complementarity_residual;
    double scaling_min;
    double scaling_max;
    uint64_t matvecs;
    uint64_t cone_projections;
    uint64_t cqp_numeric_fingerprint;
    uint64_t resolve_numeric_fingerprint;
    int32_t resolve_fingerprint_match;
    spacepdhcg_cuda_scvx_trust_action trust_action;
    spacepdhcg_cuda_warm_start_mode warm_start_mode;
    spacepdhcg_cuda_recovery_reason recovery_reason;
    int32_t forcing_satisfied;
    int32_t final_polish_handoff;
} spacepdhcg_cuda_scvx_iteration;

typedef struct spacepdhcg_cuda_scvx_result {
    uint32_t abi_version;
    spacepdhcg_cuda_scvx_status status;
    uint32_t outer_iterations;
    uint32_t accepted_steps;
    uint32_t rejected_steps;
    uint32_t resolved_steps;
    uint64_t inner_iterations;
    double objective;
    double canonical_residual;
    double dynamics_defect;
    double path_violation;
    double terminal_residual;
    double virtual_control;
    double trajectory_step;
    double final_trust_radius;
    double topology_seconds;
    double coefficient_seconds;
    double workspace_create_seconds;
    double update_seconds;
    double scaling_seconds;
    double h2d_seconds;
    double solve_seconds;
    double recovery_seconds;
    double residual_seconds;
    double replay_seconds;
    double acceptance_seconds;
    double d2h_seconds;
    double cqp_total_seconds;
    double scvx_total_seconds;
    uint64_t allocation_count;
    uint64_t allocation_bytes;
    uint64_t h2d_copy_count;
    uint64_t h2d_bytes;
    uint64_t d2h_copy_count;
    uint64_t d2h_bytes;
    uint64_t device_copy_count;
    uint64_t device_copy_bytes;
    uint64_t topology_allocation_count_after_create;
    uint64_t topology_index_copy_count_after_create;
    uint64_t recovery_iterations;
    int32_t hidden_cpu_fallback;
    int32_t used_declared_stream;
} spacepdhcg_cuda_scvx_result;

spacepdhcg_cuda_status spacepdhcg_cuda_scvx_update_numeric_async(
    const spacepdhcg_cuda_scvx_problem* problem,
    double trust_radius,
    double virtual_penalty,
    spacepdhcg_accelerator_stream stream
);

spacepdhcg_cuda_status spacepdhcg_cuda_scvx_driver_create(
    const spacepdhcg_cuda_scvx_problem* problem,
    const spacepdhcg_cuda_scvx_options* options,
    spacepdhcg_cuda_scvx_driver** driver
);

spacepdhcg_cuda_status spacepdhcg_cuda_scvx_driver_solve(
    spacepdhcg_cuda_scvx_driver* driver,
    spacepdhcg_accelerator_stream stream,
    spacepdhcg_cuda_scvx_iteration* iterations,
    size_t iteration_capacity,
    spacepdhcg_cuda_scvx_result* result
);

spacepdhcg_cuda_status spacepdhcg_cuda_scvx_driver_cancel(
    spacepdhcg_cuda_scvx_driver* driver
);

spacepdhcg_cuda_status spacepdhcg_cuda_scvx_driver_destroy(
    spacepdhcg_cuda_scvx_driver** driver
);

#ifdef __cplusplus
}
#endif
