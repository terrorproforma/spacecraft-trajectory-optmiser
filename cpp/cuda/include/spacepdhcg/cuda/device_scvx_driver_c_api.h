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
    spacepdhcg_accelerator_buffer_view target_state;
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
    double fixed_inner_tolerance;
    uint64_t fixed_inner_iteration_limit;
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
