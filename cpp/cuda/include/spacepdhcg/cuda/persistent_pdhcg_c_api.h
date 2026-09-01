/*
 * SpacePDHCG persistent CUDA workspace C ABI.
 *
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "spacepdhcg/accelerator_c_api.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define SPACEPDHCG_CUDA_WORKSPACE_ABI_VERSION 1U

typedef struct spacepdhcg_cuda_workspace spacepdhcg_cuda_workspace;

typedef enum spacepdhcg_cuda_status {
    SPACEPDHCG_CUDA_SUCCESS = 0,
    SPACEPDHCG_CUDA_INVALID_ARGUMENT = 1,
    SPACEPDHCG_CUDA_INVALID_STATE = 2,
    SPACEPDHCG_CUDA_TOPOLOGY_MISMATCH = 3,
    SPACEPDHCG_CUDA_POINTER_CONTRACT = 4,
    SPACEPDHCG_CUDA_BUSY = 5,
    SPACEPDHCG_CUDA_RUNTIME_ERROR = 6,
    SPACEPDHCG_CUDA_NUMERICAL_FAILURE = 7,
    SPACEPDHCG_CUDA_UNSUPPORTED = 8,
    SPACEPDHCG_CUDA_OUT_OF_MEMORY = 9,
    SPACEPDHCG_CUDA_INTERNAL_ERROR = 10
} spacepdhcg_cuda_status;

typedef enum spacepdhcg_cuda_workspace_state {
    SPACEPDHCG_CUDA_UNINITIALISED = 0,
    SPACEPDHCG_CUDA_CREATED = 1,
    SPACEPDHCG_CUDA_VALUES_UPDATED = 2,
    SPACEPDHCG_CUDA_WARM_STARTED = 3,
    SPACEPDHCG_CUDA_SOLVING = 4,
    SPACEPDHCG_CUDA_SOLVED = 5,
    SPACEPDHCG_CUDA_FAILED = 6,
    SPACEPDHCG_CUDA_CANCELLED = 7,
    SPACEPDHCG_CUDA_DESTROYED = 8
} spacepdhcg_cuda_workspace_state;

typedef enum spacepdhcg_cuda_cone_kind {
    SPACEPDHCG_CUDA_CONE_SECOND_ORDER = 0,
    SPACEPDHCG_CUDA_CONE_ROTATED_SECOND_ORDER = 1,
    SPACEPDHCG_CUDA_CONE_EXPONENTIAL = 2,
    SPACEPDHCG_CUDA_CONE_POWER = 3,
    SPACEPDHCG_CUDA_CONE_POSITIVE_SEMIDEFINITE = 4
} spacepdhcg_cuda_cone_kind;

typedef enum spacepdhcg_cuda_warm_start_mode {
    SPACEPDHCG_CUDA_WARM_START_NONE = 0,
    SPACEPDHCG_CUDA_WARM_START_PRIMAL = 1,
    SPACEPDHCG_CUDA_WARM_START_PRIMAL_DUAL = 2,
    SPACEPDHCG_CUDA_WARM_START_FULL_RETAINED = 3
} spacepdhcg_cuda_warm_start_mode;

typedef enum spacepdhcg_cuda_scaling_mode {
    SPACEPDHCG_CUDA_SCALING_ALWAYS_REFRESH = 0,
    SPACEPDHCG_CUDA_SCALING_REUSE = 1,
    SPACEPDHCG_CUDA_SCALING_REFRESH_IF_NEEDED = 2
} spacepdhcg_cuda_scaling_mode;

typedef enum spacepdhcg_cuda_reset_flags {
    SPACEPDHCG_CUDA_RESET_ITERATES = 1U,
    SPACEPDHCG_CUDA_RESET_SCALING = 2U,
    SPACEPDHCG_CUDA_RESET_FULL = 3U
} spacepdhcg_cuda_reset_flags;

typedef enum spacepdhcg_cuda_termination {
    SPACEPDHCG_CUDA_TERMINATION_UNSPECIFIED = 0,
    SPACEPDHCG_CUDA_TERMINATION_OPTIMAL = 1,
    SPACEPDHCG_CUDA_TERMINATION_ITERATION_LIMIT = 2,
    SPACEPDHCG_CUDA_TERMINATION_CANCELLED = 3,
    SPACEPDHCG_CUDA_TERMINATION_NUMERICAL_FAILURE = 4
} spacepdhcg_cuda_termination;

typedef enum spacepdhcg_cuda_recovery_reason {
    SPACEPDHCG_CUDA_RECOVERY_NOT_TRIGGERED = 0,
    SPACEPDHCG_CUDA_RECOVERY_TIGHT_ITERATION_LIMIT = 1,
    SPACEPDHCG_CUDA_RECOVERY_QUALIFIED = 2,
    SPACEPDHCG_CUDA_RECOVERY_UNSUPPORTED_CONE = 3,
    SPACEPDHCG_CUDA_RECOVERY_NONFINITE_INPUT = 4,
    SPACEPDHCG_CUDA_RECOVERY_ZERO_CURVATURE = 5,
    SPACEPDHCG_CUDA_RECOVERY_INCONSISTENT_ACTIVE_SET = 6,
    SPACEPDHCG_CUDA_RECOVERY_DUAL_INFEASIBLE = 7,
    SPACEPDHCG_CUDA_RECOVERY_EXHAUSTED = 8,
    SPACEPDHCG_CUDA_RECOVERY_CANCELLED = 9
} spacepdhcg_cuda_recovery_reason;

typedef struct spacepdhcg_cuda_cone_descriptor {
    spacepdhcg_cuda_cone_kind kind;
    int32_t start;
    int32_t vector_dimension;
    double power_alpha;
} spacepdhcg_cuda_cone_descriptor;

/*
 * Sparse topology is supplied through the accelerator exchange. Cone
 * descriptors are small host metadata copied synchronously during create.
 */
typedef struct spacepdhcg_cuda_structure {
    uint32_t abi_version;
    uint64_t topology_fingerprint;
    int32_t variables;
    int32_t scalar_rows;
    int32_t affine_rows;
    size_t quadratic_nonzeros;
    size_t scalar_nonzeros;
    size_t affine_nonzeros;
    const spacepdhcg_cuda_cone_descriptor* affine_cones;
    size_t affine_cone_count;
    const spacepdhcg_cuda_cone_descriptor* variable_cones;
    size_t variable_cone_count;
} spacepdhcg_cuda_structure;

typedef void (*spacepdhcg_cuda_external_retain_fn)(void* context);
typedef void (*spacepdhcg_cuda_external_release_fn)(void* context);

typedef struct spacepdhcg_cuda_create_options {
    uint32_t abi_version;
    spacepdhcg_cuda_scaling_mode scaling_mode;
    double maximum_relative_matrix_change;
    double maximum_relative_vector_change;
    uint64_t maximum_scaling_reuse_updates;
    int32_t debug_validate_aliases;
    void* external_lifetime_context;
    spacepdhcg_cuda_external_retain_fn retain_external;
    spacepdhcg_cuda_external_release_fn release_external;
} spacepdhcg_cuda_create_options;

typedef struct spacepdhcg_cuda_solve_options {
    uint32_t abi_version;
    double optimality_tolerance;
    double feasibility_tolerance;
    uint64_t iteration_limit;
    uint32_t residual_check_frequency;
} spacepdhcg_cuda_solve_options;

typedef struct spacepdhcg_cuda_diagnostics {
    uint32_t abi_version;
    spacepdhcg_cuda_workspace_state state;
    spacepdhcg_cuda_termination termination;
    uint64_t iterations;
    double objective;
    double scalar_primal_violation_inf;
    double box_violation_inf;
    double affine_cone_distance_inf;
    double stationarity_inf;
    double natural_residual_inf;
    double complementarity_inf;
    double relative_primal_residual;
    double relative_dual_residual;
    double coefficient_change_max;
    double coefficient_change_norm;
    double scaling_min;
    double scaling_max;
    double update_seconds;
    double scaling_seconds;
    double solve_seconds;
    double residual_seconds;
    uint64_t allocation_count;
    uint64_t free_count;
    uint64_t active_allocation_count;
    uint64_t active_bytes;
    uint64_t peak_active_bytes;
    uint64_t topology_allocation_count;
    uint64_t topology_index_copy_count;
    uint64_t total_copy_count;
    uint64_t total_copy_bytes;
    uint64_t update_epoch;
    uint64_t solve_epoch;
    uint64_t scaling_epoch;
    uint64_t scaling_reuse_count;
    uint64_t graph_epoch;
    uint64_t allocation_delta_last_update;
    uint64_t topology_allocation_delta_last_update;
    uint64_t topology_index_copy_delta_last_update;
    spacepdhcg_cuda_warm_start_mode warm_start_mode;
    int32_t warm_start_accepted;
    int32_t scaling_refreshed;
    int32_t used_declared_stream;
    int32_t hidden_cpu_fallback;
    uint64_t recovery_count;
    uint64_t recovery_rejected_count;
    uint64_t recovery_iterations;
    uint64_t recovery_attempt_count;
    spacepdhcg_cuda_recovery_reason recovery_trigger_reason;
    spacepdhcg_cuda_recovery_reason recovery_outcome_reason;
    double recovery_seconds;
    double recovery_initial_residual;
    double recovery_final_residual;
    double recovery_final_primal_residual;
    double recovery_final_stationarity;
    double recovery_final_complementarity;
    int32_t recovery_stationarity_index;
    double recovery_stationarity_value;
} spacepdhcg_cuda_diagnostics;

typedef struct spacepdhcg_cuda_pointer_snapshot {
    uintptr_t quadratic_offsets;
    uintptr_t quadratic_indices;
    uintptr_t scalar_offsets;
    uintptr_t scalar_indices;
    uintptr_t affine_offsets;
    uintptr_t affine_indices;
    uintptr_t quadratic_values;
    uintptr_t scalar_values;
    uintptr_t affine_values;
    uintptr_t primal;
    uintptr_t dual;
    uintptr_t scaling;
} spacepdhcg_cuda_pointer_snapshot;

spacepdhcg_cuda_status spacepdhcg_cuda_workspace_create(
    const spacepdhcg_cuda_structure* structure,
    const spacepdhcg_cqp_accelerator_exchange* exchange,
    const spacepdhcg_cuda_create_options* options,
    spacepdhcg_cuda_workspace** workspace
);

spacepdhcg_cuda_status spacepdhcg_cuda_workspace_update_async(
    spacepdhcg_cuda_workspace* workspace,
    uint64_t topology_fingerprint,
    const spacepdhcg_cqp_numeric_accelerator_views* values,
    spacepdhcg_accelerator_stream stream
);

spacepdhcg_cuda_status spacepdhcg_cuda_workspace_warm_start_async(
    spacepdhcg_cuda_workspace* workspace,
    spacepdhcg_cuda_warm_start_mode mode,
    const spacepdhcg_cqp_iterate_accelerator_views* iterates,
    spacepdhcg_accelerator_stream stream
);

spacepdhcg_cuda_status spacepdhcg_cuda_workspace_solve_async(
    spacepdhcg_cuda_workspace* workspace,
    const spacepdhcg_cuda_solve_options* options,
    spacepdhcg_accelerator_stream stream
);

spacepdhcg_cuda_status spacepdhcg_cuda_workspace_query(
    spacepdhcg_cuda_workspace* workspace,
    int32_t* complete
);

spacepdhcg_cuda_status spacepdhcg_cuda_workspace_wait(
    spacepdhcg_cuda_workspace* workspace
);

spacepdhcg_cuda_status spacepdhcg_cuda_workspace_diagnostics(
    spacepdhcg_cuda_workspace* workspace,
    spacepdhcg_cuda_diagnostics* diagnostics
);

spacepdhcg_cuda_status spacepdhcg_cuda_workspace_residuals_async(
    spacepdhcg_cuda_workspace* workspace,
    spacepdhcg_accelerator_stream stream
);

spacepdhcg_cuda_status spacepdhcg_cuda_workspace_pointer_snapshot(
    spacepdhcg_cuda_workspace* workspace,
    spacepdhcg_cuda_pointer_snapshot* snapshot
);

spacepdhcg_cuda_status spacepdhcg_cuda_workspace_reset_async(
    spacepdhcg_cuda_workspace* workspace,
    uint32_t reset_flags,
    spacepdhcg_accelerator_stream stream
);

spacepdhcg_cuda_status spacepdhcg_cuda_workspace_refresh_scaling_async(
    spacepdhcg_cuda_workspace* workspace,
    spacepdhcg_accelerator_stream stream
);

spacepdhcg_cuda_status spacepdhcg_cuda_workspace_checkpoint_bytes(
    const spacepdhcg_cuda_workspace* workspace,
    size_t* bytes
);

spacepdhcg_cuda_status spacepdhcg_cuda_workspace_checkpoint_async(
    spacepdhcg_cuda_workspace* workspace,
    spacepdhcg_accelerator_buffer_view checkpoint,
    spacepdhcg_accelerator_stream stream
);

spacepdhcg_cuda_status spacepdhcg_cuda_workspace_restore_async(
    spacepdhcg_cuda_workspace* workspace,
    uint64_t topology_fingerprint,
    spacepdhcg_accelerator_buffer_view checkpoint,
    spacepdhcg_accelerator_stream stream
);

spacepdhcg_cuda_status spacepdhcg_cuda_workspace_cancel(
    spacepdhcg_cuda_workspace* workspace
);

spacepdhcg_cuda_status spacepdhcg_cuda_workspace_last_error(
    const spacepdhcg_cuda_workspace* workspace,
    char* destination,
    size_t destination_bytes
);

spacepdhcg_cuda_status spacepdhcg_cuda_workspace_destroy(
    spacepdhcg_cuda_workspace** workspace
);

#ifdef __cplusplus
}
#endif
