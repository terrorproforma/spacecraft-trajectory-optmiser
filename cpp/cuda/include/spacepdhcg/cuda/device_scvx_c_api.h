/*
 * Device-resident deterministic dynamics linearisation for Gate G3.
 *
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "spacepdhcg/cuda/persistent_pdhcg_c_api.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum spacepdhcg_cuda_dynamics_model {
    SPACEPDHCG_CUDA_DYNAMICS_HCW = 0,
    SPACEPDHCG_CUDA_DYNAMICS_POWERED_DESCENT_3DOF = 1,
    SPACEPDHCG_CUDA_DYNAMICS_LOW_THRUST = 2,
    SPACEPDHCG_CUDA_DYNAMICS_POWERED_DESCENT_6DOF = 3
} spacepdhcg_cuda_dynamics_model;

typedef struct spacepdhcg_cuda_dynamics_config {
    uint32_t abi_version;
    spacepdhcg_cuda_dynamics_model model;
    double step_seconds;
    double mean_motion;
    double gravity[3];
    double gravitational_parameter;
    double thrust_to_acceleration;
    double mass_flow_coefficient;
    double principal_inertia[3];
} spacepdhcg_cuda_dynamics_config;

typedef struct spacepdhcg_cuda_variational_request {
    uint32_t abi_version;
    size_t intervals;
    spacepdhcg_accelerator_buffer_view reference_states;
    spacepdhcg_accelerator_buffer_view reference_controls;
    spacepdhcg_accelerator_buffer_view propagated_states;
    spacepdhcg_accelerator_buffer_view state_transition;
    spacepdhcg_accelerator_buffer_view control_sensitivity;
    spacepdhcg_accelerator_buffer_view affine_offset;
} spacepdhcg_cuda_variational_request;

typedef struct spacepdhcg_cuda_csc_dynamics_fill {
    uint32_t abi_version;
    size_t intervals;
    size_t state_dimension;
    size_t control_dimension;
    spacepdhcg_accelerator_buffer_view state_transition;
    spacepdhcg_accelerator_buffer_view control_sensitivity;
    spacepdhcg_accelerator_buffer_view affine_offset;
    spacepdhcg_accelerator_buffer_view state_positions;
    spacepdhcg_accelerator_buffer_view control_positions;
    spacepdhcg_accelerator_buffer_view next_state_positions;
    spacepdhcg_accelerator_buffer_view virtual_positions;
    spacepdhcg_accelerator_buffer_view scalar_values;
    spacepdhcg_accelerator_buffer_view scalar_lower;
    spacepdhcg_accelerator_buffer_view scalar_upper;
    size_t dynamics_row_start;
} spacepdhcg_cuda_csc_dynamics_fill;

spacepdhcg_cuda_status spacepdhcg_cuda_variational_rk4_async(
    const spacepdhcg_cuda_dynamics_config* config,
    const spacepdhcg_cuda_variational_request* request,
    spacepdhcg_accelerator_stream stream
);

spacepdhcg_cuda_status spacepdhcg_cuda_fill_dynamics_csc_async(
    const spacepdhcg_cuda_csc_dynamics_fill* request,
    spacepdhcg_accelerator_stream stream
);

#ifdef __cplusplus
}
#endif
