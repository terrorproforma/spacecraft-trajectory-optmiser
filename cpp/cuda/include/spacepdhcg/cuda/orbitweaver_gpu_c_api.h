/*
 * Fixed-memory deterministic OrbitWeaver CUDA batch primitives.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "spacepdhcg/accelerator_c_api.h"
#include "spacepdhcg/cuda/persistent_pdhcg_c_api.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define SPACEPDHCG_ORBITWEAVER_GPU_ABI_VERSION 1U

typedef struct spacepdhcg_orbitweaver_lambert_workspace
    spacepdhcg_orbitweaver_lambert_workspace;

typedef enum spacepdhcg_orbitweaver_arc_status {
    SPACEPDHCG_ORBITWEAVER_ARC_FEASIBLE = 0,
    SPACEPDHCG_ORBITWEAVER_ARC_NO_SOLUTION = 1,
    SPACEPDHCG_ORBITWEAVER_ARC_INVALID_INPUT = 2,
    SPACEPDHCG_ORBITWEAVER_ARC_UNSUPPORTED = 3,
    SPACEPDHCG_ORBITWEAVER_ARC_NUMERICAL_FAILURE = 4,
    SPACEPDHCG_ORBITWEAVER_ARC_CANCELLED = 5,
    SPACEPDHCG_ORBITWEAVER_ARC_BACKEND_FAILURE = 6
} spacepdhcg_orbitweaver_arc_status;

typedef enum spacepdhcg_orbitweaver_lambert_branch {
    SPACEPDHCG_ORBITWEAVER_LAMBERT_UNIQUE = 0,
    SPACEPDHCG_ORBITWEAVER_LAMBERT_LOWER_PARAMETER = 1,
    SPACEPDHCG_ORBITWEAVER_LAMBERT_HIGHER_PARAMETER = 2
} spacepdhcg_orbitweaver_lambert_branch;

typedef struct spacepdhcg_orbitweaver_lambert_request {
    uint64_t deterministic_id;
    double departure_position[3];
    double arrival_position[3];
    double time_of_flight;
    double gravitational_parameter;
    double time_tolerance;
    uint32_t maximum_iterations;
    uint32_t maximum_revolutions;
    int32_t include_short_way;
    int32_t include_long_way;
} spacepdhcg_orbitweaver_lambert_request;

typedef struct spacepdhcg_orbitweaver_lambert_result {
    uint64_t deterministic_id;
    uint32_t input_index;
    uint32_t family_index;
    uint32_t revolutions;
    int32_t long_way;
    spacepdhcg_orbitweaver_lambert_branch branch;
    spacepdhcg_orbitweaver_arc_status status;
    double departure_velocity[3];
    double arrival_velocity[3];
    double universal_parameter;
    double transfer_angle_radians;
    uint32_t iterations;
    double time_of_flight_residual;
} spacepdhcg_orbitweaver_lambert_result;

typedef struct spacepdhcg_orbitweaver_lambert_config {
    uint32_t abi_version;
    uint32_t device_id;
    size_t maximum_batch_size;
    uint32_t supported_maximum_revolutions;
    uint32_t scan_samples_per_band;
} spacepdhcg_orbitweaver_lambert_config;

typedef struct spacepdhcg_orbitweaver_batch_telemetry {
    uint32_t abi_version;
    uint64_t batches_submitted;
    uint64_t requests_submitted;
    uint64_t results_emitted;
    uint64_t feasible_results;
    uint64_t failed_results;
    uint64_t input_bytes;
    uint64_t output_bytes;
    size_t workspace_bytes;
    size_t maximum_batch_size;
    int32_t device_id;
} spacepdhcg_orbitweaver_batch_telemetry;

/*
 * Fixed per-input stride: short/long directions each own one zero-revolution
 * slot and two slots per supported positive revolution.
 */
size_t spacepdhcg_orbitweaver_lambert_result_stride(
    uint32_t supported_maximum_revolutions
);

spacepdhcg_cuda_status spacepdhcg_orbitweaver_lambert_workspace_create(
    const spacepdhcg_orbitweaver_lambert_config* config,
    spacepdhcg_accelerator_stream stream,
    spacepdhcg_orbitweaver_lambert_workspace** workspace
);

spacepdhcg_cuda_status spacepdhcg_orbitweaver_lambert_evaluate_async(
    spacepdhcg_orbitweaver_lambert_workspace* workspace,
    const spacepdhcg_orbitweaver_lambert_request* requests,
    size_t request_count,
    spacepdhcg_orbitweaver_lambert_result* results,
    size_t result_capacity,
    spacepdhcg_accelerator_stream stream
);

spacepdhcg_cuda_status spacepdhcg_orbitweaver_lambert_workspace_telemetry(
    const spacepdhcg_orbitweaver_lambert_workspace* workspace,
    spacepdhcg_orbitweaver_batch_telemetry* telemetry
);

spacepdhcg_cuda_status spacepdhcg_orbitweaver_lambert_workspace_cancel(
    spacepdhcg_orbitweaver_lambert_workspace* workspace
);

spacepdhcg_cuda_status spacepdhcg_orbitweaver_lambert_workspace_destroy(
    spacepdhcg_orbitweaver_lambert_workspace** workspace
);

#ifdef __cplusplus
}
#endif
