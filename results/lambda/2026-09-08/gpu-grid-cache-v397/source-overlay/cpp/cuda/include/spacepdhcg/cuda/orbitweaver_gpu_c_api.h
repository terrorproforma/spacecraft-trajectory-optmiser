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

/* Zero-revolution rendezvous screening. Both directions are always evaluated;
 * the Lambert include flags and maximum_revolutions are ignored. Units must be
 * consistent with the Lambert request. Equal total costs prefer short-way. */
typedef struct spacepdhcg_orbitweaver_hop_request {
    spacepdhcg_orbitweaver_lambert_request lambert;
    double departure_body_velocity[3];
    double arrival_body_velocity[3];
    double departure_allowance;
    double arrival_allowance;
} spacepdhcg_orbitweaver_hop_request;

typedef struct spacepdhcg_orbitweaver_hop_result {
    double departure_velocity[3];
    double arrival_velocity[3];
    double departure_delta_v;
    double arrival_delta_v;
    int32_t feasible;
    int32_t long_way;
} spacepdhcg_orbitweaver_hop_result;

/* Resident, graph-capturable combined Lambert/cost/selection operator. Failed
 * requests return feasible=0, infinite costs and NaN velocities. */
spacepdhcg_cuda_status spacepdhcg_orbitweaver_hop_launch_device(
    const spacepdhcg_orbitweaver_hop_request* requests, size_t count,
    uint32_t scan_samples, spacepdhcg_orbitweaver_hop_result* results,
    size_t result_capacity, spacepdhcg_accelerator_stream stream
);

/* Retained blocking bridge; all numerical screening and direction selection
 * run on CUDA. The host only packs/transfers requests and counts outputs. */
spacepdhcg_cuda_status spacepdhcg_orbitweaver_hop_screening_host(
    spacepdhcg_orbitweaver_lambert_workspace* workspace,
    const spacepdhcg_orbitweaver_hop_request* requests, size_t count,
    spacepdhcg_orbitweaver_hop_result* results, size_t result_capacity
);

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

/* Elliptic elements: km, radians, MJD. Mean anomaly is specified at epoch. */
typedef struct spacepdhcg_orbitweaver_elements {
    double epoch, a, e, inclination, node, perihelion, mean;
} spacepdhcg_orbitweaver_elements;

typedef struct spacepdhcg_orbitweaver_hop_elements {
    spacepdhcg_orbitweaver_elements departure, arrival;
    double gravitational_parameter, departure_allowance, arrival_allowance;
} spacepdhcg_orbitweaver_hop_elements;

/* Builds endpoint states and rendezvous requests on CUDA, then uses the same
 * Lambert operator as hop_screening_host. times contains count interleaved
 * (departure MJD, flight duration days) pairs. All host accesses finish before
 * return; workspace buffers are retained. No CPU ephemeris or CPU fallback. */
spacepdhcg_cuda_status spacepdhcg_orbitweaver_hop_elements_host(
    spacepdhcg_orbitweaver_lambert_workspace* workspace,
    const spacepdhcg_orbitweaver_hop_elements* elements,
    const double* times, size_t count,
    spacepdhcg_orbitweaver_hop_result* results, size_t result_capacity
);

/* Blocking resident grid bridge. Epochs, TOFs and output arrays are device
 * buffers on the workspace device. Only the fixed orbital elements are host
 * data. Writes departure-major scalar costs/flags, with no table download. */
spacepdhcg_cuda_status spacepdhcg_orbitweaver_hop_grid_device(
    spacepdhcg_orbitweaver_lambert_workspace* workspace,
    const spacepdhcg_orbitweaver_hop_elements* elements,
    const double* epochs, size_t epoch_count, const double* tofs, size_t tof_count,
    double* delta_v, uint8_t* feasible);

/* Blocking grid bridge with HOST epochs/TOFs and DEVICE output arrays. The
 * workspace retains at most 64 MiB of immutable device cost/flag tables in LRU
 * order. Keys compare every element and grid-axis byte; the workspace fixes the
 * scan configuration. Oversize tables are computed without retention. No table
 * is downloaded and changing the caller's input buffers cannot change old keys.
 * Axes must be finite, with positive flight durations. */
spacepdhcg_cuda_status spacepdhcg_orbitweaver_hop_grid_cached_host(
    spacepdhcg_orbitweaver_lambert_workspace* workspace,
    const spacepdhcg_orbitweaver_hop_elements* elements,
    const double* epochs,size_t epoch_count,const double* tofs,size_t tof_count,
    double* delta_v,uint8_t* feasible);
/* Host counters only; no synchronization or device transfer. */
spacepdhcg_cuda_status spacepdhcg_orbitweaver_hop_grid_cache_stats(
    spacepdhcg_orbitweaver_lambert_workspace* workspace,uint64_t* hits,uint64_t* misses,
    uint64_t* evictions,uint64_t* retained_device_bytes);

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

/* Complete the pending host-buffer transfer without polling from Python. */
spacepdhcg_cuda_status spacepdhcg_orbitweaver_lambert_workspace_finish(
    spacepdhcg_orbitweaver_lambert_workspace* workspace
);

/* Blocking screening batch with no asynchronous cancellation. Avoids managed
 * telemetry/control pages on the hot path; completes all host-buffer accesses
 * before returning. The existing cancellable asynchronous API is unchanged. */
spacepdhcg_cuda_status spacepdhcg_orbitweaver_lambert_screening_host(
    spacepdhcg_orbitweaver_lambert_workspace* workspace,
    const spacepdhcg_orbitweaver_lambert_request* requests, size_t request_count,
    spacepdhcg_orbitweaver_lambert_result* results, size_t result_capacity
);

/* Device-resident, graph-capturable operator. The caller owns all buffers and
 * provides result_stride(supported_maximum_revolutions) outputs per request.
 * No allocation, host transfer, cancellation flag or telemetry synchronization. */
spacepdhcg_cuda_status spacepdhcg_orbitweaver_lambert_launch_device(
    const spacepdhcg_orbitweaver_lambert_request* requests,
    size_t request_count, uint32_t supported_maximum_revolutions,
    uint32_t scan_samples_per_band,
    spacepdhcg_orbitweaver_lambert_result* results, size_t result_capacity,
    spacepdhcg_accelerator_stream stream
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
