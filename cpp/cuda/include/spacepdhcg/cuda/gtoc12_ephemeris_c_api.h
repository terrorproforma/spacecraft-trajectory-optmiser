#pragma once
#include "spacepdhcg/cuda/orbitweaver_gpu_c_api.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct spacepdhcg_gtoc12_ephemeris spacepdhcg_gtoc12_ephemeris;
typedef struct spacepdhcg_gtoc12_ephemeris_request {
    int32_t body_index;
    int32_t reserved; /* Must be zero. */
    double epoch_mjd;
} spacepdhcg_gtoc12_ephemeris_request;
typedef struct spacepdhcg_gtoc12_ephemeris_result {
    double position_km[3], velocity_km_s[3];
    int32_t status; /* 0: finite converged state; 1: invalid request; 2: failed Kepler solve. */
    int32_t reserved;
} spacepdhcg_gtoc12_ephemeris_result;

/* Copies immutable orbital elements once. Owns retained request/result buffers.
 * Calls belong to the creating host thread and CUDA device. */
spacepdhcg_cuda_status spacepdhcg_gtoc12_ephemeris_create(
    const spacepdhcg_orbitweaver_elements* bodies, int32_t body_count,
    int32_t capacity, double mu, int32_t device, spacepdhcg_gtoc12_ephemeris** workspace);

/* Allocation/copy/synchronization-free launch on caller-owned device buffers.
 * Caller serializes workspace use and completes all external streams/graphs
 * before destruction. Invalid rows produce NaN states and a nonzero row status.
 * count==0 is a no-op; otherwise both buffers contain at least count rows. */
spacepdhcg_cuda_status spacepdhcg_gtoc12_ephemeris_launch_device(
    spacepdhcg_gtoc12_ephemeris* workspace,
    const spacepdhcg_gtoc12_ephemeris_request* requests, int32_t count,
    spacepdhcg_gtoc12_ephemeris_result* results, spacepdhcg_accelerator_stream stream);

/* Blocking bridge for existing host boundary consumers: one request upload and
 * one result download per batch. All orbital-state arithmetic remains on CUDA. */
spacepdhcg_cuda_status spacepdhcg_gtoc12_ephemeris_host(
    spacepdhcg_gtoc12_ephemeris* workspace,
    const spacepdhcg_gtoc12_ephemeris_request* requests, int32_t count,
    spacepdhcg_gtoc12_ephemeris_result* results);
spacepdhcg_cuda_status spacepdhcg_gtoc12_ephemeris_destroy(spacepdhcg_gtoc12_ephemeris** workspace);

#ifdef __cplusplus
}
#endif
