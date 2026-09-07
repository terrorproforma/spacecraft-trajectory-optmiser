#pragma once
#include "spacepdhcg/cuda/persistent_pdhcg_c_api.h"
#include "spacepdhcg/accelerator_c_api.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif
typedef struct spacepdhcg_gtoc12_neighbours spacepdhcg_gtoc12_neighbours;
typedef struct spacepdhcg_gtoc12_neighbour_body {
    int64_t id;
    double epoch, a, e, inclination, node, perihelion, mean;
} spacepdhcg_gtoc12_neighbour_body;
typedef struct spacepdhcg_gtoc12_neighbour_query {
    int32_t source_index, neighbours;
    double epoch, band_a, band_e, band_i, band_phase, filter_scale;
} spacepdhcg_gtoc12_neighbour_query;

/* Immutable catalogue/pool/TOF snapshot. Pool indices must refer to distinct
 * bodies in ascending asteroid-ID order. km, radians, MJD and days. */
spacepdhcg_cuda_status spacepdhcg_gtoc12_neighbours_create(
    const spacepdhcg_gtoc12_neighbour_body* bodies, int32_t body_count,
    const int32_t* pool, int32_t pool_count, const double* tofs, int32_t tof_count,
    double mu, double au, int32_t device, spacepdhcg_gtoc12_neighbours** workspace);

/* Serialized device-buffer operator on the workspace's device. Scratch is
 * retained; graph capture performs no allocation or host synchronization.
 * Output capacity must be pool_count. count=-1 reports an invalid query or
 * failed ephemeris calculation. Caller serializes use, including graph replay. */
spacepdhcg_cuda_status spacepdhcg_gtoc12_neighbours_launch_device(
    spacepdhcg_gtoc12_neighbours* workspace,
    const spacepdhcg_gtoc12_neighbour_query* query, int64_t* output, int32_t* count,
    spacepdhcg_accelerator_stream stream);

/* Blocking bridge. Capacity >= min(pool_count, neighbours+neighbours/2).
 * Downloads only that bounded prefix and its count. */
spacepdhcg_cuda_status spacepdhcg_gtoc12_neighbours_host(
    spacepdhcg_gtoc12_neighbours* workspace,
    const spacepdhcg_gtoc12_neighbour_query* query, int64_t* output,
    int32_t capacity, int32_t* count);
spacepdhcg_cuda_status spacepdhcg_gtoc12_neighbours_destroy(
    spacepdhcg_gtoc12_neighbours** workspace);
#ifdef __cplusplus
}
#endif
