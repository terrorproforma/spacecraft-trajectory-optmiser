#pragma once
#include "spacepdhcg/cuda/persistent_pdhcg_c_api.h"
#include "spacepdhcg/accelerator_c_api.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif
typedef struct spacepdhcg_gtoc12_collection spacepdhcg_gtoc12_collection;
typedef struct spacepdhcg_gtoc12_collection_options spacepdhcg_gtoc12_collection_options;
typedef struct spacepdhcg_gtoc12_collection_option {
    double delta_v, departure, tof;
} spacepdhcg_gtoc12_collection_option;
typedef struct spacepdhcg_gtoc12_collection_query {
    /* mode 0: collection cost, mode 1: first feasible option in input order. */
    int32_t mode, ratio_inflation;
    double mass, epoch, max_span, authority_ratio;
    double inflation, floor, slope, wait_penalty, penalty_scale;
    double thrust, day_seconds, year_days, mining_rate, exhaust_velocity;
} spacepdhcg_gtoc12_collection_query;
typedef struct spacepdhcg_gtoc12_collection_result {
    /* index=-1: no feasible option. status=1: invalid query or option. */
    int32_t index, status;
    double cost;
} spacepdhcg_gtoc12_collection_result;

spacepdhcg_cuda_status spacepdhcg_gtoc12_collection_create(
    int32_t capacity, int32_t device, spacepdhcg_gtoc12_collection** workspace);
/* Caller-owned device buffers. 0 <= count <= capacity. Retained scratch;
 * supports graph capture without host copies/allocations. Caller serializes
 * workspace use, graph replay and destruction, on the workspace device. */
spacepdhcg_cuda_status spacepdhcg_gtoc12_collection_launch_device(
    spacepdhcg_gtoc12_collection* workspace,
    const spacepdhcg_gtoc12_collection_option* options, int32_t count,
    const spacepdhcg_gtoc12_collection_query* query,
    spacepdhcg_gtoc12_collection_result* result, spacepdhcg_accelerator_stream stream);
/* Blocking bridge; transfers option rows and query, downloads one result. */
spacepdhcg_cuda_status spacepdhcg_gtoc12_collection_host(
    spacepdhcg_gtoc12_collection* workspace,
    const spacepdhcg_gtoc12_collection_option* options, int32_t count,
    const spacepdhcg_gtoc12_collection_query* query,
    spacepdhcg_gtoc12_collection_result* result);
spacepdhcg_cuda_status spacepdhcg_gtoc12_collection_destroy(
    spacepdhcg_gtoc12_collection** workspace);
/* Immutable resident option tables belong to their creating thread/device.
 * Blocking operations; caller retains the table until selection completes.
 * read is an explicit host-consumer bridge, never needed for device selection. */
spacepdhcg_cuda_status spacepdhcg_gtoc12_collection_options_read(
    spacepdhcg_gtoc12_collection_options* table,
    spacepdhcg_gtoc12_collection_option* rows, int32_t capacity);
spacepdhcg_cuda_status spacepdhcg_gtoc12_collection_options_destroy(
    spacepdhcg_gtoc12_collection_options** table);
spacepdhcg_cuda_status spacepdhcg_gtoc12_collection_resident(
    spacepdhcg_gtoc12_collection* workspace, spacepdhcg_gtoc12_collection_options* table,
    const spacepdhcg_gtoc12_collection_query* query,
    spacepdhcg_gtoc12_collection_result* result,
    spacepdhcg_gtoc12_collection_option* selected);
#ifdef __cplusplus
}
#endif
