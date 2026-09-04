/*
 * SpacePDHCG persistent CUDA DLPack ingestion C ABI.
 *
 * The managed_tensor field points to either the standard legacy
 * DLManagedTensor or DLManagedTensorVersioned ABI. On every call which accepts
 * this structure, ownership is consumed and the producer deleter is called
 * exactly once, including validation failures.
 *
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "spacepdhcg/cuda/persistent_pdhcg_c_api.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum spacepdhcg_dlpack_managed_kind {
    SPACEPDHCG_DLPACK_LEGACY = 0,
    SPACEPDHCG_DLPACK_VERSIONED = 1
} spacepdhcg_dlpack_managed_kind;

typedef struct spacepdhcg_dlpack_managed_tensor {
    void* managed_tensor;
    spacepdhcg_dlpack_managed_kind kind;
    spacepdhcg_accelerator_access access;
} spacepdhcg_dlpack_managed_tensor;

typedef struct spacepdhcg_cqp_topology_dlpack_tensors {
    spacepdhcg_dlpack_managed_tensor quadratic_offsets;
    spacepdhcg_dlpack_managed_tensor quadratic_indices;
    spacepdhcg_dlpack_managed_tensor scalar_offsets;
    spacepdhcg_dlpack_managed_tensor scalar_indices;
    spacepdhcg_dlpack_managed_tensor affine_offsets;
    spacepdhcg_dlpack_managed_tensor affine_indices;
} spacepdhcg_cqp_topology_dlpack_tensors;

typedef struct spacepdhcg_cqp_numeric_dlpack_tensors {
    spacepdhcg_dlpack_managed_tensor quadratic;
    spacepdhcg_dlpack_managed_tensor scalar_constraint;
    spacepdhcg_dlpack_managed_tensor affine_cone;
    spacepdhcg_dlpack_managed_tensor linear_objective;
    spacepdhcg_dlpack_managed_tensor scalar_lower;
    spacepdhcg_dlpack_managed_tensor scalar_upper;
    spacepdhcg_dlpack_managed_tensor affine_offset;
    spacepdhcg_dlpack_managed_tensor variable_lower;
    spacepdhcg_dlpack_managed_tensor variable_upper;
} spacepdhcg_cqp_numeric_dlpack_tensors;

typedef struct spacepdhcg_cqp_iterate_dlpack_tensors {
    spacepdhcg_dlpack_managed_tensor primal;
    spacepdhcg_dlpack_managed_tensor dual;
} spacepdhcg_cqp_iterate_dlpack_tensors;

typedef struct spacepdhcg_cqp_dlpack_exchange {
    uint32_t abi_version;
    uint64_t topology_fingerprint;
    spacepdhcg_accelerator_stream consumer_stream;
    spacepdhcg_cqp_topology_dlpack_tensors topology;
    spacepdhcg_cqp_numeric_dlpack_tensors numeric;
    spacepdhcg_cqp_iterate_dlpack_tensors iterates;
} spacepdhcg_cqp_dlpack_exchange;

spacepdhcg_cuda_status spacepdhcg_cuda_workspace_create_from_dlpack(
    const spacepdhcg_cuda_structure* structure,
    const spacepdhcg_cqp_dlpack_exchange* exchange,
    const spacepdhcg_cuda_create_options* options,
    spacepdhcg_cuda_workspace** workspace
);

spacepdhcg_cuda_status spacepdhcg_cuda_workspace_update_from_dlpack_async(
    spacepdhcg_cuda_workspace* workspace,
    uint64_t topology_fingerprint,
    const spacepdhcg_cqp_numeric_dlpack_tensors* values,
    spacepdhcg_accelerator_stream stream
);

spacepdhcg_cuda_status spacepdhcg_cuda_workspace_warm_start_from_dlpack_async(
    spacepdhcg_cuda_workspace* workspace,
    spacepdhcg_cuda_warm_start_mode mode,
    const spacepdhcg_cqp_iterate_dlpack_tensors* iterates,
    spacepdhcg_accelerator_stream stream
);

#ifdef __cplusplus
}
#endif
