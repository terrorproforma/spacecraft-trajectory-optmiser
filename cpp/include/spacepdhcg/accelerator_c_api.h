#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define SPACEPDHCG_ACCELERATOR_EXCHANGE_ABI_VERSION 1U

typedef enum spacepdhcg_accelerator_device_type {
    SPACEPDHCG_DEVICE_CPU = 1,
    SPACEPDHCG_DEVICE_CUDA = 2,
    SPACEPDHCG_DEVICE_CUDA_HOST = 3,
    SPACEPDHCG_DEVICE_CUDA_MANAGED = 13
} spacepdhcg_accelerator_device_type;

typedef enum spacepdhcg_accelerator_scalar_type {
    SPACEPDHCG_SCALAR_INT32 = 0,
    SPACEPDHCG_SCALAR_INT64 = 1,
    SPACEPDHCG_SCALAR_FLOAT32 = 2,
    SPACEPDHCG_SCALAR_FLOAT64 = 3
} spacepdhcg_accelerator_scalar_type;

typedef enum spacepdhcg_accelerator_access {
    SPACEPDHCG_ACCESS_READ_ONLY = 0,
    SPACEPDHCG_ACCESS_READ_WRITE = 1
} spacepdhcg_accelerator_access;

typedef struct spacepdhcg_accelerator_device {
    spacepdhcg_accelerator_device_type type;
    int32_t id;
} spacepdhcg_accelerator_device;

typedef struct spacepdhcg_accelerator_stream {
    spacepdhcg_accelerator_device device;
    uintptr_t native_handle;
} spacepdhcg_accelerator_stream;

typedef struct spacepdhcg_accelerator_buffer_view {
    void* data;
    spacepdhcg_accelerator_device device;
    spacepdhcg_accelerator_scalar_type scalar_type;
    size_t elements;
    size_t byte_offset;
    ptrdiff_t element_stride;
    spacepdhcg_accelerator_access access;
} spacepdhcg_accelerator_buffer_view;

typedef struct spacepdhcg_cqp_topology_accelerator_views {
    spacepdhcg_accelerator_buffer_view quadratic_offsets;
    spacepdhcg_accelerator_buffer_view quadratic_indices;
    spacepdhcg_accelerator_buffer_view scalar_offsets;
    spacepdhcg_accelerator_buffer_view scalar_indices;
    spacepdhcg_accelerator_buffer_view affine_offsets;
    spacepdhcg_accelerator_buffer_view affine_indices;
} spacepdhcg_cqp_topology_accelerator_views;

typedef struct spacepdhcg_cqp_numeric_accelerator_views {
    spacepdhcg_accelerator_buffer_view quadratic;
    spacepdhcg_accelerator_buffer_view scalar_constraint;
    spacepdhcg_accelerator_buffer_view affine_cone;
    spacepdhcg_accelerator_buffer_view linear_objective;
    spacepdhcg_accelerator_buffer_view scalar_lower;
    spacepdhcg_accelerator_buffer_view scalar_upper;
    spacepdhcg_accelerator_buffer_view affine_offset;
    spacepdhcg_accelerator_buffer_view variable_lower;
    spacepdhcg_accelerator_buffer_view variable_upper;
} spacepdhcg_cqp_numeric_accelerator_views;

typedef struct spacepdhcg_cqp_iterate_accelerator_views {
    spacepdhcg_accelerator_buffer_view primal;
    spacepdhcg_accelerator_buffer_view dual;
} spacepdhcg_cqp_iterate_accelerator_views;

typedef struct spacepdhcg_cqp_accelerator_exchange {
    uint32_t abi_version;
    uint64_t topology_fingerprint;
    spacepdhcg_accelerator_stream consumer_stream;
    spacepdhcg_cqp_topology_accelerator_views topology;
    spacepdhcg_cqp_numeric_accelerator_views numeric;
    spacepdhcg_cqp_iterate_accelerator_views iterates;
} spacepdhcg_cqp_accelerator_exchange;

#ifdef __cplusplus
}
#endif
