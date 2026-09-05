// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cuda_runtime.h>
#include <cstddef>

namespace spacepdhcg::cuda::detail {

static __device__ unsigned long long mix_numeric_word(
    unsigned long long value,
    const unsigned long long index
) {
    value ^= index + 0x9e3779b97f4a7c15ULL + (value << 6U) + (value >> 2U);
    value ^= value >> 30U;
    value *= 0xbf58476d1ce4e5b9ULL;
    value ^= value >> 27U;
    value *= 0x94d049bb133111ebULL;
    return value ^ (value >> 31U);
}

static __global__ void hash_numeric_kernel(
    const double* values,
    const size_t elements,
    const unsigned long long tag,
    unsigned long long* fingerprint
) {
    unsigned long long local = 0ULL;
    for (size_t index = blockIdx.x * blockDim.x + threadIdx.x;
         index < elements;
         index += blockDim.x * gridDim.x) {
        local ^= mix_numeric_word(
            static_cast<unsigned long long>(__double_as_longlong(values[index])),
            tag ^ static_cast<unsigned long long>(index)
        );
    }
    // XOR is bitwise associative: reduce locally without changing the fingerprint.
    // One atomic per block avoids serializing every numeric-value thread on one word.
    __shared__ unsigned long long warp_partials[8];
    const unsigned int lane = threadIdx.x % 32U;
    const unsigned int warp = threadIdx.x / 32U;
    for (int offset = 16; offset > 0; offset >>= 1) {
        local ^= __shfl_down_sync(0xffffffffU, local, offset);
    }
    if (lane == 0U) warp_partials[warp] = local;
    __syncthreads();
    if (warp == 0U) {
        local = lane < 8U ? warp_partials[lane] : 0ULL;
        for (int offset = 16; offset > 0; offset >>= 1) {
            local ^= __shfl_down_sync(0xffffffffU, local, offset);
        }
        if (lane == 0U) atomicXor(fingerprint, local);
    }
}

}  // namespace spacepdhcg::cuda::detail
