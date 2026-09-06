#pragma once

#include <cuda_runtime.h>
#include <algorithm>
#include <cstddef>

namespace spacepdhcg::cuda::detail {

inline unsigned scvx_gather_blocks(std::size_t states, std::size_t controls, bool single_block = false) {
    if (single_block) return 1;
    const auto count = std::max(states, controls);
    return static_cast<unsigned>(std::min<std::size_t>(1024, count ? (count - 1) / 256 + 1 : 1));
}

// Each output has one writer. State and control index arrays may contain
// repeated source indices; output buffers must not alias the source or each other.
__global__ void gather_scvx_candidate_kernel(
    const double* primal, const int* state_indices, const int* control_indices,
    double* states, double* controls, const std::size_t state_elements,
    const std::size_t control_elements
) {
    const auto rank = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const auto stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (auto index = rank; index < state_elements; index += stride)
        states[index] = primal[state_indices[index]];
    for (auto index = rank; index < control_elements; index += stride)
        controls[index] = primal[control_indices[index]];
}

}  // namespace spacepdhcg::cuda::detail
