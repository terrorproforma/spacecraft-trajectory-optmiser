// SPDX-License-Identifier: Apache-2.0
// Included in the prepared CUDA algebra before inf_norm. All operations share
// QOCO's default stream and finish before releasing/reusing scalar scratch.
#include <algorithm>
#include <math_constants.h>
namespace qoco_device_scalar {
enum Operation { Maximum, Minimum, HasNan };

template<Operation operation> __device__ double combine(double a, double b) {
    // Never silently hide a NaN while evaluating a convergence norm.
    if (isnan(a) || isnan(b)) return CUDART_NAN;
    if constexpr (operation == Minimum) return fmin(a, b);
    return fmax(a, b);
}

template<Operation operation, bool partial_input>
__global__ void reduce(const double* input, int count, double* output) {
    __shared__ double scratch[256];
    double value = operation == Minimum ? CUDART_INF : 0.0;
    for (long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < count; i += static_cast<long long>(gridDim.x) * blockDim.x) {
        double next = input[i];
        if constexpr (!partial_input) {
            if constexpr (operation == HasNan) next = isnan(next) ? 1.0 : 0.0;
            else next = fabs(next);
        }
        value = combine<operation>(value, next);
    }
    scratch[threadIdx.x] = value;
    __syncthreads();
    for (int stride = 128; stride; stride /= 2) {
        if (threadIdx.x < stride)
            scratch[threadIdx.x] = combine<operation>(scratch[threadIdx.x], scratch[threadIdx.x + stride]);
        __syncthreads();
    }
    if (threadIdx.x == 0) output[blockIdx.x] = scratch[0];
}

template<Operation operation> double run(const double* input, int count) {
    if (count == 0) return operation == Minimum ? QOCOFloat_MAX : 0.0;
    const int blocks = count <= 4096 ? 1 : std::min(256, (count - 1) / 256 + 1);
    double* scratch{};
    int temporary{};
    CUDA_CHECK(qoco_gpu_acquire_scalar_workspace(blocks + 1, &scratch, &temporary));
    reduce<operation, false><<<blocks, 256>>>(input, count, scratch);
    double* result = scratch;
    if (blocks > 1) {
        result += blocks;
        reduce<operation, true><<<1, 256>>>(scratch, blocks, result);
    }
    CUDA_CHECK(cudaGetLastError());
    double host{};
    CUDA_CHECK(cudaMemcpy(&host, result, sizeof(host), cudaMemcpyDeviceToHost));
    if (temporary) CUDA_CHECK(cudaFree(scratch));
    return host;
}
} // namespace qoco_device_scalar
