// SPDX-License-Identifier: Apache-2.0
// Experimental static-degree ordering (not AMD): count symmetric graph edges
// on CUDA and stably sort vertices. Numerical values do not affect the order.
#include <cub/device/device_radix_sort.cuh>
__global__ void qoco_ordering_degrees(int n, const int* offsets, const int* columns,
    int* degree, int* identity) {
    for (long long row = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
         row < n; row += static_cast<long long>(blockDim.x) * gridDim.x) {
        identity[row] = row;
        int count = 0;
        for (int k = offsets[row]; k < offsets[row + 1]; ++k) {
            const int col = columns[k];
            if (col != row) { ++count; atomicAdd(degree + col, 1); }
        }
        atomicAdd(degree + row, count);
    }
}
extern "C" void qoco_gpu_degree_ordering(int n, const int* offsets, const int* columns, int* permutation) {
    if (n <= 0) return;
    int *degree{}, *sorted{}, *identity{};
    const size_t bytes = static_cast<size_t>(n) * sizeof(int);
    CUDA_CHECK(cudaMalloc(&degree, bytes)); CUDA_CHECK(cudaMalloc(&sorted, bytes));
    CUDA_CHECK(cudaMalloc(&identity, bytes)); CUDA_CHECK(cudaMemsetAsync(degree, 0, bytes));
    qoco_ordering_degrees<<<std::min(256, (n - 1) / 256 + 1), 256>>>(n, offsets, columns, degree, identity);
    CUDA_CHECK(cudaGetLastError());
    size_t scratch_bytes{}; void* scratch{};
    CUDA_CHECK(cub::DeviceRadixSort::SortPairs(nullptr, scratch_bytes, degree, sorted, identity, permutation, n));
    CUDA_CHECK(cudaMalloc(&scratch, scratch_bytes));
    CUDA_CHECK(cub::DeviceRadixSort::SortPairs(scratch, scratch_bytes, degree, sorted, identity, permutation, n));
    CUDA_CHECK(cudaFree(scratch)); CUDA_CHECK(cudaFree(identity));
    CUDA_CHECK(cudaFree(sorted)); CUDA_CHECK(cudaFree(degree));
}
