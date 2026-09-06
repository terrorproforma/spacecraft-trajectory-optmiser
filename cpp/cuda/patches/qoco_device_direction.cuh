// SPDX-License-Identifier: Apache-2.0
// The original recovery branch checks NaN, not infinity. Preserve that contract.
namespace qoco_device_direction {
__global__ void scan(const double* direction, int count, int* invalid) {
    int found = 0;
    for (long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < count; i += static_cast<long long>(blockDim.x) * gridDim.x)
        found |= isnan(direction[i]);
    if (__syncthreads_or(found) && threadIdx.x == 0) atomicExch(invalid, 1);
}
}
#ifndef SPACEPDHCG_DIRECTION_KERNEL_TEST
extern "C" const int* qoco_gpu_direction_flag(QOCOWorkspace* w) {
    if (!w->gpu_control) return nullptr;
    return &static_cast<qoco_device_control::State*>(w->gpu_control)->invalid_direction;
}
extern "C" void qoco_gpu_scan_direction(QOCOWorkspace* w) {
    if (!w->gpu_control) return;
    auto* invalid = &static_cast<qoco_device_control::State*>(w->gpu_control)->invalid_direction;
    CUDA_CHECK(cudaMemsetAsync(invalid, 0, sizeof(int)));
    const int count = w->xyz->len;
    if (count) qoco_device_direction::scan<<<std::min(256, (count - 1) / 256 + 1), 256>>>(
        w->xyz->d_data, count, invalid);
    CUDA_CHECK(cudaGetLastError());
}
#endif
