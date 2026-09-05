// SPDX-License-Identifier: Apache-2.0
// Included after the pinned cone kernels. Numerical cone formulas are unchanged.
extern "C" cudaError_t qoco_gpu_acquire_scalar_workspace(size_t, double**, int*);
static_assert(sizeof(QOCOFloat) == sizeof(double), "cone workspace requires FP64");

template<bool Maximum>
__global__ void qoco_finish_cone_reduction(const QOCOFloat* input, int count,
                                          QOCOFloat* result, const QOCOFloat* initial,
                                          QOCOFloat factor, bool finalize)
{
    __shared__ QOCOFloat scratch[256];
    QOCOFloat value = Maximum ? -1e7 : (initial ? *initial : 1.0);
    for (int i = threadIdx.x; i < count; i += blockDim.x) {
        if (Maximum ? input[i] > value : input[i] < value) value = input[i];
    }
    scratch[threadIdx.x] = value;
    __syncthreads();
    for (int step = 128; step; step /= 2) {
        if (threadIdx.x < step) {
            QOCOFloat other = scratch[threadIdx.x + step];
            if (Maximum ? other > scratch[threadIdx.x] : other < scratch[threadIdx.x])
                scratch[threadIdx.x] = other;
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        value = scratch[0];
        *result = finalize ? (value < 1e-12 ? 0.0 : factor * value) : value;
    }
}

__global__ void qoco_soc_steps(const QOCOFloat* u, const QOCOFloat* direction,
                              const QOCOInt* sizes, const QOCOInt* starts, int count,
                              const QOCOFloat* lp_step, QOCOFloat* partial)
{
    __shared__ QOCOFloat scratch[256];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    scratch[threadIdx.x] = i < count
        ? soc_step_length_dev(u + starts[i], direction + starts[i], sizes[i], *lp_step)
        : QOCOFloat_MAX;
    __syncthreads();
    for (int step = 128; step; step /= 2) {
        if (threadIdx.x < step && scratch[threadIdx.x + step] < scratch[threadIdx.x])
            scratch[threadIdx.x] = scratch[threadIdx.x + step];
        __syncthreads();
    }
    if (threadIdx.x == 0) partial[blockIdx.x] = scratch[0];
}

QOCOFloat linesearch(QOCOFloat* u, QOCOFloat* direction, QOCOFloat factor, QOCOSolver* solver)
{
    const auto* work = solver->work;
    const auto* data = work->data;
    const int lp_blocks = (data->l + 255) / 256;
    const int soc_blocks = (data->nsoc + 255) / 256;
    QOCOFloat* storage{};
    int temporary{};
    CUDA_CHECK(qoco_gpu_acquire_scalar_workspace(
        static_cast<size_t>(max(lp_blocks, soc_blocks)) + 2, &storage, &temporary));
    auto* lp_step = storage;
    auto* output = storage + 1;
    auto* partial = storage + 2;
    if (lp_blocks) {
        lp_step_length_kernel<<<lp_blocks, 256, 256 * sizeof(QOCOFloat)>>>(
            u, direction, data->l, partial);
        CUDA_CHECK(cudaGetLastError());
    }
    qoco_finish_cone_reduction<false><<<1, 256>>>(partial, lp_blocks, lp_step, nullptr, 1.0, false);
    CUDA_CHECK(cudaGetLastError());
    if (soc_blocks) {
        qoco_soc_steps<<<soc_blocks, 256>>>(u, direction, get_data_vectori(data->q),
            get_data_vectori(work->soc_idx), data->nsoc, lp_step, partial);
        CUDA_CHECK(cudaGetLastError());
    }
    qoco_finish_cone_reduction<false><<<1, 256>>>(partial, soc_blocks, output, lp_step, factor, true);
    CUDA_CHECK(cudaGetLastError());
    QOCOFloat result{};
    CUDA_CHECK(cudaMemcpy(&result, output, sizeof(result), cudaMemcpyDeviceToHost));
    if (temporary) CUDA_CHECK(cudaFree(storage));
    return result;
}

QOCOFloat cone_residual(const QOCOFloat* u, QOCOInt l, QOCOInt nsoc,
                        const QOCOInt* sizes, QOCOInt* starts)
{
    const int count = l + nsoc;
    if (!count) return -1e7;
    const int blocks = (count + 255) / 256;
    QOCOFloat* storage{};
    int temporary{};
    CUDA_CHECK(qoco_gpu_acquire_scalar_workspace(static_cast<size_t>(blocks) + 1, &storage, &temporary));
    cone_residual_stage1<<<blocks, 256, 256 * sizeof(QOCOFloat)>>>(u, l, nsoc, sizes, starts, storage + 1);
    CUDA_CHECK(cudaGetLastError());
    // Grid-stride final reduction covers every partial, including grids >1024.
    qoco_finish_cone_reduction<true><<<1, 256>>>(storage + 1, blocks, storage, nullptr, 1.0, false);
    CUDA_CHECK(cudaGetLastError());
    QOCOFloat result{};
    CUDA_CHECK(cudaMemcpy(&result, storage, sizeof(result), cudaMemcpyDeviceToHost));
    if (temporary) CUDA_CHECK(cudaFree(storage));
    return result;
}
