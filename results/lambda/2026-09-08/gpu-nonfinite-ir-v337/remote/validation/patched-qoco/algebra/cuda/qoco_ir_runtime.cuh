// SPDX-License-Identifier: Apache-2.0
// Optional GPU-controlled refinement. Included after the cuDSS function table.
#include <vector>
#include <cstring>
#include <cstdint>

#include "qoco_factor_runtime.cuh"

#include "qoco_ir_cache.cuh"

struct QocoIrState {
    double best;
    int accepted, attempted, restore;
};
#include "qoco_ipm_cache.cuh"

struct QocoIrRuntime {
    QocoIpmCache ipm;
    QocoIrCache cache;
    QocoIrCounts* counts{};
    QocoFactorMemory* factor_memory{};
    cudaStream_t stream{};
    cudaEvent_t ready{}, done{};
    double* norm_scratch{};
    QocoIrState* state{};
    void* range_inputs[2]{};
};

__global__ void qoco_ir_initialize_ranges(uint64_t* first, uint64_t* second, int count) {
    *first = *second = uint64_t(count - 1) << 32;
}
static QocoIrRuntime* qoco_ir_create(cudssHandle_t handle, int count) {
    auto* runtime = new QocoIrRuntime;
    CUDA_CHECK(cudaMalloc(&runtime->counts, sizeof(QocoIrCounts)));
    CUDA_CHECK(cudaMemset(runtime->counts, 0, sizeof(QocoIrCounts)));
    CUDA_CHECK(cudaMalloc(&runtime->cache.parameters, sizeof(QocoIrParameters)));
    CUDA_CHECK(cudaStreamCreateWithFlags(&runtime->stream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaEventCreateWithFlags(&runtime->ready, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&runtime->done, cudaEventDisableTiming));
    CUDA_CHECK(cudaMalloc(&runtime->norm_scratch, 257 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&runtime->state, sizeof(QocoIrState)));
    CUDA_CHECK(cudaMemset(runtime->state, 0, sizeof(QocoIrState)));
    for (auto& input : runtime->range_inputs) CUDA_CHECK(cudaMalloc(&input, 8));
    qoco_ir_initialize_ranges<<<1, 1, 0, runtime->stream>>>(
        static_cast<uint64_t*>(runtime->range_inputs[0]),
        static_cast<uint64_t*>(runtime->range_inputs[1]), count);
    CUDA_CHECK(cudaGetLastError());
    auto set_stream = reinterpret_cast<decltype(&::cudssSetStream)>(dlsym(g_cudss_handle, "cudssSetStream"));
    if (!set_stream) { fprintf(stderr, "cuDSS stream API unavailable\n"); exit(1); }
    CUDSS_CHECK(set_stream(handle, runtime->stream));
    runtime->factor_memory = qoco_factor_memory_create(handle);
    return runtime;
}
static void qoco_ir_destroy(QocoIrRuntime* runtime) {
    CUDA_CHECK(cudaStreamSynchronize(runtime->stream));
    if (getenv("SPACEPDHCG_TEST_QOCO_IR_CACHE_TRACE"))
        fprintf(stderr, "IR_CACHE builds=%zu launches=%zu\n", runtime->cache.builds, runtime->cache.launches);
    qoco_ir_cache_clear(runtime->cache);
    CUDA_CHECK(cudaFree(runtime->counts));
    CUDA_CHECK(cudaFree(runtime->cache.parameters));
    for (auto input : runtime->range_inputs) CUDA_CHECK(cudaFree(input));
    CUDA_CHECK(cudaFree(runtime->norm_scratch));
    CUDA_CHECK(cudaFree(runtime->state));
    CUDA_CHECK(cudaEventDestroy(runtime->ready));
    CUDA_CHECK(cudaEventDestroy(runtime->done));
    CUDA_CHECK(cudaStreamDestroy(runtime->stream));
    qoco_factor_memory_destroy(runtime->factor_memory);
    delete runtime;
}
static void qoco_ir_wait_default(QocoIrRuntime* runtime) {
    CUDA_CHECK(cudaEventRecord(runtime->ready, nullptr));
    CUDA_CHECK(cudaStreamWaitEvent(runtime->stream, runtime->ready));
}
static cudssStatus_t qoco_ir_execute(QocoIrRuntime* runtime, cudssHandle_t handle, int phase,
        const cudssConfig_t config, cudssData_t data, const cudssMatrix_t matrix,
        cudssMatrix_t solution, const cudssMatrix_t rhs) {
    qoco_ir_wait_default(runtime);
    const auto result = qoco_factor_execute(runtime->factor_memory, runtime->stream, handle, phase, config, data, matrix, solution, rhs);
    CUDA_CHECK(cudaEventRecord(runtime->done, runtime->stream));
    CUDA_CHECK(cudaStreamWaitEvent(nullptr, runtime->done));
    return result;
}

extern "C" cudaStream_t qoco_ir_exchange_stream(cudaStream_t);
extern "C" void qoco_ir_device_norm(const double*, int, double*, cudaStream_t);
extern "C" void qoco_ir_prepare_sparse(QOCOProblemData*);

__global__ void qoco_ir_initial(QocoIrState* state, const double* norm,
        cudaGraphConditionalHandle handle, const QocoIrParameters* parameters) {
    const double tolerance = parameters->tolerance;
    const int maximum = parameters->maximum;
    state->best = *norm;
    state->accepted = state->attempted = state->restore = 0;
    cudaGraphSetConditional(handle, maximum > 0 && isfinite(*norm) && isfinite(tolerance) && !(*norm < tolerance));
}
__global__ void qoco_ir_decide(QocoIrState* state, const double* norm,
        cudaGraphConditionalHandle handle, const QocoIrParameters* parameters) {
    const double tolerance = parameters->tolerance;
    const int maximum = parameters->maximum;
    ++state->attempted;
    if (!isfinite(*norm) || *norm >= state->best) {
        state->restore = 1;
        cudaGraphSetConditional(handle, 0);
    } else {
        ++state->accepted;
        state->best = *norm;
        state->restore = 0;
        cudaGraphSetConditional(handle, state->attempted < maximum && !(*norm < tolerance));
    }
}
__global__ void qoco_ir_save_or_restore(const QocoIrState* state, double* x, double* best, int count) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) {
        if (state->restore) x[i] = best[i];
        else best[i] = x[i];
    }
}
