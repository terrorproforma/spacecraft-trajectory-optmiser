#include "spacepdhcg/cuda/gtoc12_collection_c_api.h"
#include <cuda_runtime.h>
#include <cmath>
#include <mutex>
#include <new>

namespace {
using Option = spacepdhcg_gtoc12_collection_option;
using Query = spacepdhcg_gtoc12_collection_query;
using Result = spacepdhcg_gtoc12_collection_result;

__device__ bool valid_query(const Query& q) {
    return (q.mode == 0 || q.mode == 1) && (q.ratio_inflation == 0 || q.ratio_inflation == 1)
        && isfinite(q.mass) && q.mass > 0 && isfinite(q.epoch) && !isnan(q.max_span)
        && isfinite(q.authority_ratio) && q.authority_ratio >= 0
        && isfinite(q.inflation) && isfinite(q.floor) && isfinite(q.slope)
        && isfinite(q.wait_penalty) && isfinite(q.penalty_scale)
        && isfinite(q.thrust) && q.thrust > 0 && isfinite(q.day_seconds) && q.day_seconds > 0
        && isfinite(q.year_days) && q.year_days > 0 && isfinite(q.mining_rate) && q.mining_rate >= 0
        && isfinite(q.exhaust_velocity) && q.exhaust_velocity > 0;
}

__global__ void evaluate(const Option* options, int n, const Query* query,
                         double* costs, int* invalid) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    costs[i] = INFINITY;
    const auto q = *query;
    if (!valid_query(q)) { atomicExch(invalid, 1); return; }
    const auto o = options[i];
    if (!isfinite(o.delta_v) || o.delta_v < 0 || !isfinite(o.departure)
        || !isfinite(o.tof) || o.tof <= 0) { atomicExch(invalid, 1); return; }
    const double span = q.epoch - o.departure;
    if (q.mode == 0 && span > q.max_span) return;
    const double authority = (q.thrust / q.mass * 1e-3) * o.tof * q.day_seconds;
    if (!(o.delta_v <= q.authority_ratio * authority)) return;
    if (q.mode == 1) { costs[i] = 0; return; }
    // Match the reference's operation order, including the original exp form.
    if (span < 0 || !isfinite(span)) { atomicExch(invalid, 1); return; }
    const double inflation = q.ratio_inflation
        ? q.floor + q.slope * (o.delta_v / fmax(authority, 1e-12)) : q.inflation;
    const double lost = q.mining_rate * span / q.year_days;
    const double fuel = q.mass * (1.0 - exp(-(o.delta_v * inflation) / q.exhaust_velocity));
    const double cost = fuel + q.wait_penalty * q.penalty_scale * lost;
    if (!isfinite(cost)) { atomicExch(invalid, 1); return; }
    costs[i] = cost;
}

__global__ void select(const Option* options, int n, const Query* query,
                       const double* costs, const int* invalid, Result* result) {
    Result best{-1, 0, INFINITY};
    if (*invalid || !valid_query(*query)) { best.status = 1; *result = best; return; }
    // The 1e-12 near-tie comparator is not associative. Preserve its exact
    // input-order fold; only this cheap fold is serial, all option math is parallel.
    for (int i = 0; i < n; ++i) {
        const double cost = costs[i];
        if (!isfinite(cost)) continue;
        if (query->mode == 1) { best.index = i; best.cost = 0; break; }
        if (cost < best.cost - 1e-12 ||
            (fabs(cost - best.cost) <= 1e-12 && best.index >= 0 &&
             options[i].departure > options[best.index].departure)) {
            best.index = i; best.cost = cost;
        }
    }
    *result = best;
}

spacepdhcg_cuda_status mapped(cudaError_t s) {
    return s == cudaSuccess ? SPACEPDHCG_CUDA_SUCCESS :
        s == cudaErrorMemoryAllocation ? SPACEPDHCG_CUDA_OUT_OF_MEMORY : SPACEPDHCG_CUDA_RUNTIME_ERROR;
}
}

struct spacepdhcg_gtoc12_collection {
    int capacity{}, device{};
    Option* options{};
    Query* query{};
    Result* result{};
    double* costs{};
    int* invalid{};
    cudaStream_t stream{};
    std::mutex mutex;
};

extern "C" spacepdhcg_cuda_status spacepdhcg_gtoc12_collection_launch_device(
    spacepdhcg_gtoc12_collection* w, const Option* options, int n, const Query* query,
    Result* result, spacepdhcg_accelerator_stream stream) {
    if (!w || !options || !query || !result || n < 0 || n > w->capacity ||
        stream.device.type != SPACEPDHCG_DEVICE_CUDA || stream.device.id != w->device)
        return SPACEPDHCG_CUDA_INVALID_ARGUMENT;
    int device = -1; auto status = cudaGetDevice(&device);
    if (status != cudaSuccess) return mapped(status);
    if (device != w->device) return SPACEPDHCG_CUDA_INVALID_ARGUMENT;
    auto s = reinterpret_cast<cudaStream_t>(stream.native_handle);
    status = cudaMemsetAsync(w->invalid, 0, sizeof(int), s);
    if (status != cudaSuccess) return mapped(status);
    if (n) evaluate<<<(n - 1) / 128 + 1, 128, 0, s>>>(options, n, query, w->costs, w->invalid);
    status = cudaGetLastError();
    if (status != cudaSuccess) return mapped(status);
    select<<<1, 1, 0, s>>>(options, n, query, w->costs, w->invalid, result);
    return mapped(cudaGetLastError());
}

extern "C" spacepdhcg_cuda_status spacepdhcg_gtoc12_collection_destroy(spacepdhcg_gtoc12_collection** handle) {
    if (!handle || !*handle) return SPACEPDHCG_CUDA_INVALID_ARGUMENT;
    auto* w = *handle;
    std::unique_lock<std::mutex> lock(w->mutex, std::try_to_lock);
    if (!lock.owns_lock()) return SPACEPDHCG_CUDA_BUSY;
    int device = -1; auto status = cudaGetDevice(&device);
    if (status != cudaSuccess) return mapped(status);
    if (device != w->device) return SPACEPDHCG_CUDA_INVALID_ARGUMENT;
    status = cudaStreamSynchronize(w->stream);
    for (void* p : {static_cast<void*>(w->options), static_cast<void*>(w->query),
                   static_cast<void*>(w->result), static_cast<void*>(w->costs), static_cast<void*>(w->invalid)}) {
        const auto freed = cudaFree(p); if (status == cudaSuccess) status = freed;
    }
    if (w->stream) { const auto freed = cudaStreamDestroy(w->stream); if (status == cudaSuccess) status = freed; }
    lock.unlock(); delete w; *handle = nullptr;
    return mapped(status);
}

extern "C" spacepdhcg_cuda_status spacepdhcg_gtoc12_collection_create(
    int capacity, int device, spacepdhcg_gtoc12_collection** handle) {
    if (!handle || *handle || capacity < 1 || device < 0) return SPACEPDHCG_CUDA_INVALID_ARGUMENT;
    auto status = cudaSetDevice(device);
    if (status != cudaSuccess) return mapped(status);
    auto* w = new(std::nothrow) spacepdhcg_gtoc12_collection{};
    if (!w) return SPACEPDHCG_CUDA_OUT_OF_MEMORY;
    w->capacity = capacity; w->device = device;
    status = cudaStreamCreateWithFlags(&w->stream, cudaStreamNonBlocking);
#define ALLOC(field, bytes) if (status == cudaSuccess) status = cudaMalloc(reinterpret_cast<void**>(&w->field), (bytes))
    ALLOC(options, size_t(capacity) * sizeof(Option)); ALLOC(query, sizeof(Query));
    ALLOC(result, sizeof(Result)); ALLOC(costs, size_t(capacity) * sizeof(double)); ALLOC(invalid, sizeof(int));
#undef ALLOC
    if (status != cudaSuccess) { spacepdhcg_gtoc12_collection_destroy(&w); return mapped(status); }
    *handle = w; return SPACEPDHCG_CUDA_SUCCESS;
}

extern "C" spacepdhcg_cuda_status spacepdhcg_gtoc12_collection_host(
    spacepdhcg_gtoc12_collection* w, const Option* options, int n, const Query* query, Result* result) {
    if (!w || !options || !query || !result || n < 0 || n > w->capacity) return SPACEPDHCG_CUDA_INVALID_ARGUMENT;
    std::unique_lock<std::mutex> lock(w->mutex, std::try_to_lock);
    if (!lock.owns_lock()) return SPACEPDHCG_CUDA_BUSY;
    int device = -1; auto s = cudaGetDevice(&device);
    if (s != cudaSuccess) return mapped(s);
    if (device != w->device) return SPACEPDHCG_CUDA_INVALID_ARGUMENT;
    s = cudaMemcpyAsync(w->options, options, size_t(n) * sizeof(Option), cudaMemcpyHostToDevice, w->stream);
    if (s == cudaSuccess) s = cudaMemcpyAsync(w->query, query, sizeof(Query), cudaMemcpyHostToDevice, w->stream);
    auto result_status = mapped(s);
    if (s == cudaSuccess) result_status = spacepdhcg_gtoc12_collection_launch_device(
        w, w->options, n, w->query, w->result,
        {{SPACEPDHCG_DEVICE_CUDA, w->device}, reinterpret_cast<uintptr_t>(w->stream)});
    if (result_status == SPACEPDHCG_CUDA_SUCCESS)
        s = cudaMemcpyAsync(result, w->result, sizeof(Result), cudaMemcpyDeviceToHost, w->stream);
    const auto complete = cudaStreamSynchronize(w->stream);
    if (result_status != SPACEPDHCG_CUDA_SUCCESS) return result_status;
    return mapped(s == cudaSuccess ? complete : s);
}
