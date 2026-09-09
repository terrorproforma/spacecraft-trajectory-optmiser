#include "spacepdhcg/cuda/gtoc12_collection_c_api.h"
#include "../internal/gtoc12_collection_options.h"
#include <cuda_runtime.h>
#include <cmath>
#include <mutex>
#include <new>
#include <thread>

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

struct SharedOptionRows { Option* rows{}; size_t references{1}; };
struct spacepdhcg_gtoc12_collection_options {
    Option* rows{};
    SharedOptionRows* storage{};
    int count{},device{};
    std::thread::id owner{std::this_thread::get_id()};
};

namespace {
bool owned(const spacepdhcg_gtoc12_collection_options* table) {
    int device=-1;
    return table && table->owner==std::this_thread::get_id()
        && cudaGetDevice(&device)==cudaSuccess && device==table->device;
}
__global__ void gather_selected(const Option* rows,const Result* result,Option* selected) {
    *selected=result->status==0 && result->index>=0?rows[result->index]:Option{};
}
}

spacepdhcg_cuda_status gtoc12_collection_options_copy_device(
    const Option* rows,int count,cudaStream_t stream,spacepdhcg_gtoc12_collection_options** output) {
    if(!rows || count<0 || !output || *output)return SPACEPDHCG_CUDA_INVALID_ARGUMENT;
    auto* table=new(std::nothrow) spacepdhcg_gtoc12_collection_options;
    if(!table)return SPACEPDHCG_CUDA_OUT_OF_MEMORY;
    table->storage=new(std::nothrow) SharedOptionRows;
    if(!table->storage){delete table;return SPACEPDHCG_CUDA_OUT_OF_MEMORY;}
    table->count=count;
    auto status=cudaGetDevice(&table->device);
    if(status==cudaSuccess)status=cudaMalloc(&table->rows,size_t(count?count:1)*sizeof(Option));
    table->storage->rows=table->rows;
    if(status==cudaSuccess && count)status=cudaMemcpyAsync(table->rows,rows,size_t(count)*sizeof(Option),cudaMemcpyDeviceToDevice,stream);
    const auto done=cudaStreamSynchronize(stream);
    if(status==cudaSuccess)status=done;
    if(status!=cudaSuccess){cudaFree(table->rows);delete table->storage;delete table;return mapped(status);}
    *output=table;return SPACEPDHCG_CUDA_SUCCESS;
}

extern "C" spacepdhcg_cuda_status spacepdhcg_gtoc12_collection_options_read(
    spacepdhcg_gtoc12_collection_options* table,Option* rows,int capacity) {
    if(!owned(table)||!rows||capacity<table->count)return SPACEPDHCG_CUDA_INVALID_ARGUMENT;
    return mapped(cudaMemcpy(rows,table->rows,size_t(table->count)*sizeof(Option),cudaMemcpyDeviceToHost));
}

extern "C" spacepdhcg_cuda_status spacepdhcg_gtoc12_collection_options_destroy(
    spacepdhcg_gtoc12_collection_options** output) {
    if(!output||!owned(*output))return SPACEPDHCG_CUDA_INVALID_ARGUMENT;
    auto* table=*output;auto status=cudaSuccess;
    if(--table->storage->references==0){status=cudaFree(table->storage->rows);delete table->storage;}
    delete table;*output=nullptr;return mapped(status);
}

spacepdhcg_cuda_status gtoc12_collection_options_view(
    spacepdhcg_gtoc12_collection_options* table,const Option** rows,int* count) {
    if(!owned(table)||!rows||!count)return SPACEPDHCG_CUDA_INVALID_ARGUMENT;
    *rows=table->rows;*count=table->count;return SPACEPDHCG_CUDA_SUCCESS;
}

spacepdhcg_cuda_status gtoc12_collection_options_share(
    spacepdhcg_gtoc12_collection_options* table,spacepdhcg_gtoc12_collection_options** output) {
    if(!owned(table)||!output||*output||table->storage->references==SIZE_MAX)return SPACEPDHCG_CUDA_INVALID_ARGUMENT;
    auto* shared=new(std::nothrow) spacepdhcg_gtoc12_collection_options(*table);
    if(!shared)return SPACEPDHCG_CUDA_OUT_OF_MEMORY;
    ++table->storage->references;*output=shared;return SPACEPDHCG_CUDA_SUCCESS;
}

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

extern "C" spacepdhcg_cuda_status spacepdhcg_gtoc12_collection_resident(
    spacepdhcg_gtoc12_collection* w,spacepdhcg_gtoc12_collection_options* table,
    const Query* query,Result* result,Option* selected) {
    if(!w||!owned(table)||table->device!=w->device||table->count>w->capacity
        ||!query||!result||!selected)return SPACEPDHCG_CUDA_INVALID_ARGUMENT;
    std::unique_lock<std::mutex> lock(w->mutex,std::try_to_lock);
    if(!lock.owns_lock())return SPACEPDHCG_CUDA_BUSY;
    auto status=cudaMemcpyAsync(w->query,query,sizeof(Query),cudaMemcpyHostToDevice,w->stream);
    auto code=mapped(status);
    if(status==cudaSuccess)code=spacepdhcg_gtoc12_collection_launch_device(
        w,table->rows,table->count,w->query,w->result,
        {{SPACEPDHCG_DEVICE_CUDA,w->device},reinterpret_cast<uintptr_t>(w->stream)});
    // Reuse one row of the host-input scratch as the selected tuple. The table
    // is immutable and every call completes before another workspace lease.
    if(code==SPACEPDHCG_CUDA_SUCCESS) {
        gather_selected<<<1,1,0,w->stream>>>(table->rows,w->result,w->options);
        status=cudaGetLastError();
        if(status==cudaSuccess)status=cudaMemcpyAsync(result,w->result,sizeof(Result),cudaMemcpyDeviceToHost,w->stream);
        if(status==cudaSuccess)status=cudaMemcpyAsync(selected,w->options,sizeof(Option),cudaMemcpyDeviceToHost,w->stream);
    }
    const auto done=cudaStreamSynchronize(w->stream);
    if(code!=SPACEPDHCG_CUDA_SUCCESS)return code;
    return mapped(status==cudaSuccess?done:status);
}
