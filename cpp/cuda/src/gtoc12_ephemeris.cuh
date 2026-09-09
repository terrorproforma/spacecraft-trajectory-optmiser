// Included by orbitweaver_gpu.cu to share its tested element_state operator.
#include "spacepdhcg/cuda/gtoc12_ephemeris_c_api.h"
#include <thread>

struct spacepdhcg_gtoc12_ephemeris {
    int bodies{}, capacity{}, device{};
    double mu{};
    std::thread::id owner;
    spacepdhcg_orbitweaver_elements* elements{};
    spacepdhcg_gtoc12_ephemeris_request* requests{};
    spacepdhcg_gtoc12_ephemeris_result* results{};
    cudaStream_t stream{};
};

namespace gtoc12_ephemeris_detail {
using Workspace = spacepdhcg_gtoc12_ephemeris;
using Request = spacepdhcg_gtoc12_ephemeris_request;
using Result = spacepdhcg_gtoc12_ephemeris_result;

bool owned(const Workspace* w) {
    int device = -1;
    return w && w->owner == std::this_thread::get_id() &&
        cudaGetDevice(&device) == cudaSuccess && device == w->device;
}

__global__ void states(const spacepdhcg_orbitweaver_elements* elements, int bodies,
    const Request* requests, int count, Result* results, double mu) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const auto q = requests[i];
    Result result{};
    for (int k = 0; k < 3; ++k) result.position_km[k] = result.velocity_km_s[k] = NAN;
    result.status = 1;
    if (q.body_index >= 0 && q.body_index < bodies && q.reserved == 0 && isfinite(q.epoch_mjd)) {
        result.status = element_state(elements[q.body_index], q.epoch_mjd, mu,
            result.position_km, result.velocity_km_s) ? 0 : 2;
        if (result.status) {
            for (int k = 0; k < 3; ++k) result.position_km[k] = result.velocity_km_s[k] = NAN;
        }
    }
    results[i] = result;
}

cudaError_t release(Workspace* w) {
    cudaError_t first = cudaSuccess;
    const auto keep = [&](cudaError_t error) { if (first == cudaSuccess) first = error; };
    if (w->stream) keep(cudaStreamSynchronize(w->stream));
    if (w->elements) keep(cudaFree(w->elements));
    if (w->requests) keep(cudaFree(w->requests));
    if (w->results) keep(cudaFree(w->results));
    if (w->stream) keep(cudaStreamDestroy(w->stream));
    return first;
}
} // namespace gtoc12_ephemeris_detail

extern "C" spacepdhcg_cuda_status spacepdhcg_gtoc12_ephemeris_create(
    const spacepdhcg_orbitweaver_elements* bodies, int32_t body_count,
    int32_t capacity, double mu, int32_t device, spacepdhcg_gtoc12_ephemeris** output) {
    using namespace gtoc12_ephemeris_detail;
    if (!output || *output || !bodies || body_count < 1 || capacity < 1 || device < 0 ||
        !std::isfinite(mu) || mu <= 0) return SPACEPDHCG_CUDA_INVALID_ARGUMENT;
    for (int i = 0; i < body_count; ++i)
        if (!valid_elements(bodies[i])) return SPACEPDHCG_CUDA_INVALID_ARGUMENT;
    int current = -1;
    auto error = cudaGetDevice(&current);
    if (error != cudaSuccess) return mapped(error);
    if (current != device) return SPACEPDHCG_CUDA_INVALID_ARGUMENT;
    auto* w = new (std::nothrow) Workspace;
    if (!w) return SPACEPDHCG_CUDA_OUT_OF_MEMORY;
    w->bodies = body_count; w->capacity = capacity; w->device = device;
    w->mu = mu; w->owner = std::this_thread::get_id();
#define EPHEMERIS_CREATE(call) do { error = (call); if (error != cudaSuccess) { release(w); delete w; return mapped(error); } } while (false)
    EPHEMERIS_CREATE(cudaStreamCreateWithFlags(&w->stream, cudaStreamNonBlocking));
    EPHEMERIS_CREATE(cudaMalloc(&w->elements, size_t(body_count) * sizeof(*bodies)));
    EPHEMERIS_CREATE(cudaMalloc(&w->requests, size_t(capacity) * sizeof(*w->requests)));
    EPHEMERIS_CREATE(cudaMalloc(&w->results, size_t(capacity) * sizeof(*w->results)));
    EPHEMERIS_CREATE(cudaMemcpyAsync(w->elements, bodies, size_t(body_count) * sizeof(*bodies), cudaMemcpyHostToDevice, w->stream));
    EPHEMERIS_CREATE(cudaStreamSynchronize(w->stream));
#undef EPHEMERIS_CREATE
    *output = w;
    return SPACEPDHCG_CUDA_SUCCESS;
}

extern "C" spacepdhcg_cuda_status spacepdhcg_gtoc12_ephemeris_launch_device(
    spacepdhcg_gtoc12_ephemeris* w, const spacepdhcg_gtoc12_ephemeris_request* requests,
    int32_t count, spacepdhcg_gtoc12_ephemeris_result* results, spacepdhcg_accelerator_stream stream) {
    using namespace gtoc12_ephemeris_detail;
    if (!owned(w) || count < 0 || count > w->capacity || (count && (!requests || !results)) ||
        stream.device.type != SPACEPDHCG_DEVICE_CUDA || stream.device.id != w->device)
        return SPACEPDHCG_CUDA_INVALID_ARGUMENT;
    if (!count) return SPACEPDHCG_CUDA_SUCCESS;
    states<<<unsigned((size_t(count) + 127) / 128), 128, 0,
        reinterpret_cast<cudaStream_t>(stream.native_handle)>>>(
        w->elements, w->bodies, requests, count, results, w->mu);
    return mapped(cudaGetLastError());
}

extern "C" spacepdhcg_cuda_status spacepdhcg_gtoc12_ephemeris_host(
    spacepdhcg_gtoc12_ephemeris* w, const spacepdhcg_gtoc12_ephemeris_request* requests,
    int32_t count, spacepdhcg_gtoc12_ephemeris_result* results) {
    using namespace gtoc12_ephemeris_detail;
    if (!owned(w) || count < 0 || count > w->capacity || (count && (!requests || !results)))
        return SPACEPDHCG_CUDA_INVALID_ARGUMENT;
    if (!count) return SPACEPDHCG_CUDA_SUCCESS;
    auto error = cudaMemcpyAsync(w->requests, requests, size_t(count) * sizeof(*requests), cudaMemcpyHostToDevice, w->stream);
    if (error == cudaSuccess) {
        states<<<unsigned((size_t(count) + 127) / 128), 128, 0, w->stream>>>(
            w->elements, w->bodies, w->requests, count, w->results, w->mu);
        error = cudaGetLastError();
    }
    if (error == cudaSuccess) error = cudaMemcpyAsync(results, w->results,
        size_t(count) * sizeof(*results), cudaMemcpyDeviceToHost, w->stream);
    const auto completed = cudaStreamSynchronize(w->stream);
    return mapped(error == cudaSuccess ? completed : error);
}

extern "C" spacepdhcg_cuda_status spacepdhcg_gtoc12_ephemeris_destroy(spacepdhcg_gtoc12_ephemeris** output) {
    using namespace gtoc12_ephemeris_detail;
    if (!output || !owned(*output)) return SPACEPDHCG_CUDA_INVALID_ARGUMENT;
    const auto error = release(*output);
    delete *output; *output = nullptr;
    return mapped(error);
}
