#include "spacepdhcg/cuda/orbitweaver_gpu_c_api.h"

#include <cuda_runtime_api.h>

#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <new>

namespace {

constexpr double pi = 3.141592653589793238462643383279502884;

struct Geometry {
    double r1;
    double r2;
    double cosine;
    double a;
    double angle;
    bool valid;
};

struct Evaluation {
    double residual;
    double y;
    bool valid;
};

struct Root {
    double parameter;
    std::uint32_t iterations;
};

__device__ double dot3(const double* left, const double* right) {
    return left[0] * right[0] + left[1] * right[1] + left[2] * right[2];
}

__device__ void stumpff(const double z, double& c, double& s) {
    if (z > 1.0e-8) {
        const auto root = sqrt(z);
        c = (1.0 - cos(root)) / z;
        s = (root - sin(root)) / (root * root * root);
    } else if (z < -1.0e-8) {
        const auto root = sqrt(-z);
        c = (cosh(root) - 1.0) / (-z);
        s = (sinh(root) - root) / (root * root * root);
    } else {
        const auto z2 = z * z;
        const auto z3 = z2 * z;
        c = 0.5 - z / 24.0 + z2 / 720.0 - z3 / 40320.0;
        s = 1.0 / 6.0 - z / 120.0 + z2 / 5040.0 - z3 / 362880.0;
    }
}

__device__ Geometry geometry(
    const spacepdhcg_orbitweaver_lambert_request& request,
    const bool long_way
) {
    const auto r1 = sqrt(dot3(request.departure_position, request.departure_position));
    const auto r2 = sqrt(dot3(request.arrival_position, request.arrival_position));
    if (!(r1 > 0.0) || !(r2 > 0.0)) {
        return {r1, r2, 0.0, 0.0, 0.0, false};
    }
    auto cosine = dot3(request.departure_position, request.arrival_position) / (r1 * r2);
    cosine = fmin(1.0, fmax(-1.0, cosine));
    auto sine = sqrt(fmax(0.0, 1.0 - cosine * cosine));
    sine = long_way ? -sine : sine;
    const auto denominator = 1.0 - cosine;
    if (denominator <= 1.0e-14 || fabs(sine) <= 1.0e-14) {
        return {r1, r2, cosine, 0.0, 0.0, false};
    }
    const auto a = sine * sqrt(r1 * r2 / denominator);
    auto angle = acos(cosine);
    angle = long_way ? 2.0 * pi - angle : angle;
    return {r1, r2, cosine, a, angle, isfinite(a) && fabs(a) > 1.0e-14};
}

__device__ Evaluation evaluate(
    const double z,
    const Geometry value,
    const spacepdhcg_orbitweaver_lambert_request& request
) {
    double c = 0.0;
    double s = 0.0;
    stumpff(z, c, s);
    if (!isfinite(c) || !isfinite(s) || c <= 0.0) {
        return {0.0, 0.0, false};
    }
    const auto y =
        value.r1 + value.r2 + value.a * (z * s - 1.0) / sqrt(c);
    if (!isfinite(y) || y < 0.0) {
        return {0.0, 0.0, false};
    }
    const auto x = sqrt(y / c);
    const auto computed =
        (x * x * x * s + value.a * sqrt(y))
        / sqrt(request.gravitational_parameter);
    return {computed - request.time_of_flight, y, isfinite(computed)};
}

__device__ bool valid(
    const spacepdhcg_orbitweaver_lambert_request& request
) {
    if (!(request.time_of_flight > 0.0)
        || !(request.gravitational_parameter > 0.0)
        || !(request.time_tolerance > 0.0)
        || request.maximum_iterations == 0U
        || (!request.include_short_way && !request.include_long_way)) {
        return false;
    }
    for (int component = 0; component < 3; ++component) {
        if (!isfinite(request.departure_position[component])
            || !isfinite(request.arrival_position[component])) {
            return false;
        }
    }
    return true;
}

__device__ bool bisect(
    double lower,
    double upper,
    const Geometry value,
    const spacepdhcg_orbitweaver_lambert_request& request,
    Root& root
) {
    auto low = evaluate(lower, value, request);
    const auto high = evaluate(upper, value, request);
    if (!low.valid || !high.valid || low.residual * high.residual > 0.0) {
        return false;
    }
    for (std::uint32_t iteration = 0U;
         iteration < request.maximum_iterations;
         ++iteration) {
        const auto middle_parameter = 0.5 * (lower + upper);
        const auto middle = evaluate(middle_parameter, value, request);
        if (!middle.valid) {
            return false;
        }
        if (fabs(middle.residual) <= request.time_tolerance
            || fabs(upper - lower) <= 1.0e-13) {
            root = {middle_parameter, iteration + 1U};
            return true;
        }
        if (low.residual * middle.residual <= 0.0) {
            upper = middle_parameter;
        } else {
            lower = middle_parameter;
            low = middle;
        }
    }
    return false;
}

__device__ std::uint32_t scan(
    const double lower,
    const double upper,
    const std::uint32_t samples,
    const Geometry value,
    const spacepdhcg_orbitweaver_lambert_request& request,
    Root roots[2]
) {
    bool has_previous = false;
    double previous_parameter = 0.0;
    Evaluation previous{};
    std::uint32_t count = 0U;
    for (std::uint32_t sample = 0U; sample <= samples; ++sample) {
        const auto parameter =
            lower + static_cast<double>(sample) / static_cast<double>(samples)
                        * (upper - lower);
        const auto current = evaluate(parameter, value, request);
        if (!current.valid) {
            has_previous = false;
            continue;
        }
        Root root{};
        const auto exact = fabs(current.residual) <= request.time_tolerance;
        const auto bracket =
            has_previous && previous.residual * current.residual < 0.0;
        const auto found = exact
                               ? (root = Root{parameter, 0U}, true)
                               : bracket
                                     && bisect(
                                         previous_parameter,
                                         parameter,
                                         value,
                                         request,
                                         root
                                     );
        if (found && count < 2U
            && (count == 0U
                || fabs(root.parameter - roots[count - 1U].parameter)
                       > 1.0e-9
                             * fmax(
                                 1.0,
                                 fmax(
                                     fabs(root.parameter),
                                     fabs(roots[count - 1U].parameter)
                                 )
                             ))) {
            roots[count++] = root;
        }
        previous_parameter = parameter;
        previous = current;
        has_previous = true;
    }
    return count;
}

__device__ bool solution(
    const spacepdhcg_orbitweaver_lambert_request& request,
    const Geometry value,
    const Root root,
    spacepdhcg_orbitweaver_lambert_result& result
) {
    const auto state = evaluate(root.parameter, value, request);
    const auto g =
        value.a * sqrt(state.y / request.gravitational_parameter);
    if (!state.valid || !isfinite(g) || fabs(g) <= 1.0e-14) {
        result.status = SPACEPDHCG_ORBITWEAVER_ARC_NUMERICAL_FAILURE;
        return false;
    }
    const auto f = 1.0 - state.y / value.r1;
    const auto g_dot = 1.0 - state.y / value.r2;
    for (int component = 0; component < 3; ++component) {
        result.departure_velocity[component] =
            (request.arrival_position[component]
             - f * request.departure_position[component])
            / g;
        result.arrival_velocity[component] =
            (g_dot * request.arrival_position[component]
             - request.departure_position[component])
            / g;
    }
    result.universal_parameter = root.parameter;
    result.transfer_angle_radians = value.angle;
    result.iterations = root.iterations;
    result.time_of_flight_residual = state.residual;
    result.status = SPACEPDHCG_ORBITWEAVER_ARC_FEASIBLE;
    return true;
}

__global__ void kernel(
    const spacepdhcg_orbitweaver_lambert_request* requests,
    const std::size_t count,
    const std::uint32_t supported_revolutions,
    const std::uint32_t samples,
    spacepdhcg_orbitweaver_lambert_result* results,
    unsigned long long* counters,
    const int* cancelled
) {
    const auto input =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (input >= count) {
        return;
    }
    const auto stride =
        static_cast<std::size_t>(2U * (1U + 2U * supported_revolutions));
    auto* output = results + input * stride;
    const auto& request = requests[input];
    for (std::size_t slot = 0U; slot < stride; ++slot) {
        output[slot] = {};
        output[slot].deterministic_id = request.deterministic_id;
        output[slot].input_index = static_cast<std::uint32_t>(input);
        output[slot].family_index = static_cast<std::uint32_t>(slot);
        output[slot].status = SPACEPDHCG_ORBITWEAVER_ARC_UNSUPPORTED;
    }
    const auto invalid = !valid(request) || !geometry(request, false).valid;
    if (*cancelled != 0 || invalid) {
        const auto status = *cancelled != 0
                                ? SPACEPDHCG_ORBITWEAVER_ARC_CANCELLED
                                : SPACEPDHCG_ORBITWEAVER_ARC_INVALID_INPUT;
        for (std::size_t slot = 0U; slot < stride; ++slot) {
            output[slot].status = status;
        }
        atomicAdd(counters + 1U, static_cast<unsigned long long>(stride));
        return;
    }
    for (std::uint32_t direction = 0U; direction < 2U; ++direction) {
        const auto included =
            direction == 0U ? request.include_short_way : request.include_long_way;
        if (included == 0) {
            continue;
        }
        const auto value = geometry(request, direction != 0U);
        const auto base = static_cast<std::size_t>(direction)
                          * (1U + 2U * supported_revolutions);
        Root roots[2]{};
        auto& zero = output[base];
        zero.long_way = static_cast<int32_t>(direction);
        zero.status = SPACEPDHCG_ORBITWEAVER_ARC_NO_SOLUTION;
        if (scan(
                -4.0 * pi * pi,
                4.0 * pi * pi - 1.0e-8,
                samples,
                value,
                request,
                roots
            )
            > 0U) {
            static_cast<void>(solution(request, value, roots[0], zero));
        }
        for (std::uint32_t revolution = 1U;
             revolution <= supported_revolutions;
             ++revolution) {
            const auto first =
                base + 1U + 2U * static_cast<std::size_t>(revolution - 1U);
            auto& lower = output[first];
            auto& higher = output[first + 1U];
            lower.long_way = higher.long_way = static_cast<int32_t>(direction);
            lower.revolutions = higher.revolutions = revolution;
            lower.branch = SPACEPDHCG_ORBITWEAVER_LAMBERT_LOWER_PARAMETER;
            higher.branch = SPACEPDHCG_ORBITWEAVER_LAMBERT_HIGHER_PARAMETER;
            if (revolution > request.maximum_revolutions) {
                continue;
            }
            lower.status = higher.status = SPACEPDHCG_ORBITWEAVER_ARC_NO_SOLUTION;
            const auto lower_singularity =
                4.0 * static_cast<double>(revolution * revolution) * pi * pi;
            const auto next = revolution + 1U;
            const auto upper_singularity =
                4.0 * static_cast<double>(next * next) * pi * pi;
            const auto margin = 1.0e-9 * fmax(1.0, upper_singularity);
            const auto root_count = scan(
                lower_singularity + margin,
                upper_singularity - margin,
                samples,
                value,
                request,
                roots
            );
            if (root_count == 1U) {
                lower.branch = SPACEPDHCG_ORBITWEAVER_LAMBERT_UNIQUE;
                static_cast<void>(solution(request, value, roots[0], lower));
            } else if (root_count == 2U) {
                static_cast<void>(solution(request, value, roots[0], lower));
                static_cast<void>(solution(request, value, roots[1], higher));
            }
        }
    }
    unsigned long long feasible = 0U;
    for (std::size_t slot = 0U; slot < stride; ++slot) {
        feasible += output[slot].status == SPACEPDHCG_ORBITWEAVER_ARC_FEASIBLE
                        ? 1U
                        : 0U;
    }
    atomicAdd(counters, feasible);
    atomicAdd(counters + 1U, static_cast<unsigned long long>(stride) - feasible);
}

spacepdhcg_cuda_status mapped(const cudaError_t status) {
    if (status == cudaSuccess) {
        return SPACEPDHCG_CUDA_SUCCESS;
    }
    return status == cudaErrorMemoryAllocation ? SPACEPDHCG_CUDA_OUT_OF_MEMORY
                                               : SPACEPDHCG_CUDA_RUNTIME_ERROR;
}

}  // namespace

struct spacepdhcg_orbitweaver_lambert_workspace {
    spacepdhcg_orbitweaver_lambert_config config{};
    spacepdhcg_orbitweaver_lambert_request* requests{nullptr};
    spacepdhcg_orbitweaver_lambert_result* results{nullptr};
    unsigned long long* counters{nullptr};
    int* cancelled{nullptr};
    cudaStream_t stream{nullptr};
    cudaEvent_t completion{nullptr};
    bool owns_stream{false};
    std::atomic<bool> busy{false};
    std::uint64_t batches{0U};
    std::uint64_t request_count{0U};
    std::uint64_t result_count{0U};
    std::uint64_t feasible{0U};
    std::uint64_t failed{0U};
};

extern "C" {

size_t spacepdhcg_orbitweaver_lambert_result_stride(
    const uint32_t revolutions
) {
    return 2U * (1U + 2U * static_cast<size_t>(revolutions));
}

spacepdhcg_cuda_status spacepdhcg_orbitweaver_lambert_workspace_create(
    const spacepdhcg_orbitweaver_lambert_config* config,
    const spacepdhcg_accelerator_stream stream,
    spacepdhcg_orbitweaver_lambert_workspace** workspace
) {
    if (config == nullptr || workspace == nullptr || *workspace != nullptr
        || config->abi_version != SPACEPDHCG_ORBITWEAVER_GPU_ABI_VERSION
        || config->maximum_batch_size == 0U || config->scan_samples_per_band < 16U
        || stream.device.type != SPACEPDHCG_DEVICE_CUDA
        || stream.device.id != static_cast<int32_t>(config->device_id)) {
        return SPACEPDHCG_CUDA_INVALID_ARGUMENT;
    }
    auto* created = new (std::nothrow) spacepdhcg_orbitweaver_lambert_workspace{};
    if (created == nullptr) {
        return SPACEPDHCG_CUDA_OUT_OF_MEMORY;
    }
    created->config = *config;
    auto status = cudaSetDevice(static_cast<int>(config->device_id));
    if (status == cudaSuccess) {
        status = stream.native_handle == 0U
                     ? cudaStreamCreateWithFlags(&created->stream, cudaStreamNonBlocking)
                     : cudaSuccess;
        created->owns_stream = stream.native_handle == 0U;
        if (stream.native_handle != 0U) {
            created->stream = reinterpret_cast<cudaStream_t>(stream.native_handle);
        }
    }
    const auto stride =
        spacepdhcg_orbitweaver_lambert_result_stride(
            config->supported_maximum_revolutions
        );
    if (status == cudaSuccess) {
        status = cudaMalloc(
            reinterpret_cast<void**>(&created->requests),
            config->maximum_batch_size * sizeof(*created->requests)
        );
    }
    if (status == cudaSuccess) {
        status = cudaMalloc(
            reinterpret_cast<void**>(&created->results),
            config->maximum_batch_size * stride * sizeof(*created->results)
        );
    }
    if (status == cudaSuccess) {
        status = cudaMallocManaged(
            reinterpret_cast<void**>(&created->counters),
            2U * sizeof(*created->counters)
        );
    }
    if (status == cudaSuccess) {
        status = cudaMallocManaged(
            reinterpret_cast<void**>(&created->cancelled),
            sizeof(*created->cancelled)
        );
    }
    if (status == cudaSuccess) {
        status = cudaEventCreateWithFlags(&created->completion, cudaEventDisableTiming);
    }
    if (status != cudaSuccess) {
        cudaFree(created->requests);
        cudaFree(created->results);
        cudaFree(created->counters);
        cudaFree(created->cancelled);
        if (created->owns_stream) {
            cudaStreamDestroy(created->stream);
        }
        delete created;
        return mapped(status);
    }
    *created->cancelled = 0;
    *workspace = created;
    return SPACEPDHCG_CUDA_SUCCESS;
}

spacepdhcg_cuda_status spacepdhcg_orbitweaver_lambert_evaluate_async(
    spacepdhcg_orbitweaver_lambert_workspace* workspace,
    const spacepdhcg_orbitweaver_lambert_request* requests,
    const size_t request_count,
    spacepdhcg_orbitweaver_lambert_result* results,
    const size_t result_capacity,
    const spacepdhcg_accelerator_stream stream
) {
    if (workspace == nullptr || requests == nullptr || results == nullptr
        || request_count == 0U || request_count > workspace->config.maximum_batch_size
        || stream.device.type != SPACEPDHCG_DEVICE_CUDA
        || stream.device.id != static_cast<int32_t>(workspace->config.device_id)) {
        return SPACEPDHCG_CUDA_INVALID_ARGUMENT;
    }
    bool expected = false;
    if (!workspace->busy.compare_exchange_strong(expected, true)) {
        return SPACEPDHCG_CUDA_BUSY;
    }
    const auto stride = spacepdhcg_orbitweaver_lambert_result_stride(
        workspace->config.supported_maximum_revolutions
    );
    const auto output_count = request_count * stride;
    if (result_capacity < output_count) {
        workspace->busy.store(false);
        return SPACEPDHCG_CUDA_INVALID_ARGUMENT;
    }
    *workspace->cancelled = 0;
    workspace->counters[0] = workspace->counters[1] = 0U;
    auto status = cudaMemcpyAsync(
        workspace->requests,
        requests,
        request_count * sizeof(*requests),
        cudaMemcpyHostToDevice,
        workspace->stream
    );
    if (status == cudaSuccess) {
        constexpr std::uint32_t threads = 128U;
        const auto blocks =
            static_cast<std::uint32_t>((request_count + threads - 1U) / threads);
        kernel<<<blocks, threads, 0U, workspace->stream>>>(
            workspace->requests,
            request_count,
            workspace->config.supported_maximum_revolutions,
            workspace->config.scan_samples_per_band,
            workspace->results,
            workspace->counters,
            workspace->cancelled
        );
        status = cudaGetLastError();
    }
    if (status == cudaSuccess) {
        status = cudaMemcpyAsync(
            results,
            workspace->results,
            output_count * sizeof(*results),
            cudaMemcpyDeviceToHost,
            workspace->stream
        );
    }
    if (status == cudaSuccess) {
        status = cudaEventRecord(workspace->completion, workspace->stream);
    }
    if (status != cudaSuccess) {
        workspace->busy.store(false);
        return mapped(status);
    }
    ++workspace->batches;
    workspace->request_count += request_count;
    workspace->result_count += output_count;
    return SPACEPDHCG_CUDA_SUCCESS;
}

spacepdhcg_cuda_status spacepdhcg_orbitweaver_lambert_workspace_telemetry(
    const spacepdhcg_orbitweaver_lambert_workspace* workspace,
    spacepdhcg_orbitweaver_batch_telemetry* telemetry
) {
    if (workspace == nullptr || telemetry == nullptr
        || telemetry->abi_version != SPACEPDHCG_ORBITWEAVER_GPU_ABI_VERSION) {
        return SPACEPDHCG_CUDA_INVALID_ARGUMENT;
    }
    auto* mutable_workspace =
        const_cast<spacepdhcg_orbitweaver_lambert_workspace*>(workspace);
    if (workspace->busy.load()) {
        const auto status = cudaEventQuery(workspace->completion);
        if (status == cudaErrorNotReady) {
            return SPACEPDHCG_CUDA_BUSY;
        }
        if (status != cudaSuccess) {
            return mapped(status);
        }
        mutable_workspace->feasible += workspace->counters[0];
        mutable_workspace->failed += workspace->counters[1];
        mutable_workspace->busy.store(false);
    }
    const auto stride = spacepdhcg_orbitweaver_lambert_result_stride(
        workspace->config.supported_maximum_revolutions
    );
    telemetry->batches_submitted = workspace->batches;
    telemetry->requests_submitted = workspace->request_count;
    telemetry->results_emitted = workspace->result_count;
    telemetry->feasible_results = workspace->feasible;
    telemetry->failed_results = workspace->failed;
    telemetry->input_bytes =
        workspace->request_count * sizeof(spacepdhcg_orbitweaver_lambert_request);
    telemetry->output_bytes =
        workspace->result_count * sizeof(spacepdhcg_orbitweaver_lambert_result);
    telemetry->workspace_bytes =
        workspace->config.maximum_batch_size
            * (sizeof(spacepdhcg_orbitweaver_lambert_request)
               + stride * sizeof(spacepdhcg_orbitweaver_lambert_result))
        + 2U * sizeof(unsigned long long) + sizeof(int);
    telemetry->maximum_batch_size = workspace->config.maximum_batch_size;
    telemetry->device_id = static_cast<int32_t>(workspace->config.device_id);
    return SPACEPDHCG_CUDA_SUCCESS;
}

spacepdhcg_cuda_status spacepdhcg_orbitweaver_lambert_workspace_cancel(
    spacepdhcg_orbitweaver_lambert_workspace* workspace
) {
    if (workspace == nullptr) {
        return SPACEPDHCG_CUDA_INVALID_ARGUMENT;
    }
    *workspace->cancelled = 1;
    return SPACEPDHCG_CUDA_SUCCESS;
}

spacepdhcg_cuda_status spacepdhcg_orbitweaver_lambert_workspace_destroy(
    spacepdhcg_orbitweaver_lambert_workspace** workspace
) {
    if (workspace == nullptr || *workspace == nullptr) {
        return SPACEPDHCG_CUDA_INVALID_ARGUMENT;
    }
    auto* owned = *workspace;
    if (owned->busy.load()) {
        const auto status = cudaEventSynchronize(owned->completion);
        if (status != cudaSuccess) {
            return mapped(status);
        }
    }
    cudaEventDestroy(owned->completion);
    cudaFree(owned->cancelled);
    cudaFree(owned->counters);
    cudaFree(owned->results);
    cudaFree(owned->requests);
    if (owned->owns_stream) {
        cudaStreamDestroy(owned->stream);
    }
    delete owned;
    *workspace = nullptr;
    return SPACEPDHCG_CUDA_SUCCESS;
}

}  // extern "C"
