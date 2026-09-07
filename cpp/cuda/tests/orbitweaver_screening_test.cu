#include "spacepdhcg/cuda/orbitweaver_gpu_c_api.h"
#include <cuda_runtime_api.h>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <thread>
#include <vector>

using Request = spacepdhcg_orbitweaver_lambert_request;
using Result = spacepdhcg_orbitweaver_lambert_result;
static_assert(sizeof(Request) == 96 && sizeof(Result) == 112);
static void require(bool ok) { if (!ok) { std::fputs("screening assertion failed\n", stderr); std::abort(); } }
static void check(cudaError_t s) { require(s == cudaSuccess); }
static void pass(spacepdhcg_cuda_status s) { require(s == SPACEPDHCG_CUDA_SUCCESS); }

int main() {
    constexpr size_t count = 37, stride = 6;
    cudaStream_t native{};
    check(cudaStreamCreateWithFlags(&native, cudaStreamNonBlocking));
    spacepdhcg_accelerator_stream stream{{SPACEPDHCG_DEVICE_CUDA, 0}, reinterpret_cast<uintptr_t>(native)};
    spacepdhcg_orbitweaver_lambert_config config{1, 0, count, 1, 256};
    spacepdhcg_orbitweaver_lambert_workspace* workspace = nullptr;
    pass(spacepdhcg_orbitweaver_lambert_workspace_create(&config, stream, &workspace));
    std::vector<Request> requests(count);
    for (size_t i = 0; i < count; ++i) {
        auto& q = requests[i];
        q.deterministic_id = 100 + i;
        q.departure_position[0] = 7e6;
        q.arrival_position[1] = 8e6;
        q.time_of_flight = 3600 + i * 10;
        q.gravitational_parameter = 3.986004418e14;
        q.time_tolerance = 1e-8;
        q.maximum_iterations = 256;
        q.maximum_revolutions = 1;
        q.include_short_way = q.include_long_way = 1;
    }
    requests[1].time_of_flight = std::numeric_limits<double>::infinity();
    std::vector<Result> blocking(count * stride), legacy(count * stride), device(count * stride);
    // Hold the borrowed stream so concurrent API calls must see an in-progress
    // blocking batch, rather than observing an old completion event.
    std::atomic<bool> release{false};
    check(cudaLaunchHostFunc(native, [](void* p) {
        auto& flag = *static_cast<std::atomic<bool>*>(p);
        while (!flag.load()) std::this_thread::yield();
    }, &release));
    std::thread worker([&] {
        check(cudaSetDevice(0));
        pass(spacepdhcg_orbitweaver_lambert_screening_host(workspace, requests.data(), count, blocking.data(), blocking.size()));
    });
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (spacepdhcg_orbitweaver_lambert_workspace_finish(workspace) != SPACEPDHCG_CUDA_BUSY) {
        require(std::chrono::steady_clock::now() < deadline);
        std::this_thread::yield();
    }
    spacepdhcg_orbitweaver_batch_telemetry telemetry{};
    telemetry.abi_version = 1;
    require(spacepdhcg_orbitweaver_lambert_workspace_telemetry(workspace, &telemetry) == SPACEPDHCG_CUDA_BUSY);
    require(spacepdhcg_orbitweaver_lambert_workspace_cancel(workspace) == SPACEPDHCG_CUDA_BUSY);
    require(spacepdhcg_orbitweaver_lambert_workspace_destroy(&workspace) == SPACEPDHCG_CUDA_BUSY);
    release.store(true);
    worker.join();
    pass(spacepdhcg_orbitweaver_lambert_evaluate_async(workspace, requests.data(), count, legacy.data(), legacy.size(), stream));
    pass(spacepdhcg_orbitweaver_lambert_workspace_finish(workspace));
    pass(spacepdhcg_orbitweaver_lambert_workspace_telemetry(workspace, &telemetry));
    require(telemetry.requests_submitted == count * 2 && telemetry.results_emitted == count * stride * 2);
    require(telemetry.feasible_results + telemetry.failed_results == telemetry.results_emitted);

    Request* dq = nullptr;
    Result* dr = nullptr;
    check(cudaMalloc(&dq, count * sizeof(Request)));
    check(cudaMalloc(&dr, device.size() * sizeof(Result)));
    check(cudaMemcpyAsync(dq, requests.data(), count * sizeof(Request), cudaMemcpyHostToDevice, native));
    check(cudaStreamSynchronize(native));
    require(spacepdhcg_orbitweaver_lambert_launch_device(dq, count, 1, 256, dr, 1, stream) == SPACEPDHCG_CUDA_INVALID_ARGUMENT);
    check(cudaStreamBeginCapture(native, cudaStreamCaptureModeThreadLocal));
    pass(spacepdhcg_orbitweaver_lambert_launch_device(dq, count, 1, 256, dr, device.size(), stream));
    cudaGraph_t graph{};
    cudaGraphExec_t executable{};
    check(cudaStreamEndCapture(native, &graph));
    check(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
    for (int replay = 0; replay < 3; ++replay) {
        check(cudaGraphLaunch(executable, native));
        check(cudaMemcpyAsync(device.data(), dr, device.size() * sizeof(Result), cudaMemcpyDeviceToHost, native));
        check(cudaStreamSynchronize(native));
        for (size_t i = 0; i < device.size(); ++i) {
            require(blocking[i].status == legacy[i].status && blocking[i].status == device[i].status);
            require(device[i].deterministic_id == requests[i / stride].deterministic_id);
            if (device[i].status == SPACEPDHCG_ORBITWEAVER_ARC_FEASIBLE) {
                for (int k = 0; k < 3; ++k) {
                    require(std::abs(blocking[i].departure_velocity[k] - device[i].departure_velocity[k]) < 1e-8);
                    require(std::abs(blocking[i].arrival_velocity[k] - legacy[i].arrival_velocity[k]) < 1e-8);
                }
            }
        }
    }
    require(device[0].status == SPACEPDHCG_ORBITWEAVER_ARC_FEASIBLE);
    require(device[stride].status == SPACEPDHCG_ORBITWEAVER_ARC_INVALID_INPUT);
    check(cudaGraphExecDestroy(executable)); check(cudaGraphDestroy(graph));
    check(cudaFree(dq)); check(cudaFree(dr));
    pass(spacepdhcg_orbitweaver_lambert_workspace_destroy(&workspace));
    check(cudaStreamDestroy(native));
    std::puts("screening: host, async, device graph and concurrent API guards passed");
}
