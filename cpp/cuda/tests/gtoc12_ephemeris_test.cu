#include "spacepdhcg/cuda/gtoc12_ephemeris_c_api.h"
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <thread>

#define REQUIRE(expression) do { if (!(expression)) { std::fprintf(stderr, "Failed line %d: %s\n", __LINE__, #expression); return 1; } } while (false)
int main() {
    int devices = 0;
    REQUIRE(cudaGetDeviceCount(&devices) == cudaSuccess && devices > 0);
    REQUIRE(cudaSetDevice(0) == cudaSuccess);
    constexpr double mu = 1.32712440018e11, a = 149597870.7;
    spacepdhcg_orbitweaver_elements elements{64328, a, 0, 0, 0, 0, 0};
    spacepdhcg_gtoc12_ephemeris* workspace = nullptr;
    REQUIRE(spacepdhcg_gtoc12_ephemeris_create(&elements, 1, 5, mu, 0, &workspace) == 0);
    using Request = spacepdhcg_gtoc12_ephemeris_request;
    using Result = spacepdhcg_gtoc12_ephemeris_result;
    const double quarter_days = 0.5 * 3.14159265358979323846 / std::sqrt(mu / (a*a*a)) / 86400;
    Request requests[5]{{0,0,64328}, {0,0,64328+quarter_days}, {1,0,64328},
        {0,1,64328}, {0,0,std::numeric_limits<double>::quiet_NaN()}};
    Result actual[5];
    REQUIRE(spacepdhcg_gtoc12_ephemeris_host(workspace, requests, 5, actual) == 0);
    REQUIRE(actual[0].status == 0 && actual[1].status == 0);
    REQUIRE(std::fabs(actual[0].position_km[0]-a) < 1e-5);
    REQUIRE(std::fabs(actual[0].velocity_km_s[1]-std::sqrt(mu/a)) < 1e-12);
    REQUIRE(std::fabs(actual[1].position_km[0]) < 1e-5);
    REQUIRE(std::fabs(actual[1].position_km[1]-a) < 1e-5);
    for (int i=2; i<5; ++i) {
        REQUIRE(actual[i].status == 1);
        for (int k=0; k<3; ++k) REQUIRE(std::isnan(actual[i].position_km[k]) && std::isnan(actual[i].velocity_km_s[k]));
    }
    Result retained[5]; std::memcpy(retained, actual, sizeof(actual));
    REQUIRE(spacepdhcg_gtoc12_ephemeris_host(workspace, requests, 6, actual) == SPACEPDHCG_CUDA_INVALID_ARGUMENT);
    REQUIRE(std::memcmp(retained, actual, sizeof(actual)) == 0);
    REQUIRE(spacepdhcg_gtoc12_ephemeris_host(workspace, nullptr, 0, nullptr) == 0);
    int wrong_thread = -1;
    std::thread other([&] { wrong_thread = spacepdhcg_gtoc12_ephemeris_host(workspace, requests, 1, actual); });
    other.join(); REQUIRE(wrong_thread == SPACEPDHCG_CUDA_INVALID_ARGUMENT);

    Request* device_requests = nullptr;
    Result* device_results = nullptr;
    cudaStream_t stream = nullptr;
    REQUIRE(cudaMalloc(&device_requests, sizeof(requests)) == cudaSuccess);
    REQUIRE(cudaMalloc(&device_results, sizeof(actual)) == cudaSuccess);
    REQUIRE(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess);
    REQUIRE(cudaMemcpyAsync(device_requests, requests, sizeof(requests), cudaMemcpyHostToDevice, stream) == cudaSuccess);
    REQUIRE(cudaStreamSynchronize(stream) == cudaSuccess);
    spacepdhcg_accelerator_stream native{{SPACEPDHCG_DEVICE_CUDA,0},reinterpret_cast<uintptr_t>(stream)};
    REQUIRE(spacepdhcg_gtoc12_ephemeris_launch_device(workspace, device_requests, 6, device_results, native) == SPACEPDHCG_CUDA_INVALID_ARGUMENT);
    cudaGraph_t graph = nullptr; cudaGraphExec_t executable = nullptr;
    REQUIRE(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal) == cudaSuccess);
    REQUIRE(spacepdhcg_gtoc12_ephemeris_launch_device(workspace, device_requests, 5, device_results, native) == 0);
    REQUIRE(cudaStreamEndCapture(stream, &graph) == cudaSuccess);
    REQUIRE(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0) == cudaSuccess);
    REQUIRE(cudaGraphLaunch(executable, stream) == cudaSuccess);
    REQUIRE(cudaMemcpyAsync(actual, device_results, sizeof(actual), cudaMemcpyDeviceToHost, stream) == cudaSuccess);
    REQUIRE(cudaStreamSynchronize(stream) == cudaSuccess);
    REQUIRE(std::memcmp(retained, actual, sizeof(actual)) == 0);
    REQUIRE(cudaGraphExecDestroy(executable) == cudaSuccess);
    REQUIRE(cudaGraphDestroy(graph) == cudaSuccess);
    REQUIRE(cudaFree(device_requests) == cudaSuccess && cudaFree(device_results) == cudaSuccess);
    REQUIRE(cudaStreamDestroy(stream) == cudaSuccess);
    REQUIRE(spacepdhcg_gtoc12_ephemeris_destroy(&workspace) == 0 && workspace == nullptr);
    REQUIRE(spacepdhcg_gtoc12_ephemeris_destroy(&workspace) == SPACEPDHCG_CUDA_INVALID_ARGUMENT);
    elements.e = 1;
    REQUIRE(spacepdhcg_gtoc12_ephemeris_create(&elements, 1, 1, mu, 0, &workspace) == SPACEPDHCG_CUDA_INVALID_ARGUMENT);
    std::puts("GPU_EPHEMERIS_PASS host_rows=5 graph_rows=5 finite_rows=4 invalid_rows=6");
}
