#include "spacepdhcg/cuda/gtoc12_collection_c_api.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

using Option = spacepdhcg_gtoc12_collection_option;
using Query = spacepdhcg_gtoc12_collection_query;
using Result = spacepdhcg_gtoc12_collection_result;
static_assert(sizeof(Option) == 24 && sizeof(Query) == 120 && sizeof(Result) == 16);
static void require(bool value) { if (!value) std::abort(); }
static void check(cudaError_t s) { require(s == cudaSuccess); }
static void pass(spacepdhcg_cuda_status s) { require(s == SPACEPDHCG_CUDA_SUCCESS); }

int main() {
    constexpr int n = 513;
    std::vector<Option> options(n);
    for (int i = 0; i < n; ++i) options[i] = {0., 66000. + i, 180.};
    Query q{0, 0, 2000., 68000., INFINITY, 0.5, 1.2, 1.05, 0.65, 0., 1., 0.6, 86400., 365.25, 10., 39.2266};
    spacepdhcg_gtoc12_collection* w = nullptr;
    pass(spacepdhcg_gtoc12_collection_create(n, 0, &w));
    Option* input{}; Query* query{}; Result* output{};
    check(cudaMalloc(&input, n * sizeof(Option))); check(cudaMalloc(&query, sizeof(Query)));
    check(cudaMalloc(&output, sizeof(Result)));
    cudaStream_t stream{}; check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    check(cudaMemcpyAsync(input, options.data(), n * sizeof(Option), cudaMemcpyHostToDevice, stream));
    check(cudaStreamSynchronize(stream));
    check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    pass(spacepdhcg_gtoc12_collection_launch_device(w, input, n, query, output,
        {{SPACEPDHCG_DEVICE_CUDA, 0}, reinterpret_cast<uintptr_t>(stream)}));
    cudaGraph_t graph{}; cudaGraphExec_t executable{};
    check(cudaStreamEndCapture(stream, &graph)); check(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
    for (int replay = 0; replay < 4; ++replay) {
        q.mode = replay == 1; q.mass = replay == 2 ? 0. : 2000.;
        Result host{}, device{};
        pass(spacepdhcg_gtoc12_collection_host(w, options.data(), n, &q, &host));
        check(cudaMemcpyAsync(query, &q, sizeof(Query), cudaMemcpyHostToDevice, stream));
        check(cudaGraphLaunch(executable, stream));
        check(cudaMemcpyAsync(&device, output, sizeof(Result), cudaMemcpyDeviceToHost, stream));
        check(cudaStreamSynchronize(stream));
        require(host.index == device.index && host.status == device.status && host.cost == device.cost);
        require(device.index == (replay == 2 ? -1 : replay == 1 ? 0 : n - 1));
        require(device.status == (replay == 2));
    }
    Result empty{}; pass(spacepdhcg_gtoc12_collection_host(w, options.data(), 0, &q, &empty));
    require(empty.index == -1 && empty.status == 0 && std::isinf(empty.cost));
    require(spacepdhcg_gtoc12_collection_host(w, options.data(), n + 1, &q, &empty) == SPACEPDHCG_CUDA_INVALID_ARGUMENT);
    check(cudaGraphExecDestroy(executable)); check(cudaGraphDestroy(graph));
    check(cudaFree(input)); check(cudaFree(query)); check(cudaFree(output)); check(cudaStreamDestroy(stream));
    pass(spacepdhcg_gtoc12_collection_destroy(&w));
    std::puts("GPU collection: graph replay, ordered ties, invalid recovery and guards passed");
}
