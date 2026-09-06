// Minimal conditional graph diagnostic, independent of QOCO/cuDSS.
// Compile explicitly with CUDA12.8+; excluded from the *_test.cu CMake glob.
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#define CHECK(call) do { auto status = (call); if (status != cudaSuccess) { \
    std::fprintf(stderr, "line=%d CUDA=%d %s\n", __LINE__, int(status), cudaGetErrorString(status)); \
    std::exit(1); } } while (0)

__global__ void start(cudaGraphConditionalHandle h, int* count, int limit) {
    *count = 0;
    cudaGraphSetConditional(h, limit > 0);
}
__global__ void finish(cudaGraphConditionalHandle h, int* count, int limit) {
    cudaGraphSetConditional(h, ++*count < limit);
}
__global__ void values(const int* x, int* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = x[i] + 7;
}

int main(int argc, char** argv) {
    const bool copies = argc == 1 || std::strcmp(argv[1], "--kernels") != 0;
    const bool direct = argc > 2;
    constexpr int n = 4097;
    int *x, *y, *count;
    CHECK(cudaMalloc(&x, n * sizeof(int)));
    CHECK(cudaMalloc(&y, n * sizeof(int)));
    CHECK(cudaMalloc(&count, sizeof(int)));
    std::vector<int> input(n), output(n);
    for (int i = 0; i < n; ++i) input[i] = i;
    CHECK(cudaMemcpy(x, input.data(), n * sizeof(int), cudaMemcpyHostToDevice));
    cudaStream_t stream;
    CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    cudaGraph_t child;
    CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    if (copies) {
        CHECK(cudaMemsetAsync(y, 0, n * sizeof(int), stream));
        CHECK(cudaMemcpyAsync(y, x, n * sizeof(int), cudaMemcpyDeviceToDevice, stream));
    }
    values<<<17, 256, 0, stream>>>(x, y, n);
    CHECK(cudaGetLastError());
    CHECK(cudaStreamEndCapture(stream, &child));
    for (int limit : {0, 1, 3}) {
        cudaGraph_t graph, ended;
        CHECK(cudaGraphCreate(&graph, 0));
        cudaGraphConditionalHandle handle;
        CHECK(cudaGraphConditionalHandleCreate(&handle, graph, 0, cudaGraphCondAssignDefault));
        CHECK(cudaStreamBeginCaptureToGraph(stream, graph, nullptr, nullptr, 0, cudaStreamCaptureModeThreadLocal));
        start<<<1, 1, 0, stream>>>(handle, count, limit);
        CHECK(cudaStreamEndCapture(stream, &ended));
        size_t size = 1;
        cudaGraphNode_t root, conditional, nested;
        CHECK(cudaGraphGetNodes(graph, &root, &size));
        cudaGraphNodeParams params = {};
        params.type = cudaGraphNodeTypeConditional;
        params.conditional.handle = handle;
        params.conditional.type = cudaGraphCondTypeWhile;
        params.conditional.size = 1;
        CHECK(cudaGraphAddNode(&conditional, graph, &root, 1, &params));
        cudaGraph_t body = params.conditional.phGraph_out[0];
        if (!direct) CHECK(cudaGraphAddChildGraphNode(&nested, body, nullptr, 0, child));
        CHECK(cudaStreamBeginCaptureToGraph(stream, body, direct ? nullptr : &nested, nullptr,
                                           direct ? 0 : 1, cudaStreamCaptureModeThreadLocal));
        if (direct) {
            if (copies) {
                CHECK(cudaMemsetAsync(y, 0, n * sizeof(int), stream));
                CHECK(cudaMemcpyAsync(y, x, n * sizeof(int), cudaMemcpyDeviceToDevice, stream));
            }
            values<<<17, 256, 0, stream>>>(x, y, n);
            CHECK(cudaGetLastError());
        }
        finish<<<1, 1, 0, stream>>>(handle, count, limit);
        CHECK(cudaStreamEndCapture(stream, &ended));
        cudaGraphExec_t executable;
        CHECK(cudaGraphInstantiate(&executable, graph, 0));
        for (int replay = 0; replay < 2; ++replay) {
            CHECK(cudaMemsetAsync(y, 0, n * sizeof(int), stream));
            if (argc > 3) CHECK(cudaDeviceSynchronize());
            CHECK(cudaGraphLaunch(executable, stream));
            CHECK(cudaStreamSynchronize(stream));
            int observed;
            CHECK(cudaMemcpy(&observed, count, sizeof(int), cudaMemcpyDeviceToHost));
            CHECK(cudaMemcpy(output.data(), y, n * sizeof(int), cudaMemcpyDeviceToHost));
            if (observed != limit) return 2;
            for (int i = 0; i < n; ++i) if (output[i] != (limit ? i + 7 : 0)) return 3;
            std::printf("copies=%d direct=%d limit=%d replay=%d pass\n", copies, direct, limit, replay);
            std::fflush(stdout);
        }
        CHECK(cudaGraphExecDestroy(executable));
        CHECK(cudaGraphDestroy(graph));
    }
    CHECK(cudaGraphDestroy(child));
    CHECK(cudaStreamDestroy(stream));
    CHECK(cudaFree(count));
    CHECK(cudaFree(y));
    CHECK(cudaFree(x));
}
