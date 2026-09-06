#include "../internal/scvx_gather.cuh"
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

using spacepdhcg::cuda::detail::gather_scvx_candidate_kernel;
using spacepdhcg::cuda::detail::scvx_gather_blocks;

namespace {
void require(bool condition, const char* message) {
    if (!condition) { std::fprintf(stderr, "FAIL: %s\n", message); std::exit(1); }
}
void check(cudaError_t status) { require(status == cudaSuccess, cudaGetErrorString(status)); }
template<class T> struct Buffer {
    T* ptr{};
    explicit Buffer(std::size_t count) { check(cudaMalloc(&ptr, count * sizeof(T))); }
    ~Buffer() { cudaFree(ptr); }
};

// Frozen pre-change implementation, used only by this comparison executable.
__global__ void single_block_gather(
    const double* primal, const int* si, const int* ci, double* s, double* c,
    std::size_t ns, std::size_t nc
) {
    for (std::size_t i = threadIdx.x; i < ns; i += blockDim.x) s[i] = primal[si[i]];
    for (std::size_t i = threadIdx.x; i < nc; i += blockDim.x) c[i] = primal[ci[i]];
}

struct Fixture {
    std::size_t ns, nc, source_count;
    Buffer<double> primal, states, controls;
    Buffer<int> si, ci;
    cudaStream_t stream{};
    std::vector<std::uint64_t> bits;
    std::vector<int> state_indices, control_indices;
    static constexpr std::uint64_t sentinel = 0xdeadbeef76543210ULL;
    Fixture(std::size_t states_count, std::size_t controls_count, bool shuffled)
        : ns(states_count), nc(controls_count), source_count(std::max(ns, nc) + 31),
          primal(source_count + 2), states(ns + 2), controls(nc + 2), si(ns + 2), ci(nc + 2),
          bits(source_count + 2), state_indices(ns + 2), control_indices(nc + 2) {
        check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        for (std::size_t i = 0; i < bits.size(); ++i) {
            // Include +/-zero, subnormals, infinities and NaN payloads. A gather
            // must preserve the bit pattern, not merely numerical equality.
            const std::uint64_t special[]{0, 0x8000000000000000ULL, 1,
                0x7ff0000000000000ULL, 0x7ff8000000000123ULL, 0xfff0000000000000ULL};
            bits[i] = i % 11 < 6 ? special[i % 11] : 0x3ff0000000000000ULL + i;
        }
        for (std::size_t i = 0; i < ns; ++i)
            state_indices[i + 1] = static_cast<int>((shuffled ? i * 8191 + i / 7 : i) % source_count);
        for (std::size_t i = 0; i < nc; ++i)
            control_indices[i + 1] = static_cast<int>((shuffled ? i * 4093 + i / 3 : i) % source_count);
        check(cudaMemcpyAsync(primal.ptr, bits.data(), bits.size() * sizeof(std::uint64_t),
                              cudaMemcpyHostToDevice, stream));
        check(cudaMemcpyAsync(si.ptr, state_indices.data(), state_indices.size() * sizeof(int),
                              cudaMemcpyHostToDevice, stream));
        check(cudaMemcpyAsync(ci.ptr, control_indices.data(), control_indices.size() * sizeof(int),
                              cudaMemcpyHostToDevice, stream));
        check(cudaStreamSynchronize(stream));
    }
    ~Fixture() { cudaStreamDestroy(stream); }
    void launch(unsigned blocks) {
        if (blocks == 0)
            single_block_gather<<<1, 256, 0, stream>>>(primal.ptr + 1, si.ptr + 1, ci.ptr + 1,
                states.ptr + 1, controls.ptr + 1, ns, nc);
        else
            gather_scvx_candidate_kernel<<<blocks, 256, 0, stream>>>(primal.ptr + 1, si.ptr + 1,
                ci.ptr + 1, states.ptr + 1, controls.ptr + 1, ns, nc);
        check(cudaGetLastError());
    }
    void verify(unsigned blocks) {
        std::vector<std::uint64_t> s(ns + 2, sentinel), c(nc + 2, sentinel);
        check(cudaMemcpyAsync(states.ptr, s.data(), s.size() * sizeof(std::uint64_t),
                              cudaMemcpyHostToDevice, stream));
        check(cudaMemcpyAsync(controls.ptr, c.data(), c.size() * sizeof(std::uint64_t),
                              cudaMemcpyHostToDevice, stream));
        launch(blocks);
        check(cudaMemcpyAsync(s.data(), states.ptr, s.size() * sizeof(std::uint64_t),
                              cudaMemcpyDeviceToHost, stream));
        check(cudaMemcpyAsync(c.data(), controls.ptr, c.size() * sizeof(std::uint64_t),
                              cudaMemcpyDeviceToHost, stream));
        check(cudaStreamSynchronize(stream));
        require(s.front() == sentinel && s.back() == sentinel && c.front() == sentinel
                    && c.back() == sentinel, "output guards intact");
        for (std::size_t i = 0; i < ns; ++i)
            require(s[i + 1] == bits[state_indices[i + 1] + 1], "state bit parity");
        for (std::size_t i = 0; i < nc; ++i)
            require(c[i + 1] == bits[control_indices[i + 1] + 1], "control bit parity");
    }
    double measure(unsigned blocks) {
        constexpr int launches = 64;
        cudaGraph_t graph{}; cudaGraphExec_t executable{};
        check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
        for (int i = 0; i < launches; ++i) launch(blocks);
        check(cudaStreamEndCapture(stream, &graph));
        check(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
        cudaEvent_t begin{}, end{}; check(cudaEventCreate(&begin)); check(cudaEventCreate(&end));
        check(cudaGraphLaunch(executable, stream)); check(cudaStreamSynchronize(stream));
        check(cudaEventRecord(begin, stream)); check(cudaGraphLaunch(executable, stream));
        check(cudaEventRecord(end, stream)); check(cudaEventSynchronize(end));
        float milliseconds{}; check(cudaEventElapsedTime(&milliseconds, begin, end));
        check(cudaEventDestroy(end)); check(cudaEventDestroy(begin));
        check(cudaGraphExecDestroy(executable)); check(cudaGraphDestroy(graph));
        return milliseconds * 1000.0 / launches;
    }
};
}

int main(int argc, char** argv) {
    const bool timings = argc > 1 && std::strcmp(argv[1], "measure") == 0;
    const std::pair<std::size_t, std::size_t> sizes[]{
        {0, 0}, {1, 0}, {0, 257}, {255, 257}, {294, 140}, {7014, 3500},
        {140014, 70000}, {1000003, 500003}};
    for (auto [ns, nc] : sizes) for (bool shuffled : {false, true}) {
        Fixture fixture(ns, nc, shuffled);
        const auto natural = static_cast<unsigned>((std::max(ns, nc) + 255) / 256);
        const unsigned candidates[]{0, std::max(1U, std::min(256U, natural)),
                                    std::max(1U, std::min(1024U, natural))};
        require(scvx_gather_blocks(ns, nc) == candidates[2], "production launch selects measured cap");
        require(scvx_gather_blocks(ns, nc, true) == 1, "single-block ablation selects one block");
        for (auto blocks : candidates) fixture.verify(blocks);
        std::printf("{\"case\":\"gather_parity\",\"states\":%zu,\"controls\":%zu,"
                    "\"shuffled\":%s,\"variants\":3,\"passed\":true}\n",
                    ns, nc, shuffled ? "true" : "false");
        if (timings && ns >= 294) for (int repeat = 0; repeat < 9; ++repeat)
            for (int i = 0; i < 3; ++i) {
                const auto variant = (repeat + i) % 3;
                const auto blocks = candidates[variant];
                const double us = fixture.measure(blocks);
                std::printf("{\"case\":\"gather_timing\",\"states\":%zu,\"controls\":%zu,"
                    "\"shuffled\":%s,\"repeat\":%d,\"warmup\":%s,\"variant\":%d,"
                    "\"blocks\":%u,\"microseconds\":%.9g}\n",
                    ns, nc, shuffled ? "true" : "false", repeat, repeat < 2 ? "true" : "false", variant, blocks, us);
            }
    }
    return 0;
}
