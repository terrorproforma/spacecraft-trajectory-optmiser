#include "cuda_test_support.hpp"
#include "../internal/numeric_fingerprint.cuh"

#include <algorithm>
#include <array>

namespace test = spacepdhcg::cuda::test;
namespace detail = spacepdhcg::cuda::detail;

namespace {
// Original implementation is the benchmark/reference path.
__global__ void legacy_hash(const double* values, size_t elements,
                            unsigned long long tag, unsigned long long* fingerprint) {
    unsigned long long local = 0;
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < elements;
         i += blockDim.x * gridDim.x) {
        local ^= detail::mix_numeric_word(__double_as_longlong(values[i]), tag ^ i);
    }
    atomicXor(fingerprint, local);
}

unsigned long long cpu_hash(const std::vector<double>& values, unsigned long long tag) {
    unsigned long long result = 0;
    for (size_t i = 0; i < values.size(); ++i) {
        unsigned long long word;
        std::memcpy(&word, &values[i], sizeof(word));
        word ^= (tag ^ i) + 0x9e3779b97f4a7c15ULL + (word << 6U) + (word >> 2U);
        word ^= word >> 30U;
        word *= 0xbf58476d1ce4e5b9ULL;
        word ^= word >> 27U;
        word *= 0x94d049bb133111ebULL;
        result ^= word ^ (word >> 31U);
    }
    return result;
}

void check(size_t count, bool benchmark) {
    std::vector<double> values(count);
    for (size_t i = 0; i < count; ++i) values[i] = std::sin(static_cast<double>(i));
    if (count >= 5) {
        values[0] = 0.0;
        values[1] = -0.0;
        values[2] = INFINITY;
        values[3] = -INFINITY;
        values[4] = std::numeric_limits<double>::quiet_NaN();
    }
    test::CudaBuffer<double> device_values(count, false);
    test::CudaBuffer<unsigned long long> fingerprint(1, false);
    device_values.upload(values, nullptr);
    constexpr unsigned long long tag = 0x100000001b3ULL;
    const auto expected = cpu_hash(values, tag);
    const unsigned int blocks = static_cast<unsigned int>(std::max<size_t>(1, std::min<size_t>(256, (count + 255) / 256)));
    cudaEvent_t begin{}, end{};
    test::cuda_require(cudaEventCreate(&begin), "hash begin event");
    test::cuda_require(cudaEventCreate(&end), "hash end event");
    std::array<std::vector<double>, 2> times;
    const int repeats = benchmark ? 9 : 1;
    for (int repeat = 0; repeat < repeats; ++repeat) {
        for (int order = 0; order < 2; ++order) {
            const int variant = repeat % 2 == 0 ? order : 1 - order;
            test::cuda_require(cudaMemset(fingerprint.get(), 0, sizeof(unsigned long long)), "clear fingerprint");
            test::cuda_require(cudaEventRecord(begin), "start hash timing");
            // Odd repetition count also checks that XOR composition remains exact.
            const int launches = benchmark ? 101 : 1;
            for (int launch = 0; launch < launches; ++launch) {
                if (variant == 0) legacy_hash<<<blocks, 256>>>(device_values.get(), count, tag, fingerprint.get());
                else detail::hash_numeric_kernel<<<blocks, 256>>>(device_values.get(), count, tag, fingerprint.get());
            }
            test::cuda_require(cudaGetLastError(), "hash launch");
            test::cuda_require(cudaEventRecord(end), "stop hash timing");
            test::cuda_require(cudaEventSynchronize(end), "hash wait");
            float ms = 0;
            test::cuda_require(cudaEventElapsedTime(&ms, begin, end), "hash elapsed");
            test::require(fingerprint.download(nullptr)[0] == expected, "bitwise CPU/legacy/parallel hash parity");
            if (repeat >= 2) times[variant].push_back(ms / launches);
        }
    }
    if (benchmark) {
        for (auto& samples : times) std::sort(samples.begin(), samples.end());
        std::printf("{\"case\":\"numeric_fingerprint\",\"elements\":%zu,\"legacy_ms\":%.9g,\"block_reduce_ms\":%.9g,\"ratio\":%.6g}\n",
            count, times[0][3], times[1][3], times[0][3] / times[1][3]);
    }
    test::cuda_require(cudaEventDestroy(begin), "destroy hash begin");
    test::cuda_require(cudaEventDestroy(end), "destroy hash end");
}
}  // namespace

int main(int argc, char** argv) {
    const bool benchmark = argc == 2 && std::strcmp(argv[1], "--benchmark") == 0;
    test::require(argc == 1 || benchmark, "expected optional --benchmark");
    for (size_t count : {0U, 1U, 5U, 31U, 33U, 255U, 257U, 65539U}) check(count, false);
    if (benchmark) {
        for (size_t count : {18006U, 120012U, 600012U, 1048576U}) check(count, true);
    }
    std::puts("numeric fingerprint parity passed");
}
