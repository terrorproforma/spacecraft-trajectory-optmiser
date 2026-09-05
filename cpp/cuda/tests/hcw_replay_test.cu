#include "cuda_test_support.hpp"
#include "../internal/hcw_replay.cuh"

#include <algorithm>
#include <array>

namespace test = spacepdhcg::cuda::test;
namespace detail = spacepdhcg::cuda::detail;

namespace {
void check(size_t intervals, double step, bool zero, bool benchmark) {
    std::vector<double> initial{0.3, -0.4, 0.7, 0.002, -0.003, 0.001};
    std::vector<double> controls(intervals * 3);
    if (zero) std::fill(initial.begin(), initial.end(), 0.0);
    for (size_t i = 0; i < controls.size(); ++i) {
        controls[i] = zero ? 0.0 : 1e-5 * std::sin(i * 0.29);
    }
    test::CudaBuffer<double> di(6, false), dc(controls.size(), false);
    test::CudaBuffer<double> serial((intervals + 1) * 6, false), parallel((intervals + 1) * 6, false);
    di.upload(initial, nullptr); dc.upload(controls, nullptr);
    spacepdhcg_cuda_dynamics_config config{};
    config.model = SPACEPDHCG_CUDA_DYNAMICS_HCW;
    config.mean_motion = 0.00113;
    config.step_seconds = step;
    cudaEvent_t begin{}, end{};
    test::cuda_require(cudaEventCreate(&begin), "replay begin event");
    test::cuda_require(cudaEventCreate(&end), "replay end event");
    std::array<std::vector<double>, 2> times;
    const int repeats = benchmark ? 9 : 1;
    for (int repeat = 0; repeat < repeats; ++repeat) {
        for (int order = 0; order < 2; ++order) {
            const int variant = repeat % 2 == 0 ? order : 1 - order;
            test::cuda_require(cudaEventRecord(begin), "replay start");
            const int launches = benchmark ? 11 : 1;
            for (int launch = 0; launch < launches; ++launch) {
                if (variant == 0) detail::hcw_replay_kernel<false><<<1, 1>>>(
                    di.get(), dc.get(), serial.get(), intervals, config);
                else detail::hcw_replay_kernel<true><<<1, 32>>>(
                    di.get(), dc.get(), parallel.get(), intervals, config);
            }
            test::cuda_require(cudaGetLastError(), "replay launch");
            test::cuda_require(cudaEventRecord(end), "replay stop");
            test::cuda_require(cudaEventSynchronize(end), "replay wait");
            float ms = 0;
            test::cuda_require(cudaEventElapsedTime(&ms, begin, end), "replay elapsed");
            if (repeat >= 2) times[variant].push_back(ms / launches);
        }
        const auto expected = serial.download(nullptr), actual = parallel.download(nullptr);
        for (size_t i = 0; i < actual.size(); ++i) {
            if (actual[i] != expected[i]) {
                std::fprintf(stderr, "replay mismatch N=%zu step=%g index=%zu: %.17g != %.17g\n",
                    intervals, step, i, actual[i], expected[i]);
            }
            test::require(std::isfinite(actual[i]) && actual[i] == expected[i], "exact HCW recurrence parity");
        }
    }
    if (benchmark) {
        for (auto& samples : times) std::sort(samples.begin(), samples.end());
        std::printf("{\"case\":\"hcw_replay\",\"intervals\":%zu,\"serial_ms\":%.9g,\"warp_ms\":%.9g,\"ratio\":%.6g}\n",
            intervals, times[0][3], times[1][3], times[0][3] / times[1][3]);
    }
    test::cuda_require(cudaEventDestroy(begin), "replay destroy begin");
    test::cuda_require(cudaEventDestroy(end), "replay destroy end");
}
}  // namespace

int main(int argc, char** argv) {
    const bool benchmark = argc == 2 && std::strcmp(argv[1], "--benchmark") == 0;
    test::require(argc == 1 || benchmark, "expected optional --benchmark");
    for (size_t count : {0U, 1U, 33U, 257U, 2000U, 10000U}) {
        for (double step : {0.1, 2.0, 10.0}) check(count, step, false, false);
        check(count, 2.0, true, false);
    }
    if (benchmark) for (size_t count : {33U, 2000U, 10000U}) check(count, 2.0, false, true);
    std::puts("HCW serial/warp replay parity passed");
}
