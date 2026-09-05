#include "cuda_test_support.hpp"
#include "../internal/scvx_metrics.cuh"

#include <algorithm>
#include <array>
#include <numeric>

namespace test = spacepdhcg::cuda::test;
namespace detail = spacepdhcg::cuda::detail;

namespace {
void compare(const detail::ScvxMetrics& actual, const detail::ScvxMetrics& expected) {
#define CHECK_SUM(field) \
    test::require(std::abs(actual.field - expected.field) <= \
        3e-12 * std::max(1.0, std::abs(expected.field)), "metrics sum parity: " #field)
#define CHECK_MAX(field) \
    test::require(actual.field == expected.field, "metrics maximum parity: " #field)
    CHECK_SUM(objective); CHECK_SUM(merit); CHECK_SUM(model_merit);
    CHECK_MAX(dynamics); CHECK_MAX(path); CHECK_MAX(path_thrust);
    CHECK_MAX(path_mass); CHECK_MAX(path_altitude); CHECK_MAX(terminal);
    CHECK_MAX(virtual_control); CHECK_MAX(step); CHECK_MAX(thrust);
    CHECK_MAX(torque); CHECK_MAX(pointing); CHECK_MAX(mass);
    CHECK_MAX(altitude); CHECK_MAX(glide_slope); CHECK_MAX(angular_rate);
    CHECK_MAX(quaternion); CHECK_MAX(maximum_stage_trust_distance);
    CHECK_MAX(terminal_trust_distance);
#undef CHECK_SUM
#undef CHECK_MAX
}

void check(int model, size_t intervals, bool include_virtual, bool benchmark) {
    const bool hcw = model == SPACEPDHCG_CUDA_DYNAMICS_HCW;
    const bool pd6 = model == SPACEPDHCG_CUDA_DYNAMICS_POWERED_DESCENT_6DOF;
    const size_t nx = hcw ? 6U : (pd6 ? 14U : 7U);
    const size_t nu = hcw ? 3U : (pd6 ? 7U : 4U);
    const size_t ns = (intervals + 1) * nx, nc = intervals * nu;
    const size_t nv = intervals * nx;
    std::vector<double> states(ns), replay(ns), reference(ns), controls(nc), ref_controls(nc);
    std::vector<double> primal(nv), lower(ns + nc, 0.7), upper(ns + nc, 1.1), target(nx);
    std::vector<int> state_indices(ns), control_indices(nc), virtual_indices(nv);
    std::iota(state_indices.begin(), state_indices.end(), 0);
    std::iota(control_indices.begin(), control_indices.end(), static_cast<int>(ns));
    std::iota(virtual_indices.begin(), virtual_indices.end(), 0);
    // Distinct nonzero defects, constraint violations, and terminal errors catch
    // omitted partitions, duplicated terminal sums, and accidental max/sum mixing.
    for (size_t i = 0; i < ns; ++i) {
        states[i] = std::sin(i * 0.31) * 1.7;
        replay[i] = states[i] + std::cos(i * 0.17) * 0.3;
        reference[i] = states[i] - std::sin(i * 0.23) * 0.4;
    }
    for (size_t i = 0; i < nc; ++i) {
        controls[i] = std::cos(i * 0.29) * 2.3;
        ref_controls[i] = controls[i] + std::sin(i * 0.41);
    }
    for (size_t i = 0; i < nv; ++i) primal[i] = std::sin(i * 0.43) * 0.07;
    for (size_t i = 0; i < nx; ++i) target[i] = i * 0.13 - 0.9;
    test::CudaBuffer<double> ds(ns, false), dr(ns, false), dref(ns, false), dc(nc, false), drc(nc, false);
    test::CudaBuffer<double> dp(nv, false), dl(ns + nc, false), du(ns + nc, false), dt(nx, false);
    test::CudaBuffer<int> dsi(ns, false), dci(nc, false), dvi(nv, false);
    ds.upload(states, nullptr); dr.upload(replay, nullptr); dref.upload(reference, nullptr);
    dc.upload(controls, nullptr); drc.upload(ref_controls, nullptr); dp.upload(primal, nullptr);
    dl.upload(lower, nullptr); du.upload(upper, nullptr); dt.upload(target, nullptr);
    dsi.upload(state_indices, nullptr); dci.upload(control_indices, nullptr); dvi.upload(virtual_indices, nullptr);
    spacepdhcg_cuda_scvx_numeric_update update{};
    for (size_t i = 0; i < nx; ++i) update.state_trust_scales[i] = 0.2 + i * 0.03;
    for (size_t i = 0; i < nu; ++i) update.control_trust_scales[i] = 0.1 + i * 0.02;
    update.maximum_thrust = 1.1;
    update.maximum_torque = 0.8;
    update.maximum_angular_rate = 0.7;
    update.tilt_cosine = 0.9;
    update.glide_slope_tangent = 1.3;
    update.minimum_radius = 2.5;
    // The legacy radial merit uses scalar_lower separately; preserve both terms.
    update.radial_row_start = 0;
    const unsigned int blocks = static_cast<unsigned int>(std::min<size_t>(128, (nv + 255) / 256));
    test::CudaBuffer<detail::ScvxMetrics> expected(1, false), actual(1, false), partials(blocks, false);
#define METRIC_ARGS(output) ds.get(), dc.get(), dr.get(), dref.get(), drc.get(), dt.get(), \
    dp.get(), include_virtual ? dvi.get() : nullptr, include_virtual ? nv : 0U, \
    intervals, nx, nu, model, 11.0, 7.0, 0.25, update, dl.get(), dl.get(), du.get(), \
    dsi.get(), dci.get(), output
    cudaEvent_t begin{}, end{};
    test::cuda_require(cudaEventCreate(&begin), "metrics begin event");
    test::cuda_require(cudaEventCreate(&end), "metrics end event");
    std::array<std::vector<double>, 2> times;
    const int repeats = benchmark ? 9 : 1;
    for (int repeat = 0; repeat < repeats; ++repeat) {
        for (int order = 0; order < 2; ++order) {
            const int variant = repeat % 2 == 0 ? order : 1 - order;
            test::cuda_require(cudaEventRecord(begin), "metrics start");
            const int launches = benchmark ? 11 : 1;
            for (int launch = 0; launch < launches; ++launch) {
                if (variant == 0) {
                    detail::scvx_metrics_kernel<false><<<1, 1>>>(METRIC_ARGS(expected.get()));
                } else {
                    detail::scvx_metrics_kernel<true><<<blocks, 256>>>(METRIC_ARGS(partials.get()));
                    detail::finish_scvx_metrics_kernel<<<1, 256>>>(partials.get(), blocks, actual.get());
                }
            }
            test::cuda_require(cudaGetLastError(), "metrics launch");
            test::cuda_require(cudaEventRecord(end), "metrics stop");
            test::cuda_require(cudaEventSynchronize(end), "metrics wait");
            float ms = 0;
            test::cuda_require(cudaEventElapsedTime(&ms, begin, end), "metrics elapsed");
            if (repeat >= 2) times[variant].push_back(ms / launches);
        }
        compare(actual.download(nullptr)[0], expected.download(nullptr)[0]);
    }
#undef METRIC_ARGS
    if (benchmark) {
        for (auto& samples : times) std::sort(samples.begin(), samples.end());
        std::printf("{\"case\":\"scvx_metrics\",\"model\":%d,\"intervals\":%zu,\"serial_ms\":%.9g,\"parallel_ms\":%.9g,\"ratio\":%.6g}\n",
            model, intervals, times[0][3], times[1][3], times[0][3] / times[1][3]);
    }
    test::cuda_require(cudaEventDestroy(begin), "metrics destroy begin");
    test::cuda_require(cudaEventDestroy(end), "metrics destroy end");
}
}  // namespace

int main(int argc, char** argv) {
    const bool benchmark = argc == 2 && std::strcmp(argv[1], "--benchmark") == 0;
    test::require(argc == 1 || benchmark, "expected optional --benchmark");
    for (int model : {SPACEPDHCG_CUDA_DYNAMICS_HCW, SPACEPDHCG_CUDA_DYNAMICS_LOW_THRUST,
                     SPACEPDHCG_CUDA_DYNAMICS_POWERED_DESCENT_3DOF,
                     SPACEPDHCG_CUDA_DYNAMICS_POWERED_DESCENT_6DOF}) {
        for (size_t count : {1U, 33U, 257U, 2000U}) {
            check(model, count, false, false);
            check(model, count, true, false);
        }
        if (benchmark) for (size_t count : {33U, 2000U, 10000U}) check(model, count, true, true);
    }
    std::puts("SCvx serial/parallel metrics parity passed for all four physics families");
}
