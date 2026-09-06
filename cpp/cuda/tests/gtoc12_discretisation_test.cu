#include "spacepdhcg/cuda/gtoc12_discretisation_c_api.h"
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {
void require(bool ok, const char* message) {
    if (!ok) { std::fprintf(stderr, "%s\n", message); std::exit(1); }
}
void check(cudaError_t status) {
    require(status == cudaSuccess, cudaGetErrorString(status));
}
void run(int hold) {
    constexpr int n = 33;
    constexpr double flow = 0.003, kappa = 0.02;
    const int stencil = hold ? 4 : 1;
    std::vector<double> times(n + 1), states((n + 1) * 7), controls((n + 1) * 4);
    for (int i = 0; i <= n; ++i) {
        times[i] = 2.0 + 0.01*i;
        states[i * 7] = 2.7; states[i * 7 + 4] = 0.6; states[i * 7 + 6] = 0.9;
        controls[i * 4 + 3] = 0.8;
    }
    spacepdhcg_gtoc12_discretisation* w = nullptr;
    require(spacepdhcg_gtoc12_discretisation_create(n, hold, kappa, flow, times.data(), &w) == 0, "create");
    const double *a, *b, *c, *propagated;
    const int* invalid;
    require(spacepdhcg_gtoc12_discretisation_outputs(w, &a, &b, &c, &propagated, &invalid) == 0, "outputs");
    double *device_states, *device_controls;
    check(cudaMalloc(&device_states, states.size() * sizeof(double)));
    check(cudaMalloc(&device_controls, controls.size() * sizeof(double)));
    cudaStream_t stream;
    check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    check(cudaMemcpyAsync(device_states, states.data(), states.size() * sizeof(double), cudaMemcpyHostToDevice, stream));
    check(cudaMemcpyAsync(device_controls, controls.data(), controls.size() * sizeof(double), cudaMemcpyHostToDevice, stream));
    check(cudaStreamSynchronize(stream));
    // A device launch may be captured: it must not allocate, download, or
    // synchronize the host. The input storage is updated between graph replays.
    cudaGraph_t graph;
    cudaGraphExec_t executable;
    check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    require(spacepdhcg_gtoc12_discretisation_launch_device(w, device_states, device_controls, 8, 1, stream) == 0, "capture launch");
    check(cudaStreamEndCapture(stream, &graph));
    check(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
    std::vector<double> result(n * 7), aa(n * 49), bb(n * stencil * 28), cc(n * 7);
    for (int repeat = 0; repeat < 12; ++repeat) {
        const double thrust = repeat * 0.01;
        for (int i = 0; i <= n; ++i) controls[i * 4] = thrust;
        check(cudaMemcpyAsync(device_controls, controls.data(), controls.size() * sizeof(double), cudaMemcpyHostToDevice, stream));
        check(cudaGraphLaunch(executable, stream));
        int bad = -1;
        check(cudaMemcpyAsync(result.data(), propagated, result.size() * sizeof(double), cudaMemcpyDeviceToHost, stream));
        check(cudaMemcpyAsync(aa.data(), a, aa.size() * sizeof(double), cudaMemcpyDeviceToHost, stream));
        check(cudaMemcpyAsync(bb.data(), b, bb.size() * sizeof(double), cudaMemcpyDeviceToHost, stream));
        check(cudaMemcpyAsync(cc.data(), c, cc.size() * sizeof(double), cudaMemcpyDeviceToHost, stream));
        check(cudaMemcpyAsync(&bad, invalid, sizeof(int), cudaMemcpyDeviceToHost, stream));
        check(cudaStreamSynchronize(stream));
        require(bad == 0, "finite dynamics");
        for (int i = 0; i < n; ++i) {
            require(std::abs(result[i * 7 + 6] - (0.9 - flow * thrust * (times[i + 1] - times[i]))) < 3e-14,
                    "independent constant-thrust mass equation");
            const int first = hold ? std::min(std::max(i - 1, 0), n - 3) : i;
            for (int row = 0; row < 7; ++row) {
                double reconstructed = cc[i * 7 + row];
                for (int col = 0; col < 7; ++col) reconstructed += aa[i * 49 + row * 7 + col] * states[i * 7 + col];
                for (int s = 0; s < stencil; ++s)
                    for (int col = 0; col < 4; ++col)
                        reconstructed += bb[i * stencil * 28 + s * 28 + row * 4 + col] * controls[(first + s) * 4 + col];
                require(std::abs(reconstructed - result[i * 7 + row]) < 3e-13, "affine closure");
            }
        }
    }
    require(spacepdhcg_gtoc12_discretisation_launch_device(w, device_states, device_controls, 0, 1, stream) == 1, "reject zero substeps");
    check(cudaGraphExecDestroy(executable)); check(cudaGraphDestroy(graph));
    check(cudaStreamDestroy(stream)); check(cudaFree(device_states)); check(cudaFree(device_controls));
    spacepdhcg_gtoc12_discretisation_destroy(w);
    times[1] = times[0];
    w = nullptr;
    require(spacepdhcg_gtoc12_discretisation_create(n, hold, kappa, flow, times.data(), &w) == 1 && !w, "reject repeated time");
}
}
int main() {
    run(0); run(1);
    std::puts("GTOC12 device API: both holds, 24 changing-input graph replays, analytic mass and affine closure pass");
}
