// Test the actual IR decision/backup kernels without a conditional graph.
// Only cudaGraphSetConditional is replaced with an observable device flag:
// these checks do NOT validate graph execution or cuDSS memory ownership.
#include "cudss_backend.h"
#include <dlfcn.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <limits>
#define CUDA_CHECK(call) do { auto error = (call); if (error != cudaSuccess) { \
    std::fprintf(stderr, "%d: %s\n", __LINE__, cudaGetErrorString(error)); std::exit(1); } } while (0)
#define CUDSS_CHECK(call) do { if ((call) != CUDSS_STATUS_SUCCESS) std::exit(1); } while (0)
static CudaLibFuncs g_cuda_funcs{};
static void* g_cudss_handle{};
__device__ unsigned int observed_condition;
__device__ void test_set_condition(cudaGraphConditionalHandle, unsigned int value) {
    observed_condition = value;
}
#define cudaGraphSetConditional test_set_condition
#include "qoco_ir_runtime.cuh"
#undef cudaGraphSetConditional

struct Case {
    const char* name;
    double initial, tolerance;
    int maximum;
    std::vector<double> corrections;
    int accepted, attempted, restore, solution;
};

int main() {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double inf = std::numeric_limits<double>::infinity();
    const std::vector<Case> cases = {
        {"zero budget", 10, 1, 0, {}, 0, 0, 0, 0},
        {"initial convergence", 0.5, 1, 5, {}, 0, 0, 0, 0},
        {"tolerance equality and tie", 1, 1, 5, {1}, 0, 1, 1, 0},
        {"worse first correction", 10, 1, 5, {12}, 0, 1, 1, 0},
        {"improve then restore", 10, 1, 5, {4, 7}, 1, 2, 1, 1},
        {"improve then converge", 10, 3, 5, {4, 2}, 2, 2, 0, 2},
        {"iteration budget", 10, 1, 2, {8, 6}, 2, 2, 0, 2},
        {"initial NaN stops without accepting", nan, 1, 2, {}, 0, 0, 0, 0},
        {"initial infinity stops without accepting", inf, 1, 2, {}, 0, 0, 0, 0},
        {"NaN correction restores finite solution", 10, 1, 2, {nan}, 0, 1, 1, 0},
        {"infinite correction restores finite solution", 10, 1, 2, {inf}, 0, 1, 1, 0},
        {"improve then NaN restores best", 10, 1, 5, {4, nan}, 1, 2, 1, 1},
        {"improve then infinity restores best", 10, 1, 5, {4, inf}, 1, 2, 1, 1},
        {"NaN tolerance never starts", 10, nan, 2, {}, 0, 0, 0, 0},
    };
    constexpr int count = 4097;
    QocoIrState* state;
    double *norm, *x, *best, *scratch;
    CUDA_CHECK(cudaMalloc(&state, sizeof(QocoIrState)));
    CUDA_CHECK(cudaMalloc(&norm, sizeof(double)));
    CUDA_CHECK(cudaMalloc(&x, (count + 2) * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&best, (count + 2) * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&scratch, 257 * sizeof(double)));
    for (const auto& test : cases) {
        std::vector<double> host(count + 2, 0);
        host[count] = host[count + 1] = -12345;
        CUDA_CHECK(cudaMemcpy(x, host.data(), host.size() * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(best, x, host.size() * sizeof(double), cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(norm, &test.initial, sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(state, 0, sizeof(QocoIrState)));
        qoco_ir_initial<<<1, 1>>>(state, norm, 0, test.tolerance, test.maximum);
        CUDA_CHECK(cudaGetLastError());
        unsigned active;
        CUDA_CHECK(cudaMemcpyFromSymbol(&active, observed_condition, sizeof(active)));
        int step = 0;
        while (active) {
            if (step >= int(test.corrections.size())) return 2;
            for (int i = 0; i < count; ++i) host[i] = step + 1;
            CUDA_CHECK(cudaMemcpy(x, host.data(), host.size() * sizeof(double), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(norm, &test.corrections[step++], sizeof(double), cudaMemcpyHostToDevice));
            qoco_ir_decide<<<1, 1>>>(state, norm, 0, test.tolerance, test.maximum);
            qoco_ir_save_or_restore<<<17, 256>>>(state, x, best, count);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpyFromSymbol(&active, observed_condition, sizeof(active)));
        }
        QocoIrState observed;
        CUDA_CHECK(cudaMemcpy(&observed, state, sizeof(observed), cudaMemcpyDeviceToHost));
        if (observed.accepted != test.accepted || observed.attempted != test.attempted ||
            observed.restore != test.restore) return 3;
        for (const auto* buffer : {x, best}) {
            CUDA_CHECK(cudaMemcpy(host.data(), buffer, host.size() * sizeof(double), cudaMemcpyDeviceToHost));
            for (int i = 0; i < count; ++i) if (host[i] != test.solution) return 4;
            if (host[count] != -12345 || host[count + 1] != -12345) return 5;
        }
        std::printf("control: %s passed\n", test.name);
    }
    CUDA_CHECK(cudaFree(x));
    CUDA_CHECK(cudaMalloc(&x, 100003 * sizeof(double)));
    for (int length : {0, 1, 4096, 4097, 100003}) {
        std::vector<double> host(length, -3.5);
        CUDA_CHECK(cudaMemcpy(x, host.data(), length * sizeof(double), cudaMemcpyHostToDevice));
        qoco_ir_device_norm(x, length, scratch, nullptr);
        double result;
        CUDA_CHECK(cudaMemcpy(&result, scratch + 256, sizeof(double), cudaMemcpyDeviceToHost));
        if (result != (length ? 3.5 : 0.0)) return 6;
        if (length) {
            CUDA_CHECK(cudaMemcpy(x + length - 1, &nan, sizeof(double), cudaMemcpyHostToDevice));
            qoco_ir_device_norm(x, length, scratch, nullptr);
            CUDA_CHECK(cudaMemcpy(&result, scratch + 256, sizeof(double), cudaMemcpyDeviceToHost));
            if (!std::isnan(result)) return 7;
        }
    }
    CUDA_CHECK(cudaFree(scratch));
    CUDA_CHECK(cudaFree(best));
    CUDA_CHECK(cudaFree(x));
    CUDA_CHECK(cudaFree(norm));
    CUDA_CHECK(cudaFree(state));
    std::puts("14 control cases and 9 norm cases passed");
}
