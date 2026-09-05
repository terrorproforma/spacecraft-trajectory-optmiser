// Standalone test against the explicitly prepared QOCO CUDA library.
#include "qoco.h"
#include "../algebra/cuda/cuda_types.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <dlfcn.h>
#include <limits>
#include <vector>

static void require(bool value, const char* message) {
    if (!value) { std::fprintf(stderr, "FAIL: %s\n", message); std::exit(1); }
}
static void check(cudaError_t status) { require(status == cudaSuccess, cudaGetErrorString(status)); }

static void run_case(int n, int special) {
    std::vector<double> values(n);
    for (int i = 0; i < n; ++i) values[i] = std::sin(i * 0.37) * (i % 101 + 1);
    if (n) {
        values.back() = -1024.0;
        values[n / 2] = -std::numeric_limits<double>::denorm_min();
        if (special == 1) values.back() = std::numeric_limits<double>::quiet_NaN();
        if (special == 2) std::fill(values.begin(), values.end(), std::numeric_limits<double>::infinity());
        if (special == 3) std::fill(values.begin(), values.end(), -0.0);
    }
    double expected_max = 0.0, expected_min = n ? std::numeric_limits<double>::infinity() : QOCOFloat_MAX;
    for (double value : values) { expected_max = std::max(expected_max, std::abs(value)); expected_min = std::min(expected_min, std::abs(value)); }
    double* buffer{}; check(cudaMalloc(&buffer, (n + 2) * sizeof(double)));
    check(cudaMemset(buffer, 0, (n + 2) * sizeof(double)));
    if (n) check(cudaMemcpy(buffer + 1, values.data(), n * sizeof(double), cudaMemcpyHostToDevice));
    QOCOVectorf view{nullptr, buffer + 1, n};
    for (int repeat = 0; repeat < 3; ++repeat) {
        const double maximum = inf_norm(buffer + 1, n), minimum = min_abs_val(buffer + 1, n);
        require(check_nan(&view) == (special == 1), "NaN predicate");
        if (special == 1) require(std::isnan(maximum) && std::isnan(minimum), "norms preserve NaN failure");
        else {
            require(maximum == expected_max, "exact infinity norm");
            require(minimum == expected_min, "exact minimum absolute value");
            if (!special) {
                require(inf_norm(values.data(), n) == expected_max, "host infinity norm parity");
                require(min_abs_val(values.data(), n) == expected_min, "host minimum parity");
            }
        }
    }
    check(cudaFree(buffer));
}

int main() {
    auto begin = reinterpret_cast<int (*)()>(dlsym(RTLD_DEFAULT, "qoco_gpu_begin_reduction_scope"));
    auto end = reinterpret_cast<void (*)()>(dlsym(RTLD_DEFAULT, "qoco_gpu_end_reduction_scope"));
    require(begin && end, "complete scope API");
    for (int scoped = 0; scoped < 3; ++scoped) {
        if (scoped == 1) { require(begin() == 0, "scope begin"); require(begin() == 0, "nested scope begin"); }
        for (int n : {0, 1, 17, 255, 256, 257, 4096, 4097, 65537, 1048579}) run_case(n, 0);
        for (int special : {1, 2, 3}) for (int n : {17, 4097, 65537}) run_case(n, special);
        if (scoped == 1) { end(); end(); }
    }
    std::puts("GPU scalar reductions: exact extrema, tails, empty, subnormal, infinity, NaN, offset and nested/unscoped lifecycle PASS");
}
