// SPDX-License-Identifier: Apache-2.0
// Diagnostic only: completion fences make these setup stages additive wall
// intervals, but perturb overlap. Never use this mode for speedup measurements.
#include <chrono>
#include <cstdlib>
extern "C" int qoco_gpu_setup_profiling() {
    const char* mode = std::getenv("SPACEPDHCG_TEST_QOCO_SETUP_PROFILE");
    return mode && mode[0] == '1';
}
extern "C" void qoco_gpu_setup_mark(const char* stage) {
    if (!qoco_gpu_setup_profiling()) return;
    static thread_local std::chrono::steady_clock::time_point previous;
    CUDA_CHECK(cudaDeviceSynchronize());
    const auto now = std::chrono::steady_clock::now();
    if (stage) {
        const double seconds = std::chrono::duration<double>(now - previous).count();
        fprintf(stderr, "{\"case\":\"qoco_setup_stage\",\"stage\":\"%s\",\"seconds\":%.9g}\n", stage, seconds);
    }
    previous = std::chrono::steady_clock::now();
}
