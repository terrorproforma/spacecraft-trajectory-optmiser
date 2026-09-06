// Test-only interposer: cancel after setup, before the first outer iteration.
// A feasible retained trajectory must not turn cancellation into convergence.
#include "spacepdhcg/cuda/device_scvx_driver_c_api.h"
#include <dlfcn.h>
#include <cstdio>
#include <cstdlib>

extern "C" spacepdhcg_cuda_status spacepdhcg_cuda_scvx_driver_solve(
    spacepdhcg_cuda_scvx_driver* driver,
    const spacepdhcg_accelerator_stream stream,
    spacepdhcg_cuda_scvx_iteration* iterations,
    const size_t capacity,
    spacepdhcg_cuda_scvx_result* result
) {
    const auto actual = reinterpret_cast<decltype(&spacepdhcg_cuda_scvx_driver_solve)>(
        dlsym(RTLD_NEXT, "spacepdhcg_cuda_scvx_driver_solve"));
    const auto cancel = reinterpret_cast<decltype(&spacepdhcg_cuda_scvx_driver_cancel)>(
        dlsym(RTLD_NEXT, "spacepdhcg_cuda_scvx_driver_cancel"));
    if (!actual || !cancel || cancel(driver) != SPACEPDHCG_CUDA_SUCCESS) std::abort();
    const auto status = actual(driver, stream, iterations, capacity, result);
    std::fprintf(stderr, "{\"case\":\"cancel_before_solve\",\"api\":%d,"
                 "\"solver_status\":%d,\"outer_iterations\":%u}\n",
                 int(status), int(result->status), result->outer_iterations);
    return status;
}
