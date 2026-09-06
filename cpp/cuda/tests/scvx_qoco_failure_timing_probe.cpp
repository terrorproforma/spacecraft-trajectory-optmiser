// Test-only Linux interposer: fail after real, completed GPU inner work.
#include "../internal/native_qoco_adapter.h"
#include <dlfcn.h>
#include <cstdio>
#include <cstdlib>

spacepdhcg_cuda_status spacepdhcg_native_qoco_update_solve(
    spacepdhcg_native_qoco* workspace, const spacepdhcg_cuda_scvx_problem* problem,
    cudaStream_t stream, spacepdhcg_cuda_warm_start_mode warm,
    double* primal, double* dual, spacepdhcg_native_qoco_report* report
) {
    // This private C++ adapter uses the Linux Itanium ABI, unlike the public C API.
    const auto actual = reinterpret_cast<decltype(&spacepdhcg_native_qoco_update_solve)>(
        dlsym(RTLD_NEXT, "_Z35spacepdhcg_native_qoco_update_solveP22spacepdhcg_native_qoco"
              "PK28spacepdhcg_cuda_scvx_problemP11CUstream_st31spacepdhcg_cuda_warm_start_mode"
              "PdS7_P29spacepdhcg_native_qoco_report"));
    if (!actual) std::abort();
    const auto status = actual(workspace, problem, stream, warm, primal, dual, report);
    static int ordinal = 0;
    ++ordinal;
    const char* setting = std::getenv("SPACEPDHCG_TEST_FAIL_QOCO_ORDINAL");
    const bool injected = setting && std::atoi(setting) == ordinal
        && status == SPACEPDHCG_CUDA_SUCCESS;
    std::fprintf(stderr, "{\"case\":\"qoco_after_solve_fault\",\"ordinal\":%d,"
        "\"injected\":%s,\"actual_status\":%d,\"iterations\":%d,"
        "\"solve_seconds\":%.17g,\"update_seconds\":%.17g,\"residual_seconds\":%.17g}\n",
        ordinal, injected ? "true" : "false", int(status), report->iterations,
        report->solve_seconds, report->update_seconds, report->residual_seconds);
    if (injected) {
        report->failure = SPACEPDHCG_CUDA_QOCO_FAILURE_NUMERICAL;
        report->status_code = 3;
        return SPACEPDHCG_CUDA_NUMERICAL_FAILURE;
    }
    return status;
}
