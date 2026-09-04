#pragma once

#include "spacepdhcg/cuda/device_scvx_driver_c_api.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

struct spacepdhcg_native_qoco;

struct spacepdhcg_native_qoco_report {
    double conversion_seconds;
    double setup_seconds;
    double update_seconds;
    double solve_seconds;
    double primal_residual;
    double dual_residual;
    std::uint64_t workspace_creations;
    std::uint64_t numeric_updates;
    std::uint64_t solves;
    std::uint64_t d2h_copy_count;
    std::uint64_t d2h_bytes;
    int iterations;
    int warm_primal_accepted;
    int dual_discarded;
    // Raw QOCO solve status of the last solve (1 solved, 2 solved inaccurate,
    // 3 numerical error, 4 max iterations; -1 before any solve).
    int status_code;
    // Ruiz equilibration iterations the solver was configured with (0 = none).
    int ruiz_iterations;
    spacepdhcg_cuda_qoco_failure failure;
};

// ``ruiz_iterations`` selects QOCO's own Ruiz equilibration (0 = off, the
// pinned QOCO commit's default). It is a solver setting, independent of the
// PDHCG workspace scaling mode.
spacepdhcg_cuda_status spacepdhcg_native_qoco_create(
    const spacepdhcg_cuda_scvx_problem* problem,
    cudaStream_t stream,
    int ruiz_iterations,
    spacepdhcg_native_qoco** workspace
);

spacepdhcg_cuda_status spacepdhcg_native_qoco_update_solve(
    spacepdhcg_native_qoco* workspace,
    const spacepdhcg_cuda_scvx_problem* problem,
    cudaStream_t stream,
    spacepdhcg_cuda_warm_start_mode requested_warm,
    double* device_primal,
    double* device_dual,
    spacepdhcg_native_qoco_report* report
);

void spacepdhcg_native_qoco_accept(spacepdhcg_native_qoco* workspace);
void spacepdhcg_native_qoco_reset_warm_state(
    spacepdhcg_native_qoco* workspace,
    bool retain_primal
);
void spacepdhcg_native_qoco_destroy(spacepdhcg_native_qoco* workspace);
