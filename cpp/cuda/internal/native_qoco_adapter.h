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
    /// Relative KKT residuals in the `canonical_residual_audit` normalisation used by the
    /// planner certificate and the CPU reference: `primal_residual` covers equality /
    /// primal-cone / dual-cone violation over (1 + |rhs| + |Ax|); `dual_residual` covers
    /// stationarity over (1 + |c| + |Px| + |A^T y|) and per-cone complementarity over the
    /// objective gap scale.  The driver's natural residual is max(primal, dual).
    double primal_residual;
    double dual_residual;
    /// Unnormalised counterparts kept for diagnostics (what this field pair reported before
    /// the relative audit; not comparable across families).
    double absolute_primal_residual;
    double absolute_dual_residual;
    double dual_cone_residual;
    double complementarity_residual;
    std::uint64_t workspace_creations;
    std::uint64_t numeric_updates;
    std::uint64_t solves;
    std::uint64_t d2h_copy_count;
    std::uint64_t d2h_bytes;
    int iterations;
    int warm_primal_accepted;
    int dual_discarded;
    /// 1 when the solve that produced the reported point ended QOCO_SOLVED_INACCURATE.
    int last_status_inaccurate;
    /// Warm-started solves that stalled inaccurate and were re-solved cold (cumulative).
    std::uint64_t warm_inaccurate_cold_retries;
    spacepdhcg_cuda_qoco_failure failure;
};

spacepdhcg_cuda_status spacepdhcg_native_qoco_create(
    const spacepdhcg_cuda_scvx_problem* problem,
    cudaStream_t stream,
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
