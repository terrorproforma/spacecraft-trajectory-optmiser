#pragma once

#include "spacepdhcg/cuda/device_scvx_driver_c_api.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

struct spacepdhcg_native_qoco;
struct QocoAuditResult;
struct QocoReplayStatus;

// Called on the host to enqueue consumers of borrowed device outputs on the
// supplied stream. It must not throw or retain pointers past the next solve.
using spacepdhcg_native_qoco_consumer = cudaError_t (*)(void*,
    const QocoReplayStatus*, const QocoAuditResult*, cudaStream_t);

struct spacepdhcg_native_qoco_report {
    double conversion_seconds;
    double setup_seconds;
    double update_seconds;
    double solve_seconds;
    double residual_seconds;
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
    std::uint64_t device_numeric_updates;
    std::uint64_t solves;
    std::uint64_t d2h_copy_count;
    std::uint64_t d2h_bytes;
    // Adapter-owned transfers; opaque QOCO-internal transfers are separate.
    std::uint64_t h2d_copy_count;
    std::uint64_t h2d_bytes;
    std::uint64_t d2d_copy_count;
    std::uint64_t d2d_bytes;
    // Audit/conversion/topology memory, including temporary setup scratch in the peak.
    std::uint64_t audit_allocations;
    std::uint64_t audit_peak_bytes;
    int iterations;
    int warm_primal_accepted;
    int dual_discarded;
    // Raw QOCO solve status of the last solve (1 solved, 2 solved inaccurate,
    // 3 numerical error, 4 max iterations; -1 before any solve).
    int status_code;
    // Ruiz equilibration iterations the solver was configured with (0 = none).
    int ruiz_iterations;
    /// 1 when the solve that produced the reported point ended QOCO_SOLVED_INACCURATE.
    int last_status_inaccurate;
    /// Warm-started solves that stalled inaccurate and were re-solved cold (cumulative).
    std::uint64_t warm_inaccurate_cold_retries;
    spacepdhcg_cuda_qoco_failure failure;
    // Internal producer guard telemetry; not part of the public GTOC12 C ABI.
    int producer_invalid, producer_validation_queued;
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

// Explicit solver accuracy and required device extensions for new consumers.
// The legacy create retains its 1e-8 settings and optional extension behavior.
// Success still requires caller-side qualification against reported KKT residuals.
spacepdhcg_cuda_status spacepdhcg_native_qoco_create_configured(
    const spacepdhcg_cuda_scvx_problem* problem, cudaStream_t stream,
    int ruiz_iterations, double tolerance, bool require_device_extensions,
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

spacepdhcg_cuda_status spacepdhcg_native_qoco_accept(
    spacepdhcg_native_qoco* workspace, spacepdhcg_native_qoco_report* report);
// Cold solve: replay consumers run before report collection. Initial/stale graph
// priming remains synchronous. No consumer is invoked on failed priming.
spacepdhcg_cuda_status spacepdhcg_native_qoco_update_solve_with_consumer(
    spacepdhcg_native_qoco*, const spacepdhcg_cuda_scvx_problem*, cudaStream_t,
    double* device_primal, double* device_dual, spacepdhcg_native_qoco_report*,
    spacepdhcg_native_qoco_consumer, void* context);
// Producer flag is a borrowed device int on the supplied stream. Any nonzero
// value must reject the solve, including when all canonical values are finite.
// Fallback/priming paths collect it before invoking a synchronous solver.
spacepdhcg_cuda_status spacepdhcg_native_qoco_update_solve_with_input_guard(
    spacepdhcg_native_qoco*, const spacepdhcg_cuda_scvx_problem*, cudaStream_t,
    double* device_primal, double* device_dual, spacepdhcg_native_qoco_report*,
    spacepdhcg_native_qoco_consumer, void* context, const int* producer_invalid);
spacepdhcg_cuda_status spacepdhcg_native_qoco_reset_warm_state(
    spacepdhcg_native_qoco* workspace,
    bool retain_primal
);
void spacepdhcg_native_qoco_destroy(spacepdhcg_native_qoco* workspace);
