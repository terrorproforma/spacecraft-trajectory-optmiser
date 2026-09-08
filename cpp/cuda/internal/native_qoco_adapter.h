#pragma once

#include "spacepdhcg/cuda/device_scvx_driver_c_api.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

struct spacepdhcg_native_qoco;
struct QocoAuditResult;
struct QocoReplayStatus;
struct QocoGraphProgress;

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
    // Graph executions have a device ledger. Legacy phase timings above cover
    // imperative calls only; the graph owner measures complete graph duration.
    std::uint64_t graph_attempts, graph_iterations;
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

// Split cold replay. Requires a successfully primed device numeric/replay
// workspace and device validation, with comparison/verbose modes disabled.
// can_enqueue is a host readiness check; a stale vendor graph may still reject
// submission. Enqueue success means submission only, never qualification.
// No successful enqueue downloads or waits. Error exits may drain partial work.
// One pending solve per instance: no solve/accept/reset until finish. Borrowed
// inputs/outputs must remain alive; consumers run on the submission stream.
// Finish on the same host thread/device/stream collects the original audit and
// applies unchanged status gates. Destruction drains pending work. Not capturable.
bool spacepdhcg_native_qoco_can_enqueue(const spacepdhcg_native_qoco*);
// Restart an independently initialized leg using retained symbolic/vendor storage.
// No graphs/borrows may survive. Forces numerical refresh before any replay.
spacepdhcg_cuda_status spacepdhcg_native_qoco_restart_leg(spacepdhcg_native_qoco*);
// Borrow the native SCvx inaccurate-retry counter until explicitly cleared.
// Caller synchronizes work before rebinding; zero-Ruiz contexts only.
spacepdhcg_cuda_status spacepdhcg_native_qoco_set_conditioning_retry(
    spacepdhcg_native_qoco*, const int* device_retry);
spacepdhcg_cuda_status spacepdhcg_native_qoco_enqueue(
    spacepdhcg_native_qoco*, const spacepdhcg_cuda_scvx_problem*, cudaStream_t,
    double* device_primal, double* device_dual, spacepdhcg_native_qoco_consumer,
    void* context, const int* producer_invalid);
spacepdhcg_cuda_status spacepdhcg_native_qoco_finish(
    spacepdhcg_native_qoco*, cudaStream_t, spacepdhcg_native_qoco_report*);
// Opt-in cold-solve coordinates. Copies the device prefix into retained storage;
// no borrowed origin survives this stream's work. Original audit/output units
// are preserved. Call before each cold solve, never while pending or warm.
spacepdhcg_cuda_status spacepdhcg_native_qoco_set_origin(
    spacepdhcg_native_qoco*, const double* device_origin, int count, cudaStream_t);

// Exclusive external-graph lease. Prime cold numeric replay first. All graphs
// and executables emitted during a lease must be destroyed before end_graph;
// execute only on this stream/thread/device, serialize launches and retain all
// inputs/outputs. Other solve/accept/reset/origin APIs reject while leased.
// destroy while leased defers deletion until end_graph. End drains the stream,
// collects the device ledger once, and forces fresh setup before imperative use.
// The borrowed progress is valid until end_graph and counts actual executions.
spacepdhcg_cuda_status spacepdhcg_native_qoco_begin_graph(
    spacepdhcg_native_qoco*,cudaStream_t,const QocoGraphProgress**);
spacepdhcg_cuda_status spacepdhcg_native_qoco_emit_graph(
    spacepdhcg_native_qoco*,const spacepdhcg_cuda_scvx_problem*,cudaGraph_t,
    const cudaGraphNode_t*,std::size_t,cudaStream_t,double* primal,double* dual,
    const double* origin,int origin_count,const int* producer_invalid,
    spacepdhcg_native_qoco_consumer,void*,cudaGraphNode_t* completion);
spacepdhcg_cuda_status spacepdhcg_native_qoco_end_graph(
    spacepdhcg_native_qoco*,cudaStream_t,spacepdhcg_native_qoco_report*);
