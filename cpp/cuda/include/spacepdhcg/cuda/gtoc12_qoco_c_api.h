#pragma once
#include "spacepdhcg/cuda/gtoc12_conic_c_api.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif
typedef struct spacepdhcg_gtoc12_qoco spacepdhcg_gtoc12_qoco;
typedef struct spacepdhcg_gtoc12_qoco_report {
    int qualified, qoco_status, iterations, failure;
    double requested_tolerance, primal_residual, dual_residual;
    double absolute_primal_residual, absolute_dual_residual;
    double setup_seconds, update_seconds, solve_seconds, residual_seconds;
    uint64_t workspace_creations, numeric_updates, device_numeric_updates, solves;
    uint64_t adapter_d2h_count, adapter_d2h_bytes;
    double primal_objective, dual_objective, absolute_gap, relative_gap;
} spacepdhcg_gtoc12_qoco_report;

/* GTOC12 native interval + conic assembly + GPU QOCO solve. Cold starts match
 * the reference's independent subproblem solves. Requires the patched GPU
 * QOCO device IO/update/solution/reduction extensions; no CPU solver fallback.
 * Setup/conversion still perform host metadata work and initial matrix copies.
 * Successful numerical updates and solves keep matrices/solutions on device;
 * QOCO control flow and diagnostic scalar downloads remain host-driven.
 * Status0 qualified,1 invalid args,2 runtime/allocation failure,3 invalid input,
 * 4 unqualified conic solve,5 unavailable/unsupported QOCO extension.
 * Qualification means raw solver status1/2 and finite audited primal/dual
 * residuals and global relative objective gap <= requested tolerance, where
 * relative gap = |primal-dual| / max(1,|primal|,|dual|). It is NOT a nonlinear
 * physics certificate. The gap is independently reduced from original device
 * matrices and unscaled primal/dual vectors; four more scalars are downloaded.
 */
int spacepdhcg_gtoc12_qoco_create(int intervals, int hold, int free_departure,
    int free_arrival, double kappa, double mass_flow, const double* times,
    const double* boundary, const double* fuel_weights, double tolerance,
    int ruiz_iterations, spacepdhcg_gtoc12_qoco**);
void spacepdhcg_gtoc12_qoco_destroy(spacepdhcg_gtoc12_qoco*);
int spacepdhcg_gtoc12_qoco_get_dimensions(spacepdhcg_gtoc12_qoco*,
    spacepdhcg_gtoc12_conic_dimensions*);
/* Retained device primal output; only valid after qualified=1. */
int spacepdhcg_gtoc12_qoco_primal(spacepdhcg_gtoc12_qoco*, const double**);
/* Device inputs; synchronous native orchestration, NOT graph-capturable.
 * All external stream work must finish before workspace reuse/destruction.
 * Instance is serialized and device-bound. No mid-solve cancellation yet.
 */
int spacepdhcg_gtoc12_qoco_solve_device(spacepdhcg_gtoc12_qoco*,
    const double* states, const double* controls,
    const spacepdhcg_gtoc12_conic_parameters*, int substeps, void* stream,
    spacepdhcg_gtoc12_qoco_report*);
/* Host callback enqueues GPU work on the supplied stream; it must not throw.
 * report and primal are DEVICE pointers, borrowed until next solve. Only
 * qualification/status/iterations/tolerance/residuals/objectives are populated
 * in the device report; timing/counter/failure fields are host reporting only.
 * The callback must gate acceptance of primal on device report->qualified.
 * Return 0 on successful submission, nonzero on error. The solve drains work
 * before returning. consumed is set only after successful callback submission;
 * failed synchronous priming does not invoke it. Ordinary replay invokes it
 * before host status collection, including unqualified numerical outcomes.
 */
typedef int (*spacepdhcg_gtoc12_qoco_consumer)(void* context,
    const spacepdhcg_gtoc12_qoco_report* device_report, const double* device_primal,
    void* stream);
int spacepdhcg_gtoc12_qoco_solve_device_with_consumer(spacepdhcg_gtoc12_qoco*,
    const double* states, const double* controls,
    const spacepdhcg_gtoc12_conic_parameters*, int substeps, void* stream,
    spacepdhcg_gtoc12_qoco_report*, spacepdhcg_gtoc12_qoco_consumer,
    void* context, int* consumed);
/* Same synchronous orchestration/consumer contract, with the integration step
 * count borrowed from device memory. The value is consumed on the stream, never
 * downloaded to choose a host launch. A device value<1 follows invalid-input
 * rejection (status3, unqualified); a null pointer is invalid arguments (1).
 * This bridge is still NOT graph-capturable; only its assembly stage is.
 */
int spacepdhcg_gtoc12_qoco_solve_controlled_device_with_consumer(spacepdhcg_gtoc12_qoco*,
    const double* states, const double* controls,
    const spacepdhcg_gtoc12_conic_parameters*, const int* device_substeps, void* stream,
    spacepdhcg_gtoc12_qoco_report*, spacepdhcg_gtoc12_qoco_consumer,
    void* context, int* consumed);
/* Deferred cold replay after synchronous workspace/graph/numeric priming.
 * can_enqueue returns a host readiness hint (0/1). Successful enqueue (0)
 * queues assembly, numeric update, IPM, audit, qualification and the required
 * consumer without report downloads or a wait. It does NOT mean qualified.
 * Device validation/numeric replay/IPM graph options must be enabled; verbose
 * and CPU comparison modes are ineligible. Unsupported readiness returns 5.
 * Inputs must live through finish; consumer outputs are borrowed until reuse.
 * One pending solve per instance; solve/enqueue cannot overwrite pending work.
 * Finish on the same host thread/device/stream applies the unchanged report
 * gates and returns the ordinary solve status. Destruction drains pending work.
 * Not graph-capturable; error exits may drain partially submitted work.
 */
int spacepdhcg_gtoc12_qoco_can_enqueue(spacepdhcg_gtoc12_qoco*);
int spacepdhcg_gtoc12_qoco_enqueue_controlled(spacepdhcg_gtoc12_qoco*,
    const double* states, const double* controls,
    const spacepdhcg_gtoc12_conic_parameters*, const int* device_substeps, void* stream,
    spacepdhcg_gtoc12_qoco_consumer, void* context);
int spacepdhcg_gtoc12_qoco_finish(spacepdhcg_gtoc12_qoco*, void* stream,
    spacepdhcg_gtoc12_qoco_report*);
/* Upload states/controls/parameters and download ONLY qualified primal output.
 * Host primal[variables] is untouched on unqualified solves. Report counters
 * describe the existing native adapter, excluding this bridge and opaque
 * QOCO-internal transfers. Callers must independently verify trajectory physics.
 */
int spacepdhcg_gtoc12_qoco_solve_host(spacepdhcg_gtoc12_qoco*,
    const double* states, const double* controls,
    const spacepdhcg_gtoc12_conic_parameters*, int substeps, double* primal,
    spacepdhcg_gtoc12_qoco_report*);

#ifdef __cplusplus
}
#endif
