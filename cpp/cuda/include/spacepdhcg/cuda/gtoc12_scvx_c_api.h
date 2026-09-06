#pragma once
#include "spacepdhcg/cuda/gtoc12_qoco_c_api.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct spacepdhcg_gtoc12_scvx_settings {
    int substeps, polish_substeps, polish_iterations, max_iterations;
    double virtual_weight, smoothness_weight, initial_trust_state, initial_trust_control;
    double minimum_trust, maximum_trust, ratio_reject, ratio_shrink, ratio_grow;
    double shrink_factor, grow_factor, defect_tolerance, step_tolerance, objective_tolerance;
    double conic_tolerance, time_limit_s, minimum_mass, radius_floor, vinf_max;
} spacepdhcg_gtoc12_scvx_settings;

typedef struct spacepdhcg_gtoc12_scvx_record {
    int iteration, accepted, conic_rejected, qoco_status;
    double merit, final_mass_fraction, max_defect, virtual_inf, ratio, step;
    double trust_state, trust_control;
} spacepdhcg_gtoc12_scvx_record;

typedef struct spacepdhcg_gtoc12_scvx_result {
    /* status: 0 iteration_limit, 1 converged, 2 failed, 3 infeasible, 4 timeout.
     * diagnostic: 0 none, 1 conic failure, 2 trust collapse, 3 polish budget,
     * 4 iteration budget, 5 feasible trust exhaustion, 6 virtual control.
     */
    int status, diagnostic, iterations, accepted_iterations;
    double max_defect, virtual_inf;
    double departure_vinf[3], arrival_vinf[3]; /* scaled velocity units */
    uint64_t trajectory_upload_bytes, trajectory_download_bytes, control_download_bytes;
} spacepdhcg_gtoc12_scvx_result;

/* One native dispatch call, retained device trajectories throughout SCvx.
 * GPU kernels compute nonlinear merits, candidate/virtual norms, acceptance,
 * trust updates, polishing transitions and final status. CPU dispatches existing
 * synchronous QOCO (including its scalar audit) and enforces wall time between
 * subproblems; this is NOT yet a graph-capturable end-to-end GPU solver.
 * Pass both seed pointers as null to generate Lambert/Kepler seed on the GPU;
 * otherwise supply both host arrays once. Fixed topology setup remains host.
 * Outputs are
 * copied only at completion; records/reports have capacity max+polish iterations.
 * No CPU numerical fallback. Return 0 completed, 1 invalid input, 2 runtime,
 * 3 invalid dynamics, 5 missing QOCO extension. On nonzero return, outputs are
 * unspecified and MUST NOT be treated as a solution. Independent nonlinear
 * physics certification remains required even for status=converged.
 */
int spacepdhcg_gtoc12_scvx_solve_host(
    int intervals, int hold, int free_departure, int free_arrival,
    double kappa, double mass_flow, const double* times, const double* boundary,
    const double* fuel_weights, const double* seed_states, const double* seed_controls,
    int ruiz_iterations, const spacepdhcg_gtoc12_scvx_settings* settings,
    double* states, double* controls, spacepdhcg_gtoc12_scvx_record* records,
    spacepdhcg_gtoc12_qoco_report* reports, spacepdhcg_gtoc12_scvx_result* result);

/* Standalone GPU seed bridge for seed inspection and independent parity checks.
 * Times and boundary are in the same scaled GTOC12 units as the solver.
 * This bridge downloads the seed; the null-seed solve path above does not.
 */
int spacepdhcg_gtoc12_seed_evaluate_host(int nodes, const double* times,
    const double* boundary, int free_departure, int free_arrival, double vinf_max,
    double* states, double* controls);

#ifdef __cplusplus
}
#endif
