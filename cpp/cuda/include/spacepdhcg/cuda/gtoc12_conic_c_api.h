#pragma once

#ifdef __cplusplus
extern "C" {
#endif

typedef struct spacepdhcg_gtoc12_conic spacepdhcg_gtoc12_conic;
typedef struct spacepdhcg_gtoc12_conic_parameters {
    double trust_state, trust_control, virtual_weight, minimum_mass;
    double radius_floor, vinf_max, smoothness_weight;
} spacepdhcg_gtoc12_conic_parameters;
typedef struct spacepdhcg_gtoc12_conic_dimensions {
    int variables, rows, equalities, inequalities, soc_count, a_nonzeros, p_nonzeros;
} spacepdhcg_gtoc12_conic_dimensions;
typedef struct spacepdhcg_gtoc12_conic_device_outputs {
    const int *a_offsets, *a_indices, *p_offsets, *p_indices;
    const double *a, *b, *q, *p;
    const int* invalid;
} spacepdhcg_gtoc12_conic_device_outputs;

/* GTOC12 conic layout: minimize .5*x'P*x + q'x, A*x+s=b, cones ordered
 * zero, nonnegative, then SOC(4). P is upper triangular CSC. All potential
 * dynamics entries are retained, including numeric zeros; the identically
 * zero mass-row derivatives are omitted. The structure is
 * compiled once on the host; subsequent arithmetic is native FP64 CUDA.
 * Boundary[12] is r0,v0,rf,vf; fuel_weights[nodes] are already scaled by lam.
 * Status 0 success, 1 bad arguments, 2 CUDA/allocation failure, 3 invalid data.
 * Device-bound, serialized workspace; no silent CPU numerical fallback.
 */
int spacepdhcg_gtoc12_conic_create(int intervals, int hold, int free_departure,
    int free_arrival, double kappa, double mass_flow, const double* times,
    const double* boundary, const double* fuel_weights, spacepdhcg_gtoc12_conic**);
void spacepdhcg_gtoc12_conic_destroy(spacepdhcg_gtoc12_conic*);
int spacepdhcg_gtoc12_conic_get_dimensions(spacepdhcg_gtoc12_conic*,
    spacepdhcg_gtoc12_conic_dimensions*);
/* Copies retained HOST topology into caller arrays of nnz and variables+1.
 * Does not download GPU topology. Useful for the transitional CPU solver.
 */
int spacepdhcg_gtoc12_conic_copy_topology_host(spacepdhcg_gtoc12_conic*,
    int* a_offsets, int* a_indices, int* p_offsets, int* p_indices);
int spacepdhcg_gtoc12_conic_outputs(spacepdhcg_gtoc12_conic*,
    spacepdhcg_gtoc12_conic_device_outputs*);
/* All inputs are device pointers. Runs interval linearisation then assembly
 * on the supplied stream. No allocations, host numerical work or host sync.
 * Parameters may be updated on that stream before a graph replay. Outputs and
 * topology persist until reuse/destruction. Check device invalid before use.
 * Inputs must not alias outputs; caller completes all external-stream work
 * before reuse or destruction. No mid-kernel cancellation is provided.
 */
int spacepdhcg_gtoc12_conic_launch_device(spacepdhcg_gtoc12_conic*,
    const double* states, const double* controls,
    const spacepdhcg_gtoc12_conic_parameters* parameters, int substeps, void* stream);
/* Capture-compatible device scheduling. Scalar inputs obey the same ordering,
 * lifetime, device and non-aliasing contract as numerical inputs. A null enable
 * means always run; zero preserves all outputs, including invalid. Enabled
 * substeps<1 or invalid dynamics marks invalid without consuming stale dynamics
 * coefficients. Return0 means enqueued, not valid: always check device invalid.
 */
int spacepdhcg_gtoc12_conic_launch_controlled_device(spacepdhcg_gtoc12_conic*,
    const double* states, const double* controls,
    const spacepdhcg_gtoc12_conic_parameters* parameters, const int* device_substeps,
    const int* device_enabled, void* stream);
/* Transitional bridge: upload state/control/parameters, linearise and assemble
 * on GPU, download packed [A values, b, q, P values] and sync once. Caller owns
 * packed output with a_nonzeros+rows+variables+p_nonzeros doubles.
 */
int spacepdhcg_gtoc12_conic_evaluate_host(spacepdhcg_gtoc12_conic*,
    const double* states, const double* controls,
    const spacepdhcg_gtoc12_conic_parameters*, int substeps, double* packed_output);

#ifdef __cplusplus
}
#endif
