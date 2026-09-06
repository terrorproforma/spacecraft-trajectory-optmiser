#pragma once

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* FP64 GTOC12 interval RK4 in AU/TU/initial-mass/thrust-limit units.
 * Each interval starts at its own supplied state. This is NOT a sequential
 * trajectory replay or a complete SCvx solver. Nonlinear mass flow uses |u_xyz|;
 * the variational mass-control row uses the existing Gamma convex surrogate.
 * hold=0: ZOH; hold=1: clamped four-node cubic Lagrange interpolation.
 * Status: 0 success, 1 invalid arguments, 2 CUDA failure, 3 invalid dynamics.
 * A workspace is bound to its creating CUDA device and is not thread-safe.
 */
typedef struct spacepdhcg_gtoc12_discretisation spacepdhcg_gtoc12_discretisation;

int spacepdhcg_gtoc12_discretisation_create(
    int intervals, int hold, double kappa, double mass_flow,
    const double* host_node_times, spacepdhcg_gtoc12_discretisation** output);
void spacepdhcg_gtoc12_discretisation_destroy(spacepdhcg_gtoc12_discretisation* workspace);

/* Retained output buffers: A[N,7,7], B[N,S,7,4], c[N,7], propagated[N,7],
 * where S=1 (ZOH) or 4 (Lagrange). They remain device-resident until overwritten
 * by the next call or workspace destruction. No allocation or host sync in
 * launch_device. The caller must order consumers on the supplied CUDA stream
 * and finish all external-stream work before reuse/destruction. Inputs must
 * belong to the same device and not alias the output buffers. A/B/c are valid
 * only after linearise=1. device_invalid is reset and filled on that stream.
 */
int spacepdhcg_gtoc12_discretisation_launch_device(
    spacepdhcg_gtoc12_discretisation* workspace, const double* device_states,
    const double* device_controls, int substeps, int linearise, void* cuda_stream);
/* Capture-compatible scheduling: substeps is read on the GPU at execution time.
 * enabled may be null (always run); zero preserves every output, including the
 * invalid flag. Any nonzero enables execution. Enabled substeps < 1 sets invalid
 * without changing numerical outputs. The return code only describes enqueue
 * success; consumers must check device_invalid before using numerical outputs.
 * Both scalar inputs follow the same device, lifetime, ordering and non-aliasing
 * contract as states/controls. No allocation, host download or synchronization.
 */
int spacepdhcg_gtoc12_discretisation_launch_controlled_device(
    spacepdhcg_gtoc12_discretisation* workspace, const double* device_states,
    const double* device_controls, const int* device_substeps,
    const int* device_enabled, int linearise, void* cuda_stream);
int spacepdhcg_gtoc12_discretisation_outputs(
    spacepdhcg_gtoc12_discretisation* workspace, const double** a, const double** b,
    const double** c, const double** propagated, const int** device_invalid);

/* Transitional host bridge for the existing Python SCvx assembly. Uploads,
 * launches and downloads on a retained nonblocking stream; synchronizes once.
 * All input/output arrays are contiguous row-major FP64. A/B/c may be null
 * when linearise=0. No host dynamics or silent CPU fallback is performed.
 */
int spacepdhcg_gtoc12_discretisation_evaluate_host(
    spacepdhcg_gtoc12_discretisation* workspace, const double* states,
    const double* controls, int substeps, int linearise, double* a, double* b,
    double* c, double* propagated);

#ifdef __cplusplus
}
#endif
