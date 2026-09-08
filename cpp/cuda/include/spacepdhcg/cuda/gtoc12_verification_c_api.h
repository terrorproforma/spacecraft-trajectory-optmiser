#pragma once
#include <cuda_runtime_api.h>
#include <stdint.h>

// Physical units: seconds from departure, km, km/s, kg, N. Sample epochs
// must be strictly increasing. Arcs must be ordered, disjoint, inside the leg.
// Zero samples are represented by zero arcs (a coast), never an empty burn.
typedef struct {
    double initial[7];
    double duration_s;
    int32_t arc_offset, arc_count;
} spacepdhcg_verify_leg;
typedef struct { int32_t sample_offset, sample_count; } spacepdhcg_verify_arc;
typedef struct { double seconds, thrust[3]; } spacepdhcg_verify_sample;
typedef struct {
    double final_state[7], minimum_radius_km;
    int32_t status, accepted_steps, rejected_steps, evaluations;
} spacepdhcg_verify_result;
// Per-leg status: 0 success, 1 invalid input, 2 nonfinite/nonphysical state,
// 3 step budget exhausted, 4 step underflow. Failed outputs contain NaNs.
// Fixed independent accuracy: rtol 1e-12, atol [1e-7 km,1e-10 km/s,1e-9 kg].
// Minimum radius is sampled at daily burn / five-day coast points + endpoints,
// matching the CPU certificate's sample rule, not a continuous-radius proof.
#ifdef __cplusplus
extern "C" {
#endif
// All buffers are device resident, on the caller's device and stream. No host
// iteration, allocation, download or synchronization. Buffers must outlive work.
cudaError_t spacepdhcg_gtoc12_verify_launch(
    const spacepdhcg_verify_leg* legs, int32_t leg_count,
    const spacepdhcg_verify_arc* arcs, int32_t arc_count,
    const spacepdhcg_verify_sample* samples, int32_t sample_count,
    int32_t max_steps, spacepdhcg_verify_result* results, cudaStream_t stream);
// Exact same-grid ZOH initialization. Inputs are device arrays: one physical
// initial state [km, km/s, kg], increasing nondimensional node times, and N*3
// thrust samples in newtons (the inactive last sample must be zero). Outputs
// are N*7 states [r/AU, v/(AU/TU), m/m0] and N*4 controls [T, |T|]/0.6.
// DOP853 advances the original piecewise-constant control without clipping or
// endpoint snapping. This prepares an iterate, not an optimality certificate.
// The step budget covers the entire leg; every output is NaN on failure.
cudaError_t spacepdhcg_gtoc12_verify_zoh_seed_launch(
    int32_t nodes, const double* times, const double* initial,
    const double* thrust, int32_t max_steps, double* states, double* controls,
    spacepdhcg_verify_result* result, cudaStream_t stream);
// Retained host bridge for existing application callers. Fixed capacities,
// serialized calls from the creating thread/device; one upload and final read.
typedef struct spacepdhcg_verify_workspace spacepdhcg_verify_workspace;
cudaError_t spacepdhcg_gtoc12_verify_create(int32_t legs, int32_t arcs, int32_t samples,
    spacepdhcg_verify_workspace** workspace);
cudaError_t spacepdhcg_gtoc12_verify_host(spacepdhcg_verify_workspace* workspace,
    const spacepdhcg_verify_leg* legs, int32_t leg_count,
    const spacepdhcg_verify_arc* arcs, int32_t arc_count,
    const spacepdhcg_verify_sample* samples, int32_t sample_count,
    int32_t max_steps, spacepdhcg_verify_result* results);
cudaError_t spacepdhcg_gtoc12_verify_destroy(spacepdhcg_verify_workspace** workspace);
#ifdef __cplusplus
}
#endif
