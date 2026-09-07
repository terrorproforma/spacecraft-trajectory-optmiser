#pragma once
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
/* One collection Held-Karp pass. Arrays are contiguous host inputs, copied
 * before create returns. Costs retain the reference float32 table values.
 * Pair-major arrays use [source][target][epoch][tof]; geometry/penalty omit tof.
 * Return arrays use [asteroid][epoch][return_tof]. override is NaN for the
 * ordinary return model, +inf for a refused sweep cell, or measured inflation.
 * State traversal, pricing, camping maxima and backtracking execute on CUDA. */
typedef struct spacepdhcg_collect_policy {
    int32_t k,n,nt,nr,camp,hop_model,return_model,reserved;
    double thrust,exhaust,hop_ratio,hop_flat,hop_floor,hop_slope;
    double return_ratio,return_flat,fit_floor,fit_coefficients[5];
} spacepdhcg_collect_policy;
typedef struct spacepdhcg_collect_inputs {
    const double *dv,*returns,*tofs,*return_tofs,*mined,*geometry_a,*geometry_l;
    const double *penalty,*override_inflation;
    const int32_t *steps,*banned;
} spacepdhcg_collect_inputs;
typedef struct spacepdhcg_collect_result {
    int32_t feasible,states,hops,terminal_j,terminal_t,terminal_r,reposition,reserved;
    double objective,penalty;
    int32_t collected_at[16],source[16],target[16],departure[16],tof[16];
    double hop_propellant[16];
} spacepdhcg_collect_result;
/* Status: 0 success, 1 invalid input, 2 CUDA/allocation error, 3 busy. */
int spacepdhcg_collect_create(const spacepdhcg_collect_policy*,
    const spacepdhcg_collect_inputs*,void** workspace);
int spacepdhcg_collect_solve(void* workspace,const double* mass_by_subset,
    double camp_mass,double price,spacepdhcg_collect_result* result);
int spacepdhcg_collect_destroy(void* workspace);
#ifdef __cplusplus
}
#endif
