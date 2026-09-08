#pragma once
#include <stdint.h>
#include "spacepdhcg/cuda/orbitweaver_gpu_c_api.h"
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
typedef struct spacepdhcg_collect_result_v2 {
    int32_t feasible,states,hops,terminal_j,terminal_t,terminal_r,reposition,reserved;
    double objective,penalty;
    int32_t collected_at[16],source[16],target[16],departure[16],tof[16];
    double hop_propellant[16],hop_dv[16],return_dv;
} spacepdhcg_collect_result_v2;
/* Status: 0 success, 1 invalid input, 2 CUDA/allocation error, 3 busy. */
int spacepdhcg_collect_create(const spacepdhcg_collect_policy*,
    const spacepdhcg_collect_inputs*,void** workspace);
int spacepdhcg_collect_solve(void* workspace,const double* mass_by_subset,
    double camp_mass,double price,spacepdhcg_collect_result* result);
int spacepdhcg_collect_solve_v2(void* workspace,const double* mass_by_subset,
    double camp_mass,double price,spacepdhcg_collect_result_v2* result);
int spacepdhcg_collect_destroy(void* workspace);
/* Immutable float32 device tables. Optional read is for host consumers only. */
int spacepdhcg_collect_table_create(spacepdhcg_orbitweaver_lambert_workspace*,
    const spacepdhcg_orbitweaver_hop_elements*,const double* epochs,int32_t n,
    const double* tofs,int32_t nt,double end,void** table);
int spacepdhcg_collect_table_read(void* table,float* output);
/* Price a resident hop table over rows [lo,hi) and TOFs <= maximum_tof + 1e-9.
 * Returns only the minimum propellant. For fitted inflation, geometry_l is a
 * host array of hi-lo wrapped phase differences; otherwise it is ignored. */
int spacepdhcg_collect_table_min_propellant(void* table,
    const spacepdhcg_collect_policy*,int32_t lo,int32_t hi,double maximum_tof,
    double mass,double geometry_a,const double* geometry_l,double* result);
int spacepdhcg_collect_table_destroy(void* table);
/* Rank all candidates after initial_count: first representatives of unseen
 * prefixes, then remaining candidates in original rank order. No rejection.
 * Each host prefix has 13 doubles: length, then up to three (from,to,t0,tf)
 * tuples, zero padded. Output has count-initial_count int32 ranks. */
int spacepdhcg_route_recovery_order(int32_t device,const double* prefixes,
    int32_t count,int32_t initial_count,int32_t* output);
/* pairs[k*k] may be null for banned/self pairs; returns[k] are required.
 * Copies device table slices starting at t0 to owned DP storage before return.
 * inputs.dv/returns are ignored; other policy inputs remain host arrays. */
int spacepdhcg_collect_create_tables(const spacepdhcg_collect_policy*,
    const spacepdhcg_collect_inputs*,void* const* pairs,void* const* returns,
    int32_t t0,void** workspace);
#ifdef __cplusplus
}
#endif
