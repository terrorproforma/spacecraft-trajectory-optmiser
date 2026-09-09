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
/* Replace all numerical inputs/topology while retaining allocations. Each new
 * k/n/nt/nr must fit the creation dimensions; status 4 means capacity exceeded.
 * Invalid/busy/capacity failures leave the previous problem unchanged. A CUDA
 * failure disables solve until a successful update or destruction. Host inputs
 * are copied before return and need not remain alive. No allocation on update. */
int spacepdhcg_collect_update(void* workspace,const spacepdhcg_collect_policy*,
    const spacepdhcg_collect_inputs*);
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
/* Same retained-capacity/update contract, copying immutable device table slices.
 * Validated table handles are held against eviction for the duration of copy. */
int spacepdhcg_collect_update_tables(void* workspace,const spacepdhcg_collect_policy*,
    const spacepdhcg_collect_inputs*,void* const* pairs,void* const* returns,int32_t t0);
/* Native mining and two-pass burn scheduling. Only O(n+k) schedule data is
 * uploaded; mined epoch values and all 2^k subset masses are computed on CUDA.
 * Existing APIs/structs retain their ABI. inputs.mined is ignored by plan APIs.
 * Epochs are strictly increasing; weights may be any finite numbers.
 * All pointers are host arrays copied before create/update returns. */
typedef struct spacepdhcg_collect_plan_inputs {
    int32_t abi_version,reserved;
    const double *epochs,*deploy_epochs,*weights;
    double minimum_stay,mining_rate,year_days,floor_mass;
} spacepdhcg_collect_plan_inputs;
typedef struct spacepdhcg_collect_plan_result {
    spacepdhcg_collect_result_v2 tour;
    double burn_per_hop,first_objective;
    int32_t passes,has_first_objective;
} spacepdhcg_collect_plan_result;
int spacepdhcg_collect_create_plan(const spacepdhcg_collect_policy*,
    const spacepdhcg_collect_inputs*,const spacepdhcg_collect_plan_inputs*,void**);
int spacepdhcg_collect_update_plan(void*,const spacepdhcg_collect_policy*,
    const spacepdhcg_collect_inputs*,const spacepdhcg_collect_plan_inputs*);
int spacepdhcg_collect_create_plan_tables(const spacepdhcg_collect_policy*,
    const spacepdhcg_collect_inputs*,const spacepdhcg_collect_plan_inputs*,
    void* const*,void* const*,int32_t,void**);
int spacepdhcg_collect_update_plan_tables(void*,const spacepdhcg_collect_policy*,
    const spacepdhcg_collect_inputs*,const spacepdhcg_collect_plan_inputs*,
    void* const*,void* const*,int32_t);
/* NaN burn selects the reference automatic heavy/estimated-burn policy.
 * A finite burn requests one pass (negative values clamp to zero).
 * Pass selection, fallback and mean-hop calculation stay on CUDA; one final
 * result download. Invalid inputs leave the output and current problem intact.
 * Legacy updates disable plan solve until a successful plan update.
 * Updating a legacy-only allocation to a plan returns capacity status 4. */
int spacepdhcg_collect_solve_plan(void*,double camp_mass,double price,double burn,
    spacepdhcg_collect_plan_result*);
/* Optional native pair geometry and harvest-phase prior. elements[k][5] are
 * (epoch MJD, semi-major axis km, node rad, perihelion rad, mean anomaly rad).
 * Geometry and penalties are derived on CUDA on the plan's exact epoch slice.
 * inputs.geometry_a/geometry_l/penalty are ignored by these entry points.
 * Metadata is copied before return; old plan APIs retain their ABI. */
typedef struct spacepdhcg_collect_geometry_inputs {
    int32_t abi_version,reserved;
    const double* elements;
    double mu,day_seconds,au_km,phase_threshold,phase_slope,phase_weight;
} spacepdhcg_collect_geometry_inputs;
int spacepdhcg_collect_create_plan_geometry(const spacepdhcg_collect_policy*,
    const spacepdhcg_collect_inputs*,const spacepdhcg_collect_plan_inputs*,
    const spacepdhcg_collect_geometry_inputs*,void**);
int spacepdhcg_collect_update_plan_geometry(void*,const spacepdhcg_collect_policy*,
    const spacepdhcg_collect_inputs*,const spacepdhcg_collect_plan_inputs*,
    const spacepdhcg_collect_geometry_inputs*);
int spacepdhcg_collect_create_plan_tables_geometry(const spacepdhcg_collect_policy*,
    const spacepdhcg_collect_inputs*,const spacepdhcg_collect_plan_inputs*,
    const spacepdhcg_collect_geometry_inputs*,void* const*,void* const*,int32_t,void**);
int spacepdhcg_collect_update_plan_tables_geometry(void*,const spacepdhcg_collect_policy*,
    const spacepdhcg_collect_inputs*,const spacepdhcg_collect_plan_inputs*,
    const spacepdhcg_collect_geometry_inputs*,void* const*,void* const*,int32_t);
/* Optional diagnostic read, never needed by the planner. k/n must equal the
 * current problem; outputs hold k*k, k*k*n and k*k*n doubles respectively. */
int spacepdhcg_collect_read_geometry(void*,int32_t k,int32_t n,double* a,double* longitude,double* penalty);
#ifdef __cplusplus
}
#endif
