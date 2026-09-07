#pragma once
#include <stdint.h>
#include "spacepdhcg/cuda/orbitweaver_gpu_c_api.h"
#ifdef __cplusplus
extern "C" {
#endif

typedef struct spacepdhcg_gtoc12_retime_stage {
    int32_t cell_offset, tof_offset, tofs, camp_min, camp_max, pinned_next, model;
    int32_t reserved;
    double arrival_rate, departure_rate, earliest_collect, mass, ratio_limit;
    double flat, floor, slope, calibration;
} spacepdhcg_gtoc12_retime_stage;

typedef struct spacepdhcg_gtoc12_sweep_cell {
    int32_t departure, tof;
    double delta_v;
    int32_t certified, reserved;
} spacepdhcg_gtoc12_sweep_cell;

typedef struct spacepdhcg_gtoc12_forward_visit {
    int32_t deploy, collect, donor, reserved; /* donor: earlier/self visit, -1 foreign, -2 missing */
    double foreign_epoch;
} spacepdhcg_gtoc12_forward_visit;
typedef struct spacepdhcg_gtoc12_forward_policy {
    double initial_mass, minimum_stay, mining_rate, year_days, miner_mass, dry_mass, step;
} spacepdhcg_gtoc12_forward_policy;
typedef struct spacepdhcg_gtoc12_forward_result {
    int32_t failure, mass_count;
    double propellant, final_mass;
} spacepdhcg_gtoc12_forward_result;
typedef struct spacepdhcg_gtoc12_driver_policy {
    double price_growth, orphan_credit, orphan_margin, mission_end;
    int32_t max_prices, max_masses;
} spacepdhcg_gtoc12_driver_policy;
typedef struct spacepdhcg_gtoc12_driver_weight {
    double weight;
    int32_t orphan, reserved;
} spacepdhcg_gtoc12_driver_weight;
typedef struct spacepdhcg_gtoc12_driver_result {
    double objective, price;
    int32_t feasible, failure, price_rounds, mass_rounds, evaluations, reserved;
} spacepdhcg_gtoc12_driver_result;
/* One conditional CUDA graph drives mass corrections and price search; only
 * its best path, decision summary and final profile are downloaded. */
int spacepdhcg_gtoc12_retime_order_host(void* workspace,
    const spacepdhcg_gtoc12_retime_stage* stages,double price,double thrust,double exhaust,
    int32_t* arrivals,int32_t* departures,double* objective,int32_t* feasible,
    double* delta_v,double* swept_inflation,uint8_t* swept_ok,
    const spacepdhcg_gtoc12_forward_visit* visits,const spacepdhcg_gtoc12_forward_policy* policy,
    spacepdhcg_gtoc12_forward_result* result,double* masses,double* inflations,double* collected,
    const spacepdhcg_gtoc12_driver_policy* driver,const spacepdhcg_gtoc12_driver_weight* weights,
    spacepdhcg_gtoc12_driver_result* summary,double* final_profile);
/* Schedule selection and forward bookkeeping share the retained graph/output.
 * Forward failure: 0 success, 1 no deployer, 2 short stay, 3 TOF outside grid,
 * 4 infeasible leg, 5 authority, 6 dry+payload, 7 invalid stay, 8 infeasible DP. */
int spacepdhcg_gtoc12_retime_forward_host(void* workspace,
    const spacepdhcg_gtoc12_retime_stage* stages,double price,double thrust,double exhaust,
    int32_t* arrivals,int32_t* departures,double* objective,int32_t* feasible,
    double* delta_v,double* swept_inflation,uint8_t* swept_ok,
    const spacepdhcg_gtoc12_forward_visit* visits,const spacepdhcg_gtoc12_forward_policy* policy,
    spacepdhcg_gtoc12_forward_result* result,double* masses,double* inflations,double* collected);

/* Immutable table snapshot; all arrays are host buffers copied on creation.
 * Cells concatenate departure-major (epochs x stage TOFs) tables. Swept
 * inflation is NaN for unmeasured cells; swept_ok defaults to 1 when absent.
 * Returns 0 on success, 1 invalid input, 2 CUDA error, 3 busy. */
int spacepdhcg_gtoc12_retime_create(
    int32_t device, int32_t epochs, int32_t stages, int32_t cells, int32_t tofs,
    const double* epoch_values, const double* dv, const uint8_t* feasible,
    const double* swept, const uint8_t* swept_ok, const double* tof_values,
    const int32_t* shifts, void** workspace);
int spacepdhcg_gtoc12_retime_host(void* workspace,
    const spacepdhcg_gtoc12_retime_stage* stages, double price,
    double thrust, double exhaust_velocity, int32_t* arrivals,
    int32_t* departures, double* objective, int32_t* feasible);
/* Build unswept immutable transfer tables directly on the device. Stage offsets
 * describe a contiguous partition of cells and TOFs. Host elements have one
 * entry per stage; the Lambert workspace owns bounded construction scratch. */
int spacepdhcg_gtoc12_retime_create_elements(
    int32_t device,int32_t epochs,int32_t stages,int32_t cells,int32_t tofs,
    const double* epoch_values,const double* tof_values,const int32_t* shifts,
    const spacepdhcg_gtoc12_retime_stage* params,
    const spacepdhcg_orbitweaver_hop_elements* elements,
    spacepdhcg_orbitweaver_lambert_workspace* lambert,void** workspace);
/* Same evaluation, also downloading one scalar delta-v per selected leg. */
int spacepdhcg_gtoc12_retime_path_host(void* workspace,
    const spacepdhcg_gtoc12_retime_stage* stages,double price,double thrust,double exhaust,
    int32_t* arrivals,int32_t* departures,double* objective,int32_t* feasible,double* delta_v);
/* Compact attempted sweep samples update an existing table region. Equal
 * distances prefer the first input sample. Zero samples remove the override. */
int spacepdhcg_gtoc12_retime_set_sweep(void* workspace,int32_t cell_offset,int32_t tofs,
    int32_t count,const spacepdhcg_gtoc12_sweep_cell* samples,int32_t reach);
int spacepdhcg_gtoc12_retime_swept_path_host(void* workspace,
    const spacepdhcg_gtoc12_retime_stage* stages,double price,double thrust,double exhaust,
    int32_t* arrivals,int32_t* departures,double* objective,int32_t* feasible,
    double* delta_v,double* swept_inflation,uint8_t* swept_ok);
/* Explicit diagnostic export; ordinary scheduling only downloads its path. */
int spacepdhcg_gtoc12_retime_read_sweep(void* workspace,int32_t cell_offset,int32_t tofs,
    double* inflation,uint8_t* feasible);
/* Graph replay is enabled by default. Disabling it uses the same kernels with
 * ordinary launches for matched benchmarks. Retained graphs survive toggles;
 * changed policies and sweep masks are read from current device buffers. */
int spacepdhcg_gtoc12_retime_set_graph(void* workspace,int32_t enabled);
/* Read-only counters; no device synchronization or transfer. */
int spacepdhcg_gtoc12_retime_graph_stats(void* workspace,uint64_t* builds,uint64_t* launches);
int spacepdhcg_gtoc12_retime_destroy(void** workspace);

#ifdef __cplusplus
}
#endif
