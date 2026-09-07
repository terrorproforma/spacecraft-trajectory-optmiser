#pragma once
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

typedef struct spacepdhcg_gtoc12_retime_stage {
    int32_t cell_offset, tof_offset, tofs, camp_min, camp_max, pinned_next, model;
    int32_t reserved;
    double arrival_rate, departure_rate, earliest_collect, mass, ratio_limit;
    double flat, floor, slope, calibration;
} spacepdhcg_gtoc12_retime_stage;

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
int spacepdhcg_gtoc12_retime_destroy(void** workspace);

#ifdef __cplusplus
}
#endif
