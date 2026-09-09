#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Batched RouteSearch._finish costing, one CUDA thread per candidate. This
 * prices supplied proxy flights; it does not solve or certify trajectories.
 * ABI v1 uses normal C alignment, FP64 arithmetic and --fmad=false. */
enum spacepdhcg_gtoc12_completion_role {
    SPACEPDHCG_COMPLETION_CAMP = 0,
    SPACEPDHCG_COMPLETION_EARTH_OUT = 1,
    SPACEPDHCG_COMPLETION_DEPLOY_HOP = 2,
    SPACEPDHCG_COMPLETION_COLLECT_HOP = 3,
    SPACEPDHCG_COMPLETION_EARTH_RETURN = 4
};
enum spacepdhcg_gtoc12_completion_model {
    SPACEPDHCG_COMPLETION_FLAT = 0,
    SPACEPDHCG_COMPLETION_RATIO = 1,
    SPACEPDHCG_COMPLETION_FIT5 = 2,
    SPACEPDHCG_COMPLETION_RETURN = 3,
    SPACEPDHCG_COMPLETION_CERTIFIED_FLAT = 4,
    SPACEPDHCG_COMPLETION_TABLE_RETURN = 5
};
/* TABLE_RETURN matches CollectPairTable.return_inflation's finite-DV feature;
 * RETURN matches RouteSearch.return_inflation_for. They agree for finite DV.
 * Certified-cell lookup and pair geometry are resolved by the caller using the
 * existing table; certified-flat has the same arithmetic as flat, and never
 * bypasses the authority gate. Custom callbacks/models are not supported. */
enum spacepdhcg_gtoc12_completion_failure {
    SPACEPDHCG_COMPLETION_OK = 0,
    SPACEPDHCG_COMPLETION_UNCOLLECTED = 1,
    SPACEPDHCG_COMPLETION_STAY_TOO_SHORT = 2,
    SPACEPDHCG_COMPLETION_LEG_AUTHORITY = 3,
    SPACEPDHCG_COMPLETION_INVALID_INFLATION = 4,
    SPACEPDHCG_COMPLETION_MASS_BELOW_DRY_PLUS_COLLECTED = 5,
    SPACEPDHCG_COMPLETION_INVALID_MINING_STAY = 6
};
/* INVALID_MINING_STAY represents maximum_collected_mass raising ValueError;
 * Python adapters must raise, rather than treating it as an ordinary None. */
enum spacepdhcg_gtoc12_completion_leg_stage {
    SPACEPDHCG_COMPLETION_UNVISITED = 0,
    SPACEPDHCG_COMPLETION_CAMP_PASSTHROUGH = 1,
    SPACEPDHCG_COMPLETION_AUTHORITY_REJECTED = 2,
    SPACEPDHCG_COMPLETION_INFLATION_REJECTED = 3,
    SPACEPDHCG_COMPLETION_COSTED = 4,
    SPACEPDHCG_COMPLETION_MINING_REJECTED = 5
};

typedef struct spacepdhcg_gtoc12_completion_policy {
    int32_t abi_version, sum_mode, reserved0, reserved1;
    double initial_mass, dry_mass, miner_mass, thrust, exhaust;
    double mining_rate, year_days, minimum_stay;
} spacepdhcg_gtoc12_completion_policy;
/* abi_version=1, sum_mode=1 (CPython 3.12 compensated float sum), reserved=0.
 * Physical constants must be finite, initial/dry/miner/mining/minimum_stay >=0,
 * thrust/exhaust/year_days >0. Native fixed constants DAY_S=86400 and the source
 * thresholds stay < minimum_stay-1e-6; abs(collect-departure) < 1e-6 are exact.
 * minimum_stay is in days, mining_rate in kg/year, exhaust in km/s. */

typedef struct spacepdhcg_gtoc12_completion_candidate {
    int32_t deploy_begin, deploy_count, leg_begin, leg_count;
    double partial_mass;
} spacepdhcg_gtoc12_completion_candidate;
/* Input segments form contiguous, nonoverlapping partitions in candidate order.
 * Prefix legs are already priced. Initial fuel is computed on CUDA as
 * (policy.initial_mass-partial_mass)-policy.miner_mass*deploy_count. */

typedef struct spacepdhcg_gtoc12_completion_deploy {
    double deploy_epoch, collect_epoch;
    int32_t has_collect, reserved;
} spacepdhcg_gtoc12_completion_deploy;
/* Slots retain deploy-dictionary insertion order. has_collect is 0/1; a missing
 * collect is a candidate failure. Nonfinite deploy/collect epochs are preserved
 * for source gate/ValueError semantics, not silently replaced or thresholded. */

typedef struct spacepdhcg_gtoc12_completion_leg {
    int32_t role, model, source_deploy, reserved;
    double departure, arrival, dv, flat, floor, slope;
    double fit[5], delta_a_au, delta_longitude_rad, authority_ratio;
} spacepdhcg_gtoc12_completion_leg;
/* source_deploy is candidate-local and required for collect_hop/earth_return;
 * other roles use -1. Departures/arrivals may be nonfinite (source comparisons
 * decide the failure); no timing/pruning gate is added. Models are allowed:
 * camp/earth_out/deploy_hop: flat; collect_hop: flat/ratio/fit5;
 * earth_return: flat/return/table_return/certified-flat.
 * floor/slope are ratio parameters or fit floor; fit=[intercept,r,tof/year,
 * abs(delta_a)/0.1,abs(delta_longitude)/pi]. flat is input inflation for camps.
 * Model numbers/roles/reserved/index errors reject the whole API call; numeric
 * model values including NaN/inf/negative are evaluated after authority, so
 * invalid inflation remains a candidate failure with the source precedence. */

typedef struct spacepdhcg_gtoc12_completion_result {
    int32_t failure, failed_leg, failed_deploy, processed_legs;
    double propellant, final_mass, collected, margin;
} spacepdhcg_gtoc12_completion_result;
/* failed_* are candidate-local, -1 when absent. processed_legs counts appended
 * forward legs including camps, excluding the failing leg. Totals expose the
 * state at the first failure; margin is final_mass-(dry_mass+collected), exactly
 * the source gate's grouping. There is no payload shrinking or extra dry gate. */

typedef struct spacepdhcg_gtoc12_completion_leg_result {
    int32_t stage, pickup;
    double mass_before, gained, departure_mass, mass_after;
    double authority, inflation, propellant, tof;
} spacepdhcg_gtoc12_completion_leg_result;
/* Unvisited records are zero. pickup=1 means the epoch predicate matched, even
 * if mining then raised or gained zero. Mass_before is before pickup. At an
 * authority/inflation rejection, mass_after is the unburnt departure mass.
 * Camps preserve input inflation but perform no mining/model/authority checks.
 * Repeated matching pickups add cargo again to the running mass, while the
 * collected dictionary keeps one value at its FIRST pickup insertion position.
 * Rebuild that dictionary by traversing pickup records, not deployment order.
 * Collected-by-deploy output is zero for absent pickups; use pickup to disambiguate.
 */

typedef struct spacepdhcg_gtoc12_completion_stats {
    double upload_ms, kernel_ms, download_ms;
    uint64_t candidates, deploy_slots, leg_slots;
} spacepdhcg_gtoc12_completion_stats;
/* Optional CUDA event timings describe only the three named stream phases,
 * excluding host validation/packing, allocation, event queries and destruction.
 * Slot counts are supplied batch sizes, not counts of successfully costed legs. */

/* Status 0: completed (inspect every result.failure); 1: malformed input/device;
 * 2: CUDA/allocation failure; 3: workspace busy; 4: unsupported ABI/model/flags.
 * All buffers and one nonblocking stream are retained to supplied capacities;
 * no allocation occurs during evaluation. Evaluation requires the workspace's
 * current CUDA device; create/destroy restore the caller's previous device.
 * Invalid/unsupported inputs leave caller outputs untouched. CUDA failures can
 * leave outputs incomplete and must not be consumed. No input/output aliasing.
 */
int spacepdhcg_gtoc12_completion_create(
    int32_t device, int32_t max_candidates, int32_t max_deploys,
    int32_t max_legs, void** workspace);
int spacepdhcg_gtoc12_completion_evaluate_host(
    void* workspace, int32_t candidate_count, int32_t deploy_count, int32_t leg_count,
    const spacepdhcg_gtoc12_completion_policy* policy,
    const spacepdhcg_gtoc12_completion_candidate* candidates,
    const spacepdhcg_gtoc12_completion_deploy* deploys,
    const spacepdhcg_gtoc12_completion_leg* legs,
    spacepdhcg_gtoc12_completion_result* results,
    spacepdhcg_gtoc12_completion_leg_result* leg_results,
    double* collected_by_deploy, spacepdhcg_gtoc12_completion_stats* stats);
/* Detail outputs and stats are optional. For an empty batch all three counts
 * must be zero, data/output pointers may be null and stats (if supplied) is zero;
 * ownership and capacity checks still apply. No input sorting or result ranking. */
int spacepdhcg_gtoc12_completion_destroy(void** workspace);

/* Additive compact-request API. A model is an immutable, device-owned snapshot;
 * re-create it when catalogue, model parameters or certified return grids change.
 * Inputs are copied during create, and may then be released by the caller.
 * Geometry is evaluated on CUDA at each flight departure (the scalar, one-epoch
 * CollectPairTable.pair_geometry convention). No Lambert or trajectory solve is
 * performed here. The ordinary completion gates and output ABI are unchanged.
 */
typedef struct spacepdhcg_gtoc12_completion_model_policy {
    int32_t abi_version, hop_model, table_hop_model, return_model;
    int32_t table_return_model, reserved0, reserved1, reserved2;
    double hop_flat, hop_floor, hop_slope, return_flat, table_return_flat, fit_floor;
    double fit[5], authority_ratio[5], epoch0, step_days;
} spacepdhcg_gtoc12_completion_model_policy;
/* ABI=1; hop_model=0/1, table_hop_model=0/1/2, return_model=0/3,
 * table_return_model=0/5; reserved=0. Model values preserve the old numeric
 * failure semantics. epoch0 finite and step_days finite/positive. */
typedef struct spacepdhcg_gtoc12_completion_orbit {
    double semi_major_axis_km, epoch_mjd, mean_anomaly_rad;
    double ascending_node_rad, argument_of_perihelion_rad;
} spacepdhcg_gtoc12_completion_orbit;
/* Body i+1 is orbits[i], using official catalogue identity order. All entries
 * finite, a>0; no Earth entry. Constants are the repository's official GTOC12
 * AU, solar mu and day. Host validation rejects unsupported geometry inputs. */
typedef struct spacepdhcg_gtoc12_completion_return_grid {
    int32_t body_id, rows, cell_begin, reserved;
} spacepdhcg_gtoc12_completion_return_grid;
/* Grids have unique valid body IDs and partition cell arrays in supplied order.
 * Each row has return_tof_count cells. CUDA uses ties-to-even epoch rounding and
 * the FIRST nearest TOF, matching round/argmin. Only ok=1 overrides the generic
 * model. A cell's inflation still passes the ordinary post-authority gate. */
typedef struct spacepdhcg_gtoc12_completion_compact_candidate {
    int32_t deploy_begin, deploy_count, leg_begin, leg_count;
    double partial_mass;
    int32_t use_table, reserved;
} spacepdhcg_gtoc12_completion_compact_candidate;
typedef struct spacepdhcg_gtoc12_completion_compact_deploy {
    double deploy_epoch, collect_epoch;
    int32_t has_collect, body_id, reserved0, reserved1;
} spacepdhcg_gtoc12_completion_compact_deploy;
typedef struct spacepdhcg_gtoc12_completion_compact_leg {
    int32_t role, from_id, to_id, candidate;
    double departure, arrival, dv, input_inflation;
} spacepdhcg_gtoc12_completion_compact_leg;
/* Partitions/order match the ordinary ABI. Compact epochs must be finite;
 * source/body IDs must be valid, each collection source must be deployed, and
 * every leg's candidate index must match its partition. Structural errors reject
 * the whole call without touching output buffers. No new physical timing gate.
 */
int spacepdhcg_gtoc12_completion_model_create(
    int32_t device, const spacepdhcg_gtoc12_completion_model_policy* policy,
    int32_t orbit_count, const spacepdhcg_gtoc12_completion_orbit* orbits,
    int32_t return_tof_count, const double* return_tofs,
    int32_t grid_count, const spacepdhcg_gtoc12_completion_return_grid* grids,
    int32_t cell_count, const double* cell_inflation, const uint8_t* cell_ok,
    void** model);
/* Create independent immutable pricing/grid state using a source model's
 * resident catalogue. No catalogue validation, allocation, or upload is repeated.
 * Other inputs have the same validation/copy semantics as model_create.
 * The source must remain alive until this call returns; afterwards either model
 * may be destroyed first. The catalogue is released with its last model.
 * The source is unchanged on failure and *model is null. Device is inherited;
 * the caller's current device is restored, as for model_create.
 */
int spacepdhcg_gtoc12_completion_model_with_catalogue(
    const void* source_model, const spacepdhcg_gtoc12_completion_model_policy* policy,
    int32_t return_tof_count, const double* return_tofs,
    int32_t grid_count, const spacepdhcg_gtoc12_completion_return_grid* grids,
    int32_t cell_count, const double* cell_inflation, const uint8_t* cell_ok,
    void** model);
int spacepdhcg_gtoc12_completion_model_destroy(void** model);
int spacepdhcg_gtoc12_completion_evaluate_compact_host(
    void* workspace, const void* model,
    int32_t candidate_count, int32_t deploy_count, int32_t leg_count,
    const spacepdhcg_gtoc12_completion_policy* policy,
    const spacepdhcg_gtoc12_completion_compact_candidate* candidates,
    const spacepdhcg_gtoc12_completion_compact_deploy* deploys,
    const spacepdhcg_gtoc12_completion_compact_leg* legs,
    spacepdhcg_gtoc12_completion_result* results,
    spacepdhcg_gtoc12_completion_leg_result* leg_results,
    double* collected_by_deploy, spacepdhcg_gtoc12_completion_stats* stats,
    spacepdhcg_gtoc12_completion_leg* expanded_legs);
/* expanded_legs is optional diagnostic readout, outside normal search use.
 * Retained workspace buffers cover both input APIs. No allocation during eval;
 * kernel_ms includes metadata assembly and the original forward-cost kernel.
 * Model must remain alive until eval returns. Models and workspaces must use the
 * current device. Outputs from a CUDA error are incomplete and must not be used.
 */

#ifdef __cplusplus
}
#endif
