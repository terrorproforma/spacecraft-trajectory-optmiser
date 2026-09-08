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

#ifdef __cplusplus
}
#endif
