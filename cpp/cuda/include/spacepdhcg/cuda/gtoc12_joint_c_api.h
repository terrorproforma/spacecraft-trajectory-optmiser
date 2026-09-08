#pragma once

#include <stdint.h>
#include "spacepdhcg/cuda/orbitweaver_gpu_c_api.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Continuous fixed-order JointItinerary arithmetic, one thread per candidate.
 * This evaluates supplied Lambert/measured costs; it neither screens transfers
 * nor certifies a low-thrust trajectory. Compile the implementation --fmad=false.
 * All structures use ordinary C alignment and all arrays are row-major FP64.
 */
enum spacepdhcg_gtoc12_joint_failure {
    SPACEPDHCG_JOINT_OK = 0,
    SPACEPDHCG_JOINT_LAUNCH_BEFORE_WINDOW = 1,
    SPACEPDHCG_JOINT_RETURN_AFTER_WINDOW = 2,
    SPACEPDHCG_JOINT_EARTH_DWELL = 3,
    SPACEPDHCG_JOINT_NEGATIVE_DWELL = 4,
    SPACEPDHCG_JOINT_DWELL_TOO_LONG = 5,
    SPACEPDHCG_JOINT_PINNED_ARRIVAL = 6,
    SPACEPDHCG_JOINT_TOF_OUTSIDE_LIMITS = 7,
    SPACEPDHCG_JOINT_DOUBLE_DEPLOY = 8,
    SPACEPDHCG_JOINT_DOUBLE_COLLECT = 9,
    SPACEPDHCG_JOINT_COLLECT_WITHOUT_DEPLOY = 10,
    SPACEPDHCG_JOINT_STAY_TOO_SHORT = 11,
    SPACEPDHCG_JOINT_LEG_INFEASIBLE = 12,
    SPACEPDHCG_JOINT_EARTH_OUT_UNMEASURED_BELOW_FLOOR = 13,
    SPACEPDHCG_JOINT_LEG_AUTHORITY = 14,
    SPACEPDHCG_JOINT_MASS_BELOW_DRY_PLUS_COLLECTED = 15,
    SPACEPDHCG_JOINT_MASS_BELOW_DRY = 16,
    /* Maps to maximum_collected_mass's ValueError, not a feasible candidate. */
    SPACEPDHCG_JOINT_INVALID_STAY = 17
};

typedef struct spacepdhcg_gtoc12_joint_visit {
    int32_t deploy, collect, donor, structural_failure;
    double foreign_epoch, pinned_arrival, dwell_limit, weight;
} spacepdhcg_gtoc12_joint_visit;
/* donor >= 0: index of the body's own deployment visit; -1: foreign_epoch;
 * -2: missing donor. pinned_arrival=NaN means unpinned.
 * The wrapper scans visits in Python dictionary insertion order and packs the
 * first duplicate/missing-donor error at the relevant visit (failure 8/9/10).
 * All other structural_failure entries are zero. The kernel checks these only
 * after epoch/TOF gates, preserving evaluate()'s failure precedence. For unusual
 * collect-before-own-deploy input, donor is the FINAL deploy dictionary index,
 * while structural_failure still reflects the original ordered scan.
 */

typedef struct spacepdhcg_gtoc12_joint_stage {
    int32_t model, earth_out, reserved0, reserved1;
    double tof_min, tof_max, ratio_limit, flat, floor, slope, calibration;
} spacepdhcg_gtoc12_joint_stage;
/* model: 0 flat (_limits already includes the unmodelled calibration),
 * 1 ratio-dependent hop, 2 TOF/ratio-dependent Earth return. tof_min/max are
 * JointItinerary.tof_limits(role), including the protected Earth floor when
 * appropriate. ratio_limit already includes pair bans. No custom callbacks or
 * alternative inflation models are representable; the wrapper must reject them.
 */

typedef struct spacepdhcg_gtoc12_joint_cost {
    int32_t measured, reserved;
    double lambert, measured_delta_v, measured_mass;
} spacepdhcg_gtoc12_joint_cost;
/* measured refers to the wrapper's exact-epoch measured-leg lookup. A present
 * measurement is used only inside measured_mass_tolerance. Nonfinite Lambert
 * costs are allowed and reproduce the measured/unmeasured Python distinction.
 */

typedef struct spacepdhcg_gtoc12_joint_policy {
    double mission_start, latest_arrival, initial_mass, dry_mass, miner_mass;
    double minimum_stay, mining_rate, year_days, thrust, exhaust;
    double measured_mass_tolerance, margin_price;
    double earth_out_tof_floor, earth_out_inflation; /* NaN means absent */
    int32_t free_earth_leg, screen_earth_out, reserved0, reserved1;
} spacepdhcg_gtoc12_joint_policy;

typedef struct spacepdhcg_gtoc12_joint_result {
    int32_t failure, mass_count, measured_legs, rounds;
    double objective, weighted, collected, spare, propellant, final_mass;
} spacepdhcg_gtoc12_joint_result;
/* failure=0 is feasible. _fail-style failures have objective=-inf, remaining
 * values zero and mass_count=0; rounds records the number of forward passes.
 * Failure 15 preserves the final attempted pass's spare/propellant/masses and
 * measured count, as Python does after its four payload-sizing rounds.
 */

/* Status: 0 success (inspect EACH result.failure), 1 malformed input/device,
 * 2 CUDA/allocation error, 3 workspace busy, 4 unsupported policy/model/flags.
 * Creation retains evaluation buffers for max_candidates. The optional geometry
 * API adds retained buffers lazily and grows sparse-record storage as needed.
 * Workspaces bind to device and reject a different current
 * device at evaluation. One internal nonblocking stream, one final sync.
 */
int spacepdhcg_gtoc12_joint_create(
    int32_t device, int32_t max_candidates, int32_t visits, void** workspace);
int spacepdhcg_gtoc12_joint_evaluate_host(
    void* workspace, int32_t candidates,
    const spacepdhcg_gtoc12_joint_policy* policy,
    const spacepdhcg_gtoc12_joint_visit* visits,
    const spacepdhcg_gtoc12_joint_stage* stages,
    const double* arrivals, const double* departures,
    const spacepdhcg_gtoc12_joint_cost* costs,
    spacepdhcg_gtoc12_joint_result* results,
    double* masses, double* inflations, double* proxies, double* collected);
/* Inputs: visits[N], stages[N-1], arrivals/departures[B,N], costs[B,N-1].
 * Outputs: results[B], masses/inflations/proxies[B,N-1], collected[B,N].
 * Each detail output is optional (null). Noncollect visits have collected=0.
 * Detail rows are valid for feasible candidates and failure 15; other failures
 * have mass_count=0 and their detail values must be ignored. No output sorting.
 * For candidates=0, data/output pointers may be null; the workspace still must
 * belong to the current device. Invalid input leaves all caller outputs intact.
 */
int spacepdhcg_gtoc12_joint_destroy(void** workspace);

typedef struct spacepdhcg_gtoc12_joint_selection {
    int32_t index, invalid_stay;
    spacepdhcg_gtoc12_joint_result value;
} spacepdhcg_gtoc12_joint_selection;
/* Select the first highest-objective feasible row above minimum_objective+1e-9
 * on CUDA. index=-1 means no improvement. invalid_stay is set if ANY row has
 * failure 17, even when that row cannot win. value and compact detail outputs
 * are meaningful only when index>=0; details have N-1 entries (collected: N).
 * Zero candidates returns index=-1, invalid_stay=0 without dereferencing inputs.
 * NaN/+infinite minimum objectives select no row, as with NumPy comparisons.
 * All other input/status/ownership contracts match evaluate_host above.
 */
int spacepdhcg_gtoc12_joint_best_host(
    void* workspace, int32_t candidates,
    const spacepdhcg_gtoc12_joint_policy* policy,
    const spacepdhcg_gtoc12_joint_visit* visits,
    const spacepdhcg_gtoc12_joint_stage* stages,
    const double* arrivals, const double* departures,
    const spacepdhcg_gtoc12_joint_cost* costs, double minimum_objective,
    spacepdhcg_gtoc12_joint_selection* selection,
    double* masses, double* inflations, double* proxies, double* collected);

/* Sparse exact-epoch overrides for one fixed stage. cached=0 computes Lambert
 * on CUDA while retaining an optional measured leg; cached=1 reuses value.lambert.
 * Duplicate (leg, departure, arrival) records are forbidden. */
typedef struct spacepdhcg_gtoc12_joint_cached_cost {
    int32_t leg, cached;
    double departure, arrival;
    spacepdhcg_gtoc12_joint_cost value;
} spacepdhcg_gtoc12_joint_cached_cost;
typedef struct spacepdhcg_gtoc12_joint_geometry_stats {
    uint64_t computed_hops, cached_hops, rejected_hops;
} spacepdhcg_gtoc12_joint_geometry_stats;

/* Keeps preflight, sparse exact-epoch lookup, ephemerides and Lambert costs on
 * CUDA, then evaluates and optionally selects candidates. elements[N-1] and
 * records[record_count] are host inputs. selection=null returns every result;
 * nonnull selection uses the compact best_host output contract. stats is a
 * required host output. No CPU geometry fallback and no intermediate downloads.
 * All elements must be valid elliptic orbits, even for skipped/cached requests.
 * Record order is strictly increasing (leg, departure, arrival).
 */
int spacepdhcg_gtoc12_joint_geometry_host(
    void* workspace, int32_t candidates,
    const spacepdhcg_gtoc12_joint_policy* policy,
    const spacepdhcg_gtoc12_joint_visit* visits,
    const spacepdhcg_gtoc12_joint_stage* stages,
    const double* arrivals, const double* departures,
    const spacepdhcg_orbitweaver_hop_elements* elements,
    const spacepdhcg_gtoc12_joint_cached_cost* records, int32_t record_count,
    double minimum_objective, spacepdhcg_gtoc12_joint_result* results,
    spacepdhcg_gtoc12_joint_selection* selection,
    double* masses, double* inflations, double* proxies, double* collected,
    spacepdhcg_gtoc12_joint_geometry_stats* stats);

/* Generate the ordered pattern-search neighbourhood on CUDA, then use the
 * resident geometry/evaluation pipeline. B=10*N-14 must fit workspace capacity.
 * Input arrivals/departures each contain N incumbent epochs, delta is finite
 * and positive, and all +/-delta epoch values must remain finite. Metadata and
 * sparse override contracts match geometry_host. Optional output epochs contain
 * B*N doubles with selection=null, or N doubles with a selection output. With
 * no selected improvement, compact output epochs retain the incumbent.
 * Candidate order matches JointItinerary.moves, including first-row ties.
 */
int spacepdhcg_gtoc12_joint_mesh_host(
    void* workspace, double delta,
    const spacepdhcg_gtoc12_joint_policy* policy,
    const spacepdhcg_gtoc12_joint_visit* visits,
    const spacepdhcg_gtoc12_joint_stage* stages,
    const double* arrivals, const double* departures,
    const spacepdhcg_orbitweaver_hop_elements* elements,
    const spacepdhcg_gtoc12_joint_cached_cost* records, int32_t record_count,
    double minimum_objective, spacepdhcg_gtoc12_joint_result* results,
    spacepdhcg_gtoc12_joint_selection* selection,
    double* masses, double* inflations, double* proxies, double* collected,
    double* output_arrivals, double* output_departures,
    spacepdhcg_gtoc12_joint_geometry_stats* stats);

/* Entire fixed-order epoch search, including initial evaluation, mesh-level
 * transitions and accepted epochs, executes in one conditional CUDA graph.
 * No per-neighbourhood host decisions/downloads. Same first-row tie rule and
 * objective + 1e-9 improvement gate as mesh_host. Metadata is immutable for the
 * call; learned/cached exact-epoch overrides are uploaded once.
 * remaining_seconds includes native setup; +infinity disables the deadline.
 * Deadlines are soft: checked between complete neighbourhoods. Device timing
 * uses the NVIDIA global nanosecond timer on the validated CUDA targets.
 * stop: 0 levels/move budget exhausted, 1 deadline, 2 initial infeasible,
 * 3 invalid mining stay. All outputs describe the retained incumbent.
 */
typedef struct spacepdhcg_gtoc12_joint_search_report {
    int32_t levels, moves, stop, invalid_stay;
    uint64_t batches, evaluations;
} spacepdhcg_gtoc12_joint_search_report;
int spacepdhcg_gtoc12_joint_search_host(
    void* workspace, const double* mesh_days, int32_t levels,
    int32_t max_moves_per_mesh, double remaining_seconds,
    const spacepdhcg_gtoc12_joint_policy* policy,
    const spacepdhcg_gtoc12_joint_visit* visits,
    const spacepdhcg_gtoc12_joint_stage* stages,
    const double* arrivals, const double* departures,
    const spacepdhcg_orbitweaver_hop_elements* elements,
    const spacepdhcg_gtoc12_joint_cached_cost* records, int32_t record_count,
    spacepdhcg_gtoc12_joint_selection* incumbent,
    double* masses, double* inflations, double* proxies, double* collected,
    double* output_arrivals, double* output_departures,
    spacepdhcg_gtoc12_joint_geometry_stats* stats,
    spacepdhcg_gtoc12_joint_search_report* report);

/* Batched insertion screening. Workspace N is the expanded visit count; base
 * epochs contain N-2 entries. Each layout has visits[N], stages/elements[N-1],
 * and slots[2] = (deploy-after, collect-after) in the original itinerary.
 * CUDA generates four ordered seeds per layout, preserving the scalar borrow
 * arithmetic and its >1 day admission rule. Skipped seeds have failure=18 and
 * enabled=0; their detail outputs are unspecified. Other rows use the ordinary
 * joint result contract. All result rows retain layout-major, seed-major order.
 * records are independently sorted within offsets[L+1], beginning at 0 and
 * ending at record_count. Base epochs are finite and bounded by DBL_MAX/16.
 * Required outputs: results[4L], enabled[4L], stats. Detail/epoch outputs are
 * optional, with 4L rows. L=0 permits null inputs/outputs except stats.
 * Metadata is host supplied; all candidate epochs, geometry and mass arithmetic
 * run on CUDA with one final synchronization. No CPU geometry fallback.
 */
int spacepdhcg_gtoc12_joint_insertions_host(
    void* workspace, int32_t layouts, int32_t camp,
    const spacepdhcg_gtoc12_joint_policy* policy,
    const spacepdhcg_gtoc12_joint_visit* visits,
    const spacepdhcg_gtoc12_joint_stage* stages,
    const int32_t* slots, const double* base_arrivals, const double* base_departures,
    const spacepdhcg_orbitweaver_hop_elements* elements,
    const spacepdhcg_gtoc12_joint_cached_cost* records, int32_t record_count,
    const int32_t* record_offsets,
    spacepdhcg_gtoc12_joint_result* results, uint8_t* enabled,
    double* masses, double* inflations, double* proxies, double* collected,
    double* arrivals, double* departures,
    spacepdhcg_gtoc12_joint_geometry_stats* stats);

#ifdef __cplusplus
}
#endif
