#include "spacepdhcg/cuda/gtoc12_joint_c_api.h"
#include "../internal/gtoc12_joint_geometry.h"
#include "../internal/graph_append.h"

#include <cuda_runtime.h>
#include <cmath>
#include <chrono>
#include <cstddef>
#include <limits>
#include <mutex>
#include <new>

namespace {
using Visit = spacepdhcg_gtoc12_joint_visit;
using Stage = spacepdhcg_gtoc12_joint_stage;
using Cost = spacepdhcg_gtoc12_joint_cost;
using Policy = spacepdhcg_gtoc12_joint_policy;
using Result = spacepdhcg_gtoc12_joint_result;
using Selection = spacepdhcg_gtoc12_joint_selection;
using CachedCost = spacepdhcg_gtoc12_joint_cached_cost;
using GeometryStats = spacepdhcg_gtoc12_joint_geometry_stats;
using SearchReport = spacepdhcg_gtoc12_joint_search_report;
struct SearchState {
    SearchReport report{};
    Result best{};
    int moves_here{};
    uint64_t started{}, budget{};
};
static_assert(sizeof(SearchReport) == 32, "joint search report ABI");
static_assert(sizeof(Visit) == 48, "joint visit ABI");
static_assert(sizeof(Stage) == 72, "joint stage ABI");
static_assert(sizeof(Cost) == 32, "joint cost ABI");
static_assert(sizeof(Policy) == 128, "joint policy ABI");
static_assert(sizeof(Result) == 64, "joint result ABI");
static_assert(sizeof(Selection) == 72, "joint selection ABI");
static_assert(sizeof(CachedCost) == 56, "joint cached cost ABI");
static_assert(sizeof(GeometryStats) == 24, "joint geometry stats ABI");

struct Workspace {
    int device = -1, capacity = 0, n = 0;
    cudaStream_t stream = nullptr;
    Policy* policy = nullptr;
    Visit* visits = nullptr;
    Stage* stages = nullptr;
    double *arrivals = nullptr, *departures = nullptr;
    double *base_arrivals = nullptr, *base_departures = nullptr;
    Cost* costs = nullptr;
    Result* results = nullptr;
    Selection* selection = nullptr;
    double *masses = nullptr, *inflations = nullptr, *proxies = nullptr, *collected = nullptr;
    spacepdhcg_orbitweaver_hop_elements* elements = nullptr;
    spacepdhcg_orbitweaver_hop_request* hop_requests = nullptr;
    spacepdhcg_orbitweaver_hop_result* hop_results = nullptr;
    CachedCost* cached_costs = nullptr;
    int cached_capacity = 0;
    GeometryStats* geometry_stats = nullptr;
    SearchState* search = nullptr;
    double *mesh_days = nullptr, *best_payload = nullptr;
    int mesh_capacity = 0;
    std::mutex mutex;
};

__device__ Result failure(int code, int rounds = 0) {
    Result result{};
    result.failure = code;
    result.objective = -INFINITY;
    result.rounds = rounds;
    return result;
}

/* Same piecewise-linear np.interp table and operation order as return_base in
 * gtoc12_retime.cu. No fast-math substitution or fused multiply-add is allowed.
 */
__device__ double return_base(double tof) {
    constexpr double days[] = {352, 420, 450, 480, 510, 540, 578, 630, 690, 810};
    constexpr double values[] = {1.323, 1.383, 1.295, 1.195, 1.099, .977, .885, .930, .932, 1.014};
    if (tof <= days[0]) return values[0];
    for (int i = 1; i < 10; ++i)
        if (tof <= days[i])
            return values[i - 1] + (values[i] - values[i - 1]) / (days[i] - days[i - 1])
                * (tof - days[i - 1]);
    return values[9];
}

__device__ double authority(const Policy& p, double mass, double tof) {
    // thrust_authority_km_s(mass, tof, 1.0), including its left-to-right order.
    return ((p.thrust / mass * 1e-3) * tof) * 86400.0;
}

__device__ double leg_inflation(const Policy& p, const Stage& s,
    double lambert, double mass, double tof) {
    if (s.model == 0) return s.flat;
    const double ratio = lambert / fmax(authority(p, mass, tof), 1e-12);
    if (s.model == 1) return (s.floor + s.slope * ratio) * s.calibration;
    const double correction = 1.0 + 0.6 * (ratio - 0.33);
    const double model = fmax(return_base(tof) * fmin(fmax(correction, 0.85), 1.2), 0.85);
    return model * s.calibration;
}

__device__ Result forward(int n, const Policy& p, const Visit* visits,
    const Stage* stages, const double* arr, const double* dep, const Cost* costs,
    const double* collected, double* masses, double* inflations, double* proxies, int rounds) {
    Result result{};
    result.rounds = rounds;
    double mass = p.initial_mass, propellant_total = 0.0;
    int measured_count = 0;
    for (int j = 0; j < n - 1; ++j) {
        const Visit v = visits[j];
        const Stage s = stages[j];
        const Cost cost = costs[j];
        if (v.collect) mass += collected[j];
        const double tof = arr[j + 1] - dep[j];
        const double lambert = cost.lambert;
        double effective, inflation, proxy;
        if (cost.measured && fabs(cost.measured_mass - mass) <= p.measured_mass_tolerance) {
            effective = cost.measured_delta_v;
            inflation = isfinite(lambert) && lambert > 0.0 ? effective / lambert : 1.0;
            proxy = isfinite(lambert) ? lambert : effective;
            ++measured_count;
        } else {
            if (!isfinite(lambert)) return failure(SPACEPDHCG_JOINT_LEG_INFEASIBLE, rounds);
            const bool below_floor = s.earth_out && p.free_earth_leg
                && !isnan(p.earth_out_tof_floor) && tof < p.earth_out_tof_floor - 1e-9;
            if (below_floor) {
                if (!p.screen_earth_out)
                    return failure(SPACEPDHCG_JOINT_EARTH_OUT_UNMEASURED_BELOW_FLOOR, rounds);
                inflation = !isnan(p.earth_out_inflation) ? p.earth_out_inflation
                    : leg_inflation(p, s, lambert, mass, tof);
            } else {
                // The gate uses the raw authority denominator, not the model's floor.
                if (lambert / authority(p, mass, tof) > s.ratio_limit)
                    return failure(SPACEPDHCG_JOINT_LEG_AUTHORITY, rounds);
                inflation = leg_inflation(p, s, lambert, mass, tof);
            }
            effective = lambert * inflation;
            proxy = lambert;
        }
        const double propellant = mass * (1.0 - exp(-effective / p.exhaust));
        masses[j] = mass;
        inflations[j] = inflation;
        proxies[j] = proxy;
        mass -= propellant;
        propellant_total += propellant;
        if (visits[j + 1].deploy) mass -= p.miner_mass;
    }
    // Python dict values are summed in collect-visit insertion order. Do not
    // parallel-reduce these short sums, nor sum zero entries from other visits.
    double payload = 0.0;
    for (int j = 0; j < n; ++j) if (visits[j].collect) payload += collected[j];
    result.spare = mass - (p.dry_mass + payload);
    result.propellant = propellant_total;
    result.final_mass = mass;
    result.mass_count = n - 1;
    result.measured_legs = measured_count;
    if (result.spare < -1e-9) {
        result.failure = SPACEPDHCG_JOINT_MASS_BELOW_DRY_PLUS_COLLECTED;
        result.objective = -INFINITY;
        return result;
    }
    double weighted = 0.0;
    for (int j = 0; j < n; ++j)
        if (visits[j].collect) weighted += visits[j].weight * collected[j];
    result.weighted = weighted;
    result.collected = payload;
    result.objective = weighted + p.margin_price * result.spare;
    return result;
}

__global__ void evaluate_candidates(int count, int n, const Policy* policy,
    const Visit* visits, const Stage* stages, const double* arrivals, const double* departures,
    const Cost* all_costs, Result* results, double* all_masses, double* all_inflations,
    double* all_proxies, double* all_collected) {
    const size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= size_t(count)) return;
    const Policy p = *policy;
    const double* arr = arrivals + index * n;
    const double* dep = departures + index * n;
    const Cost* costs = all_costs + index * (n - 1);
    double* masses = all_masses + index * (n - 1);
    double* inflations = all_inflations + index * (n - 1);
    double* proxies = all_proxies + index * (n - 1);
    double* collected = all_collected + index * n;
    for (int j = 0; j < n; ++j) collected[j] = 0.0;
    for (int j = 0; j < n - 1; ++j) masses[j] = inflations[j] = proxies[j] = 0.0;
    Result result = failure(SPACEPDHCG_JOINT_OK);
    if (arr[0] < p.mission_start - 1e-9) {
        results[index] = failure(SPACEPDHCG_JOINT_LAUNCH_BEFORE_WINDOW); return;
    }
    if (arr[n - 1] > p.latest_arrival + 1e-9) {
        results[index] = failure(SPACEPDHCG_JOINT_RETURN_AFTER_WINDOW); return;
    }
    if (fabs(dep[0] - arr[0]) > 1e-9 || fabs(dep[n - 1] - arr[n - 1]) > 1e-9) {
        results[index] = failure(SPACEPDHCG_JOINT_EARTH_DWELL); return;
    }
    for (int j = 1; j < n - 1; ++j) {
        const double dwell = dep[j] - arr[j];
        if (dwell < -1e-9) {
            results[index] = failure(SPACEPDHCG_JOINT_NEGATIVE_DWELL); return;
        }
        if (dwell > visits[j].dwell_limit + 1e-9) {
            results[index] = failure(SPACEPDHCG_JOINT_DWELL_TOO_LONG); return;
        }
        if (!isnan(visits[j].pinned_arrival) && fabs(arr[j] - visits[j].pinned_arrival) > 1e-6) {
            results[index] = failure(SPACEPDHCG_JOINT_PINNED_ARRIVAL); return;
        }
    }
    for (int j = 0; j < n - 1; ++j) {
        const double tof = arr[j + 1] - dep[j];
        if (tof < stages[j].tof_min - 1e-9 || tof > stages[j].tof_max + 1e-9) {
            results[index] = failure(SPACEPDHCG_JOINT_TOF_OUTSIDE_LIMITS); return;
        }
    }
    // Structural failures precede all mining-stay checks, as in evaluate().
    for (int j = 0; j < n; ++j) {
        if (visits[j].structural_failure) {
            results[index] = failure(visits[j].structural_failure); return;
        }
        if (visits[j].collect && visits[j].donor == -2) {
            results[index] = failure(SPACEPDHCG_JOINT_COLLECT_WITHOUT_DEPLOY); return;
        }
    }
    for (int j = 0; j < n; ++j) {
        if (!visits[j].collect) continue;
        const double deployed = visits[j].donor >= 0 ? arr[visits[j].donor] : visits[j].foreign_epoch;
        const double stay = dep[j] - deployed;
        if (stay < p.minimum_stay - 1e-6) {
            results[index] = failure(SPACEPDHCG_JOINT_STAY_TOO_SHORT); return;
        }
        if (!isfinite(stay) || stay < 0.0) {
            results[index] = failure(SPACEPDHCG_JOINT_INVALID_STAY); return;
        }
        collected[j] = p.mining_rate * stay / p.year_days;
    }
    for (int round = 0; round < 4; ++round) {
        result = forward(n, p, visits, stages, arr, dep, costs, collected,
            masses, inflations, proxies, round + 1);
        if (result.failure != SPACEPDHCG_JOINT_MASS_BELOW_DRY_PLUS_COLLECTED) break;
        double total = 0.0;
        for (int j = 0; j < n; ++j) if (visits[j].collect) total += collected[j];
        const double deficit = -result.spare;
        if (total <= 0.0 || deficit >= total) {
            result = failure(SPACEPDHCG_JOINT_MASS_BELOW_DRY, round + 1); break;
        }
        // The fourth failed pass is returned by Python; its subsequent local
        // dictionary scaling cannot change that Evaluation. Keep matching detail
        // outputs for the returned pass instead of an unevaluated fifth payload.
        if (round == 3) break;
        const double scale = (total - 1.02 * deficit) / total;
        for (int j = 0; j < n; ++j) if (visits[j].collect) collected[j] *= scale;
    }
    results[index] = result;
}

// Reduce only scores and original indices, retaining first-in-order ties.
// Gathering into row zero is safe: a nonzero winning row cannot overlap it.
__global__ void select_candidate(int count, int n, double minimum,
    const Result* results, Selection* selection, double* masses,
    double* inflations, double* proxies, double* collected) {
    __shared__ double scores[128];
    __shared__ int indices[128], invalid[128];
    const int lane = threadIdx.x;
    double best = -INFINITY;
    int index = INT32_MAX, bad_stay = 0;
    const double threshold = minimum + 1e-9;
    for (int64_t row = lane; row < count; row += 128) {
        const Result value = results[row];
        bad_stay |= value.failure == SPACEPDHCG_JOINT_INVALID_STAY;
        if (value.failure == 0 && value.objective > threshold
            && (value.objective > best || (value.objective == best && row < index))) {
            best = value.objective;
            index = int(row);
        }
    }
    scores[lane] = best; indices[lane] = index; invalid[lane] = bad_stay;
    __syncthreads();
    for (int offset = 64; offset > 0; offset /= 2) {
        if (lane < offset) {
            const int other = lane + offset;
            if (scores[other] > scores[lane]
                || (scores[other] == scores[lane] && indices[other] < indices[lane])) {
                scores[lane] = scores[other];
                indices[lane] = indices[other];
            }
            invalid[lane] |= invalid[other];
        }
        __syncthreads();
    }
    const int winner = indices[0] == INT32_MAX ? -1 : indices[0];
    if (lane == 0) {
        selection->index = winner;
        selection->invalid_stay = invalid[0];
        selection->value = winner >= 0 ? results[winner] : failure(SPACEPDHCG_JOINT_OK);
    }
    if (winner < 0) return;
    const size_t legs = size_t(winner) * (n - 1), visits = size_t(winner) * n;
    for (int64_t j = lane; j < n - 1; j += 128) {
        masses[j] = masses[legs + j];
        inflations[j] = inflations[legs + j];
        proxies[j] = proxies[legs + j];
    }
    for (int64_t j = lane; j < n; j += 128) collected[j] = collected[visits + j];
}

// The conditional-loop reducer uses one complete warp. All lanes participate
// in each shuffle, independent of their candidate count or feasibility. This
// avoids block barriers in the repeatedly executed graph and keeps stable ties.
__global__ void select_search_candidate(int count, int n, const Result* results,
    Selection* selection, double* masses, double* inflations, double* proxies,
    double* collected, const SearchState* search) {
    const int lane=threadIdx.x;
    double best=-INFINITY;
    int index=INT32_MAX, invalid=0;
    const double threshold=search->best.objective+1e-9;
    for(int64_t row=lane;row<count;row+=32) {
        const Result value=results[row];
        invalid|=value.failure==SPACEPDHCG_JOINT_INVALID_STAY;
        if(value.failure==0 && value.objective>threshold
            && (value.objective>best || (value.objective==best && row<index))) {
            best=value.objective;index=int(row);
        }
    }
    for(int offset=16;offset>0;offset/=2) {
        const double other=__shfl_down_sync(0xffffffff,best,offset);
        const int other_index=__shfl_down_sync(0xffffffff,index,offset);
        const int other_invalid=__shfl_down_sync(0xffffffff,invalid,offset);
        if(lane+offset<32) {
            if(other>best || (other==best && other_index<index)) {best=other;index=other_index;}
            invalid|=other_invalid;
        }
    }
    const int winner=__shfl_sync(0xffffffff,index,0);
    if(lane==0) {
        selection->index=winner==INT32_MAX ? -1 : winner;
        selection->invalid_stay=invalid;
        selection->value=winner==INT32_MAX ? failure(SPACEPDHCG_JOINT_OK) : results[winner];
    }
    if(winner==INT32_MAX)return;
    for(int64_t j=lane;j<n-1;j+=32) {
        const size_t cell=size_t(winner)*(n-1)+j;
        masses[j]=masses[cell];inflations[j]=inflations[cell];proxies[j]=proxies[cell];
    }
    for(int64_t j=lane;j<n;j+=32)collected[j]=collected[size_t(winner)*n+j];
}

bool flag(int value) { return value == 0 || value == 1; }
__global__ void generate_mesh(int n, int count, double delta,
    const double* base_arrivals, const double* base_departures,
    double* arrivals, double* departures,
    const SearchState* search = nullptr, const double* mesh_days = nullptr) {
    const size_t cell = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (cell >= size_t(count) * n) return;
    const int row = int(cell / n), j = int(cell % n);
    const int per_sign = count / 2, move = row % per_sign;
    const double step = search ? mesh_days[search->report.levels] : delta;
    const double d = row < per_sign ? step : -step;
    bool touched = false, arrival_shift = false, departure_shift = false;
    if (move < 2) {
        touched = j == (move == 0 ? 0 : n - 1);
        arrival_shift = departure_shift = true;
    } else if (move < 2 + 3 * (n - 2)) {
        const int local = move - 2, kind = local % 3;
        touched = j == 1 + local / 3;
        arrival_shift = kind != 1; departure_shift = kind != 0;
    } else if (move < per_sign - 1) {
        const int local = move - (2 + 3 * (n - 2)), pivot = 1 + local / 2;
        if (local % 2 == 0) {
            touched = j <= pivot; arrival_shift = true; departure_shift = j < pivot;
        } else {
            touched = j >= pivot; arrival_shift = j > pivot; departure_shift = true;
        }
    } else {
        touched = arrival_shift = departure_shift = true;
    }
    // Even a zero component is added for touched visits, matching Python's
    // in-place additions; untouched epochs are copied, including signed zero.
    arrivals[cell] = touched ? base_arrivals[j] + (arrival_shift ? d : 0.0) : base_arrivals[j];
    departures[cell] = touched ? base_departures[j] + (departure_shift ? d : 0.0) : base_departures[j];
}

__global__ void select_mesh_epochs(int n, const Selection* selected,
    const double* arrivals, const double* departures,
    double* output_arrivals, double* output_departures) {
    const int j = int(blockIdx.x * blockDim.x + threadIdx.x);
    if (j >= n || selected->index < 0) return;
    const size_t cell = size_t(selected->index) * n + j;
    output_arrivals[j] = arrivals[cell]; output_departures[j] = departures[cell];
}

__device__ uint64_t global_nanoseconds() {
    uint64_t value;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(value));
    return value;
}

// One block owns the controller and copies the compact incumbent payload.
// The expensive candidate/geometry work remains distributed across blocks.
__global__ void search_save(int n, int count, int max_moves, bool initial,
    SearchState* state, const Result* results, const Selection* selection,
    const double* arrivals, const double* departures,
    double* base_arrivals, double* base_departures,
    const double* masses, const double* inflations, const double* proxies,
    const double* collected, double* best_payload) {
    const int lane = threadIdx.x;
    const bool invalid = initial ? results[0].failure == SPACEPDHCG_JOINT_INVALID_STAY
                                 : selection->invalid_stay != 0;
    const bool accept = initial || (!invalid && selection->index >= 0);
    const int row = initial ? 0 : selection->index;
    if (accept) {
        for (int j = lane; j < n; j += blockDim.x) {
            base_arrivals[j] = arrivals[size_t(row)*n+j];
            base_departures[j] = departures[size_t(row)*n+j];
            best_payload[3*(n-1)+j] = collected[j];
        }
        for (int j = lane; j < n-1; j += blockDim.x) {
            best_payload[j] = masses[j];
            best_payload[n-1+j] = inflations[j];
            best_payload[2*(n-1)+j] = proxies[j];
        }
    }
    if (lane == 0) {
        if (accept) state->best = initial ? results[0] : selection->value;
        state->report.invalid_stay |= invalid;
        if (initial) {
            state->report.evaluations = 1;
            if (state->best.failure) state->report.stop = 2;
        } else {
            ++state->report.batches;
            state->report.evaluations += count;
            if (accept) { ++state->report.moves; ++state->moves_here; }
            if (!accept || state->moves_here >= max_moves) {
                ++state->report.levels; state->moves_here = 0;
            }
        }
        if (invalid) state->report.stop = 3;
    }
}

__global__ void search_start(SearchState* state) {
    state->started = global_nanoseconds();
}
__global__ void search_gate(SearchState* state, int levels, int max_moves,
    cudaGraphConditionalHandle loop) {
    bool active = state->report.stop == 0 && state->report.levels < levels && max_moves > 0;
    if (active && global_nanoseconds() - state->started >= state->budget) {
        state->report.stop = 1; active = false;
    }
    cudaGraphSetConditional(loop, active);
}
__global__ void search_finish(const SearchState* state, Selection* output) {
    output->index = 0;
    output->invalid_stay = state->report.invalid_stay;
    output->value = state->best;
}

bool finite_or_nan(double value) { return std::isfinite(value) || std::isnan(value); }
bool nonnegative_bound(double value) { return !std::isnan(value) && value >= 0.0; }
bool correct_device(const Workspace* w) {
    int device = -1;
    return w && cudaGetDevice(&device) == cudaSuccess && device == w->device;
}

int validate(int n, int count, const Policy* p, const Visit* visits, const Stage* stages,
    const double* arr, const double* dep, const Cost* costs, const Result* results,
    bool inspect_costs = true) {
    if (!p || !visits || !stages || !arr || !dep || (inspect_costs && !costs) || !results) return 1;
    if (p->reserved0 || p->reserved1 || !flag(p->free_earth_leg) || !flag(p->screen_earth_out)) return 4;
    if (!std::isfinite(p->mission_start) || !std::isfinite(p->latest_arrival)
        || p->latest_arrival < p->mission_start || !std::isfinite(p->initial_mass)
        || p->initial_mass <= 0.0 || !std::isfinite(p->dry_mass) || p->dry_mass < 0.0
        || !std::isfinite(p->miner_mass) || p->miner_mass < 0.0
        || !std::isfinite(p->minimum_stay) || p->minimum_stay < 0.0
        || !std::isfinite(p->mining_rate) || p->mining_rate < 0.0
        || !std::isfinite(p->year_days) || p->year_days <= 0.0
        || !std::isfinite(p->thrust) || p->thrust <= 0.0
        || !std::isfinite(p->exhaust) || p->exhaust <= 0.0
        || !std::isfinite(p->measured_mass_tolerance) || p->measured_mass_tolerance < 0.0
        || !std::isfinite(p->margin_price) || !finite_or_nan(p->earth_out_tof_floor)
        || !finite_or_nan(p->earth_out_inflation)) return 1;
    for (int j = 0; j < n; ++j) {
        const Visit& v = visits[j];
        if (!flag(v.deploy) || !flag(v.collect)) return 4;
        if (v.structural_failure != 0 && v.structural_failure != SPACEPDHCG_JOINT_DOUBLE_DEPLOY
            && v.structural_failure != SPACEPDHCG_JOINT_DOUBLE_COLLECT
            && v.structural_failure != SPACEPDHCG_JOINT_COLLECT_WITHOUT_DEPLOY) return 4;
        if (v.donor < -2 || v.donor >= n || !finite_or_nan(v.pinned_arrival)
            || !nonnegative_bound(v.dwell_limit) || !std::isfinite(v.weight)) return 1;
        if (v.collect && v.donor == -1 && !std::isfinite(v.foreign_epoch)) return 1;
        if (v.collect && v.donor >= 0 && !visits[v.donor].deploy) return 1;
    }
    for (int j = 0; j < n - 1; ++j) {
        const Stage& s = stages[j];
        if (s.model < 0 || s.model > 2 || !flag(s.earth_out) || s.reserved0 || s.reserved1) return 4;
        if (!std::isfinite(s.tof_min) || !nonnegative_bound(s.tof_max) || s.tof_max < s.tof_min
            || !nonnegative_bound(s.ratio_limit) || !std::isfinite(s.flat)
            || !std::isfinite(s.floor) || !std::isfinite(s.slope) || !std::isfinite(s.calibration)) return 1;
    }
    const size_t visits_count = size_t(count) * n;
    for (size_t k = 0; k < visits_count; ++k)
        if (!std::isfinite(arr[k]) || !std::isfinite(dep[k])) return 1;
    const size_t legs_count = size_t(count) * (n - 1);
    for (size_t k = 0; inspect_costs && k < legs_count; ++k) {
        const Cost& c = costs[k];
        if (!flag(c.measured) || c.reserved) return 4;
        if (c.measured && (!std::isfinite(c.measured_delta_v) || !std::isfinite(c.measured_mass))) return 1;
    }
    return 0;
}

template<class T> bool allocate(T*& output, size_t count) {
    return count <= std::numeric_limits<size_t>::max() / sizeof(T)
        && cudaMalloc(&output, count * sizeof(T)) == cudaSuccess;
}
template<class T> bool upload(T* output, const T* input, size_t count, cudaStream_t stream) {
    return cudaMemcpyAsync(output, input, count * sizeof(T), cudaMemcpyHostToDevice, stream) == cudaSuccess;
}
template<class T> bool download(T* output, const T* input, size_t count, cudaStream_t stream) {
    return !output || cudaMemcpyAsync(output, input, count * sizeof(T), cudaMemcpyDeviceToHost, stream) == cudaSuccess;
}
bool release(Workspace* w) {
    bool ok = true;
    if (w->stream && cudaStreamSynchronize(w->stream) != cudaSuccess) ok = false;
    const auto free_buffer = [&ok](void* pointer) {
        if (pointer && cudaFree(pointer) != cudaSuccess) ok = false;
    };
    free_buffer(w->policy); free_buffer(w->visits); free_buffer(w->stages);
    free_buffer(w->arrivals); free_buffer(w->departures); free_buffer(w->costs);
    free_buffer(w->base_arrivals); free_buffer(w->base_departures);
    free_buffer(w->results); free_buffer(w->selection); free_buffer(w->masses); free_buffer(w->inflations);
    free_buffer(w->proxies); free_buffer(w->collected);
    free_buffer(w->elements); free_buffer(w->hop_requests); free_buffer(w->hop_results);
    free_buffer(w->cached_costs); free_buffer(w->geometry_stats);
    free_buffer(w->search); free_buffer(w->mesh_days); free_buffer(w->best_payload);
    if (w->stream && cudaStreamDestroy(w->stream) != cudaSuccess) ok = false;
    return ok;
}

bool valid_orbit(const spacepdhcg_orbitweaver_elements& e) {
    return std::isfinite(e.epoch) && std::isfinite(e.a) && e.a>0.0
        && std::isfinite(e.e) && e.e>=0.0 && e.e<1.0
        && std::isfinite(e.inclination) && std::isfinite(e.node)
        && std::isfinite(e.perihelion) && std::isfinite(e.mean);
}
bool key_before(const CachedCost& a, const CachedCost& b) {
    return a.leg<b.leg || (a.leg==b.leg && (a.departure<b.departure
        || (a.departure==b.departure && a.arrival<b.arrival)));
}
__global__ void clear_geometry_costs(size_t count, Cost* costs) {
    const size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i<count){costs[i]={};costs[i].lambert=NAN;}
}
} // namespace

extern "C" int spacepdhcg_gtoc12_joint_create(
    int32_t device, int32_t capacity, int32_t n, void** output) {
    if (!output) return 1;
    *output = nullptr;
    if (device < 0 || capacity < 1 || n < 2) return 1;
    const size_t cap = size_t(capacity), visits_count = cap * size_t(n);
    if (visits_count / cap != size_t(n)
        || visits_count > std::numeric_limits<size_t>::max() / sizeof(Cost)) return 1;
    auto* w = new (std::nothrow) Workspace;
    if (!w) return 2;
    int previous = -1;
    if (cudaGetDevice(&previous) != cudaSuccess || cudaSetDevice(device) != cudaSuccess) {
        delete w; return 2;
    }
    w->device = device; w->capacity = capacity; w->n = n;
    const size_t legs_count = cap * size_t(n - 1);
    bool ok = cudaStreamCreateWithFlags(&w->stream, cudaStreamNonBlocking) == cudaSuccess
        && allocate(w->policy, 1) && allocate(w->visits, size_t(n))
        && allocate(w->stages, size_t(n - 1)) && allocate(w->arrivals, visits_count)
        && allocate(w->departures, visits_count) && allocate(w->costs, legs_count)
        && allocate(w->results, cap) && allocate(w->selection, 1) && allocate(w->masses, legs_count)
        && allocate(w->inflations, legs_count) && allocate(w->proxies, legs_count)
        && allocate(w->collected, visits_count);
    if (!ok) release(w);
    if (previous != device && cudaSetDevice(previous) != cudaSuccess) {
        // Keep destruction on the owning device even if restoring the caller's
        // device fails. No partially initialized workspace escapes.
        if (ok) { cudaSetDevice(device); release(w); }
        ok = false;
    }
    if (!ok) { delete w; return 2; }
    *output = w;
    return 0;
}

static int evaluate_host_impl(void* opaque, int32_t count,
    const Policy* policy, const Visit* visits, const Stage* stages,
    const double* arrivals, const double* departures, const Cost* costs, Result* results,
    double* masses, double* inflations, double* proxies, double* collected,
    Selection* selection = nullptr, double minimum = 0.0) {
    auto* w = static_cast<Workspace*>(opaque);
    if (!correct_device(w) || count < 0 || count > w->capacity) return 1;
    std::unique_lock<std::mutex> lock(w->mutex, std::try_to_lock);
    if (!lock.owns_lock()) return 3;
    if (count == 0) {
        if (selection) { *selection = {}; selection->index = -1; }
        return 0;
    }
    const int status = validate(w->n, count, policy, visits, stages, arrivals, departures, costs,
        selection ? &selection->value : results);
    if (status) return status;
    const size_t visit_count = size_t(count) * w->n;
    const size_t leg_count = size_t(count) * (w->n - 1);
    if (!upload(w->policy, policy, 1, w->stream)
        || !upload(w->visits, visits, size_t(w->n), w->stream)
        || !upload(w->stages, stages, size_t(w->n - 1), w->stream)
        || !upload(w->arrivals, arrivals, visit_count, w->stream)
        || !upload(w->departures, departures, visit_count, w->stream)
        || !upload(w->costs, costs, leg_count, w->stream)) {
        cudaStreamSynchronize(w->stream); return 2;
    }
    const unsigned blocks = unsigned((size_t(count) + 127) / 128);
    evaluate_candidates<<<blocks, 128, 0, w->stream>>>(count, w->n, w->policy, w->visits,
        w->stages, w->arrivals, w->departures, w->costs, w->results,
        w->masses, w->inflations, w->proxies, w->collected);
    if (cudaGetLastError() != cudaSuccess) {
        cudaStreamSynchronize(w->stream); return 2;
    }
    if (selection) {
        select_candidate<<<1, 128, 0, w->stream>>>(count, w->n, minimum, w->results,
            w->selection, w->masses, w->inflations, w->proxies, w->collected);
    }
    const size_t output_legs = selection ? size_t(w->n - 1) : leg_count;
    const size_t output_visits = selection ? size_t(w->n) : visit_count;
    if (cudaGetLastError() != cudaSuccess
        || !(selection ? download(selection, w->selection, 1, w->stream)
                       : download(results, w->results, size_t(count), w->stream))
        || !download(masses, w->masses, output_legs, w->stream)
        || !download(inflations, w->inflations, output_legs, w->stream)
        || !download(proxies, w->proxies, output_legs, w->stream)
        || !download(collected, w->collected, output_visits, w->stream)) {
        cudaStreamSynchronize(w->stream); return 2;
    }
    return cudaStreamSynchronize(w->stream) == cudaSuccess ? 0 : 2;
}

extern "C" int spacepdhcg_gtoc12_joint_evaluate_host(void* opaque, int32_t count,
    const Policy* policy, const Visit* visits, const Stage* stages,
    const double* arrivals, const double* departures, const Cost* costs, Result* results,
    double* masses, double* inflations, double* proxies, double* collected) {
    return evaluate_host_impl(opaque, count, policy, visits, stages, arrivals, departures,
        costs, results, masses, inflations, proxies, collected);
}

extern "C" int spacepdhcg_gtoc12_joint_best_host(void* opaque, int32_t count,
    const Policy* policy, const Visit* visits, const Stage* stages,
    const double* arrivals, const double* departures, const Cost* costs, double minimum,
    Selection* selection, double* masses, double* inflations, double* proxies, double* collected) {
    if (!selection) return 1;
    return evaluate_host_impl(opaque, count, policy, visits, stages, arrivals, departures,
        costs, nullptr, masses, inflations, proxies, collected, selection, minimum);
}

extern "C" int spacepdhcg_gtoc12_joint_destroy(void** output) {
    if (!output) return 1;
    auto* w = static_cast<Workspace*>(*output);
    if (!w) return 0;
    std::unique_lock<std::mutex> lock(w->mutex, std::try_to_lock);
    if (!lock.owns_lock()) return 3;
    int previous = -1;
    if (cudaGetDevice(&previous) != cudaSuccess || cudaSetDevice(w->device) != cudaSuccess) return 2;
    bool ok = release(w);
    if (previous != w->device && cudaSetDevice(previous) != cudaSuccess) ok = false;
    lock.unlock();
    delete w;
    *output = nullptr;
    return ok ? 0 : 2;
}

struct SearchConfig {
    const double* days;
    int levels, max_moves;
    double seconds;
    SearchReport* report;
    std::chrono::steady_clock::time_point entered;
};

static int search_graph(Workspace* w, int count, int record_count, const SearchConfig& config,
    Selection* selection, double* masses, double* inflations, double* proxies,
    double* collected, double* output_arrivals, double* output_departures, GeometryStats* stats) {
    struct Graph {
        cudaStream_t stream;
        cudaGraph_t graph{};
        cudaGraphExec_t executable{};
        ~Graph() {
            cudaStreamSynchronize(stream);
            if(executable) cudaGraphExecDestroy(executable);
            if(graph) cudaGraphDestroy(graph);
        }
    } graph{w->stream};
    if(!w->search && !allocate(w->search,1)) return 2;
    if(!w->best_payload && !allocate(w->best_payload,size_t(4)*w->n-3)) return 2;
    if(config.levels>w->mesh_capacity) {
        double* next{};
        if(!allocate(next,size_t(config.levels))) return 2;
        if(w->mesh_days && cudaFree(w->mesh_days)!=cudaSuccess) {cudaFree(next);return 2;}
        w->mesh_days=next; w->mesh_capacity=config.levels;
    }
    if(config.levels && !upload(w->mesh_days,config.days,size_t(config.levels),w->stream)) return 2;
    if(cudaGraphCreate(&graph.graph,0)!=cudaSuccess) return 2;
    cudaGraphConditionalHandle loop{};
    if(cudaGraphConditionalHandleCreate(&loop,graph.graph,0,cudaGraphCondAssignDefault)!=cudaSuccess) return 2;
    const auto evaluate=[&](int rows) {
        const size_t legs=size_t(rows)*(w->n-1);
        clear_geometry_costs<<<unsigned((legs+127)/128),128,0,w->stream>>>(legs,w->costs);
        if(cudaGetLastError()!=cudaSuccess) return cudaErrorUnknown;
        const auto values=[&]() {
            evaluate_candidates<<<unsigned((size_t(rows)+127)/128),128,0,w->stream>>>(rows,w->n,
                w->policy,w->visits,w->stages,w->arrivals,w->departures,w->costs,w->results,
                w->masses,w->inflations,w->proxies,w->collected);
            return cudaGetLastError();
        };
        auto error=values();if(error!=cudaSuccess)return error;
        error=spacepdhcg_joint_geometry_launch(rows,w->n,w->elements,w->arrivals,w->departures,
            w->results,w->cached_costs,record_count,w->costs,w->hop_requests,w->hop_results,
            w->geometry_stats,w->stream);
        return error==cudaSuccess ? values() : error;
    };
    const auto save=[&](bool initial) {
        search_save<<<1,128,0,w->stream>>>(w->n,count,config.max_moves,initial,w->search,
            w->results,w->selection,w->arrivals,w->departures,w->base_arrivals,w->base_departures,
            w->masses,w->inflations,w->proxies,w->collected,w->best_payload);
        return cudaGetLastError();
    };
    const auto gate=[&]() {
        search_gate<<<1,1,0,w->stream>>>(w->search,config.levels,config.max_moves,loop);
        return cudaGetLastError();
    };
    cudaGraphNode_t head{};
    if(spacepdhcg_graph_append(w->stream,graph.graph,nullptr,0,[&]() {
        search_start<<<1,1,0,w->stream>>>(w->search);
        if(cudaGetLastError()!=cudaSuccess)return cudaErrorUnknown;
        for(auto pair : {std::pair<double*,double*>{w->arrivals,w->base_arrivals},
                        std::pair<double*,double*>{w->departures,w->base_departures}}) {
            auto error=cudaMemcpyAsync(pair.first,pair.second,size_t(w->n)*sizeof(double),cudaMemcpyDeviceToDevice,w->stream);
            if(error!=cudaSuccess)return error;
        }
        for(auto pointer : {w->masses,w->inflations,w->proxies,w->collected}) {
            auto error=cudaMemsetAsync(pointer,0,size_t(w->n-1)*sizeof(double),w->stream);
            if(error!=cudaSuccess)return error;
        }
        auto error=cudaMemsetAsync(w->collected,0,size_t(w->n)*sizeof(double),w->stream);
        if(error!=cudaSuccess)return error;
        error=evaluate(1);if(error!=cudaSuccess)return error;
        error=save(true);return error==cudaSuccess ? gate() : error;
    },&head)!=cudaSuccess)return 2;
    cudaGraphNodeParams params{};params.type=cudaGraphNodeTypeConditional;
    params.conditional.handle=loop;params.conditional.type=cudaGraphCondTypeWhile;
    params.conditional.size=1;cudaGraphNode_t outer{};
    if(cudaGraphAddNode(&outer,graph.graph,&head,1,&params)!=cudaSuccess)return 2;
    cudaGraphNode_t tail{};
    if(spacepdhcg_graph_append(w->stream,params.conditional.phGraph_out[0],nullptr,0,[&]() {
        generate_mesh<<<unsigned((size_t(count)*w->n+127)/128),128,0,w->stream>>>(w->n,count,0,
            w->base_arrivals,w->base_departures,w->arrivals,w->departures,w->search,w->mesh_days);
        auto error=cudaGetLastError();if(error!=cudaSuccess)return error;
        error=evaluate(count);if(error!=cudaSuccess)return error;
        select_search_candidate<<<1,32,0,w->stream>>>(count,w->n,w->results,w->selection,
            w->masses,w->inflations,w->proxies,w->collected,w->search);
        error=cudaGetLastError();if(error!=cudaSuccess)return error;
        error=save(false);return error==cudaSuccess ? gate() : error;
    },&tail)!=cudaSuccess)return 2;
    if(spacepdhcg_graph_append(w->stream,graph.graph,&outer,1,[&]() {
        search_finish<<<1,1,0,w->stream>>>(w->search,w->selection);return cudaGetLastError();
    },&tail)!=cudaSuccess || cudaGraphInstantiate(&graph.executable,graph.graph,0)!=cudaSuccess)return 2;
    SearchState initial{};
    const double elapsed=std::chrono::duration<double>(std::chrono::steady_clock::now()-config.entered).count();
    const double remaining=std::max(0.0,config.seconds-elapsed);
    initial.budget=remaining>=double(UINT64_MAX)*1e-9 ? UINT64_MAX : uint64_t(remaining*1e9);
    if(!upload(w->search,&initial,1,w->stream)
        ||cudaGraphLaunch(graph.executable,w->stream)!=cudaSuccess
        ||!download(selection,w->selection,1,w->stream)
        ||!download(config.report,&w->search->report,1,w->stream)
        ||!download(masses,w->best_payload,size_t(w->n-1),w->stream)
        ||!download(inflations,w->best_payload+w->n-1,size_t(w->n-1),w->stream)
        ||!download(proxies,w->best_payload+2*(w->n-1),size_t(w->n-1),w->stream)
        ||!download(collected,w->best_payload+3*(w->n-1),size_t(w->n),w->stream)
        ||!download(output_arrivals,w->base_arrivals,size_t(w->n),w->stream)
        ||!download(output_departures,w->base_departures,size_t(w->n),w->stream)
        ||!download(stats,w->geometry_stats,1,w->stream))return 2;
    return cudaStreamSynchronize(w->stream)==cudaSuccess ? 0 : 2;
}

static int geometry_host_impl(void* opaque, int32_t count,
    const Policy* policy, const Visit* visits, const Stage* stages,
    const double* arrivals, const double* departures,
    const spacepdhcg_orbitweaver_hop_elements* elements,
    const CachedCost* records, int32_t record_count, double minimum,
    Result* results, Selection* selection,
    double* masses, double* inflations, double* proxies, double* collected,
    GeometryStats* stats, bool mesh = false, double delta = 0.0,
    double* output_arrivals = nullptr, double* output_departures = nullptr,
    const SearchConfig* search = nullptr) {
    auto* w=static_cast<Workspace*>(opaque);
    if(!correct_device(w) || count<0 || count>w->capacity || !stats || record_count<0) return 1;
    std::unique_lock<std::mutex> lock(w->mutex,std::try_to_lock);
    if(!lock.owns_lock())return 3;
    if(!count){*stats={};if(selection){*selection={};selection->index=-1;}return 0;}
    const int status=validate(w->n,mesh?1:count,policy,visits,stages,arrivals,departures,
        nullptr,selection?&selection->value:results,false);
    if(status)return status;
    if(mesh && !search) {
        if(!std::isfinite(delta) || delta<=0.0)return 1;
        for(int j=0;j<w->n;++j)
            if(!std::isfinite(arrivals[j]+delta)||!std::isfinite(arrivals[j]-delta)
                ||!std::isfinite(departures[j]+delta)||!std::isfinite(departures[j]-delta))return 1;
    }
    if(!elements || (record_count && !records) || size_t(count)*(w->n-1)>UINT32_MAX)return 1;
    for(int j=0;j<w->n-1;++j) {
        const auto& e=elements[j];
        if(!valid_orbit(e.departure)||!valid_orbit(e.arrival)
            ||!std::isfinite(e.gravitational_parameter)||e.gravitational_parameter<=0.0
            ||!std::isfinite(e.departure_allowance)||e.departure_allowance<0.0
            ||!std::isfinite(e.arrival_allowance)||e.arrival_allowance<0.0)return 1;
    }
    for(int i=0;i<record_count;++i) {
        const auto& r=records[i];
        if(!flag(r.cached)||!flag(r.value.measured)||r.value.reserved)return 4;
        if(r.leg<0||r.leg>=w->n-1||!std::isfinite(r.departure)||!std::isfinite(r.arrival)
            ||(i && !key_before(records[i-1],r))
            ||(r.value.measured && (!std::isfinite(r.value.measured_delta_v)
                ||!std::isfinite(r.value.measured_mass))))return 1;
    }
    const size_t maximum=size_t(w->capacity)*(w->n-1),legs=size_t(count)*(w->n-1),epochs=size_t(count)*w->n;
    if((!w->elements&&!allocate(w->elements,size_t(w->n-1)))
        ||(!w->hop_requests&&!allocate(w->hop_requests,maximum))
        ||(!w->hop_results&&!allocate(w->hop_results,maximum))
        ||(!w->geometry_stats&&!allocate(w->geometry_stats,1)))return 2;
    if(mesh && ((!w->base_arrivals&&!allocate(w->base_arrivals,size_t(w->n)))
        ||(!w->base_departures&&!allocate(w->base_departures,size_t(w->n)))))return 2;
    if(record_count>w->cached_capacity) {
        CachedCost* replacement=nullptr;
        if(!allocate(replacement,size_t(record_count)))return 2;
        if(w->cached_costs && cudaFree(w->cached_costs)!=cudaSuccess){cudaFree(replacement);return 2;}
        w->cached_costs=replacement;w->cached_capacity=record_count;
    }
    const auto failed=[&](){cudaStreamSynchronize(w->stream);return 2;};
    if(!upload(w->policy,policy,1,w->stream)||!upload(w->visits,visits,size_t(w->n),w->stream)
        ||!upload(w->stages,stages,size_t(w->n-1),w->stream)
        ||!upload(mesh?w->base_arrivals:w->arrivals,arrivals,mesh?size_t(w->n):epochs,w->stream)
        ||!upload(mesh?w->base_departures:w->departures,departures,mesh?size_t(w->n):epochs,w->stream)
        ||!upload(w->elements,elements,size_t(w->n-1),w->stream)
        ||(record_count&&!upload(w->cached_costs,records,size_t(record_count),w->stream))
        ||cudaMemsetAsync(w->geometry_stats,0,sizeof(GeometryStats),w->stream)!=cudaSuccess)return failed();
    if(search) return search_graph(w,count,record_count,*search,selection,masses,inflations,
        proxies,collected,output_arrivals,output_departures,stats);
    if(mesh) {
        generate_mesh<<<unsigned((epochs+127)/128),128,0,w->stream>>>(w->n,count,delta,
            w->base_arrivals,w->base_departures,w->arrivals,w->departures);
        if(cudaGetLastError()!=cudaSuccess)return failed();
    }
    clear_geometry_costs<<<unsigned((legs+127)/128),128,0,w->stream>>>(legs,w->costs);
    if(cudaGetLastError()!=cudaSuccess)return failed();
    const unsigned blocks=unsigned((size_t(count)+127)/128);
    const auto evaluate=[&](){evaluate_candidates<<<blocks,128,0,w->stream>>>(count,w->n,
        w->policy,w->visits,w->stages,w->arrivals,w->departures,w->costs,w->results,
        w->masses,w->inflations,w->proxies,w->collected);return cudaGetLastError();};
    if(evaluate()!=cudaSuccess)return failed();
    if(spacepdhcg_joint_geometry_launch(count,w->n,w->elements,w->arrivals,w->departures,
        w->results,w->cached_costs,record_count,w->costs,w->hop_requests,w->hop_results,
        w->geometry_stats,w->stream)!=cudaSuccess)return failed();
    if(evaluate()!=cudaSuccess)return failed();
    if(selection)select_candidate<<<1,128,0,w->stream>>>(count,w->n,minimum,w->results,
        w->selection,w->masses,w->inflations,w->proxies,w->collected);
    if(cudaGetLastError()!=cudaSuccess)return failed();
    if(mesh && selection)
        select_mesh_epochs<<<unsigned((size_t(w->n)+127)/128),128,0,w->stream>>>(w->n,
            w->selection,w->arrivals,w->departures,w->base_arrivals,w->base_departures);
    if(cudaGetLastError()!=cudaSuccess
        ||!(selection?download(selection,w->selection,1,w->stream):download(results,w->results,size_t(count),w->stream))
        ||!download(masses,w->masses,selection?size_t(w->n-1):legs,w->stream)
        ||!download(inflations,w->inflations,selection?size_t(w->n-1):legs,w->stream)
        ||!download(proxies,w->proxies,selection?size_t(w->n-1):legs,w->stream)
        ||!download(collected,w->collected,selection?size_t(w->n):epochs,w->stream)
        ||!download(stats,w->geometry_stats,1,w->stream)
        ||(mesh && (!download(output_arrivals,selection?w->base_arrivals:w->arrivals,
                selection?size_t(w->n):epochs,w->stream)
            ||!download(output_departures,selection?w->base_departures:w->departures,
                selection?size_t(w->n):epochs,w->stream))))return failed();
    return cudaStreamSynchronize(w->stream)==cudaSuccess?0:2;
}

extern "C" int spacepdhcg_gtoc12_joint_geometry_host(void* opaque, int32_t count,
    const Policy* policy, const Visit* visits, const Stage* stages,
    const double* arrivals, const double* departures,
    const spacepdhcg_orbitweaver_hop_elements* elements,
    const CachedCost* records, int32_t record_count, double minimum,
    Result* results, Selection* selection,
    double* masses, double* inflations, double* proxies, double* collected,
    GeometryStats* stats) {
    return geometry_host_impl(opaque,count,policy,visits,stages,arrivals,departures,
        elements,records,record_count,minimum,results,selection,masses,inflations,
        proxies,collected,stats);
}

extern "C" int spacepdhcg_gtoc12_joint_mesh_host(void* opaque, double delta,
    const Policy* policy, const Visit* visits, const Stage* stages,
    const double* arrivals, const double* departures,
    const spacepdhcg_orbitweaver_hop_elements* elements,
    const CachedCost* records, int32_t record_count, double minimum,
    Result* results, Selection* selection,
    double* masses, double* inflations, double* proxies, double* collected,
    double* output_arrivals, double* output_departures, GeometryStats* stats) {
    auto* w=static_cast<Workspace*>(opaque);
    if(!correct_device(w))return 1;
    const int64_t count=10*int64_t(w->n)-14;
    if(count>INT32_MAX)return 1;
    return geometry_host_impl(opaque,int32_t(count),policy,visits,stages,arrivals,departures,
        elements,records,record_count,minimum,results,selection,masses,inflations,
        proxies,collected,stats,true,delta,output_arrivals,output_departures);
}

extern "C" int spacepdhcg_gtoc12_joint_search_host(void* opaque, const double* days,
    int32_t levels, int32_t max_moves, double seconds, const Policy* policy,
    const Visit* visits, const Stage* stages, const double* arrivals, const double* departures,
    const spacepdhcg_orbitweaver_hop_elements* elements, const CachedCost* records,
    int32_t record_count, Selection* selection, double* masses, double* inflations,
    double* proxies, double* collected, double* output_arrivals, double* output_departures,
    GeometryStats* stats, SearchReport* report) {
    const SearchConfig config{days,levels,max_moves,seconds,report,std::chrono::steady_clock::now()};
    auto* w=static_cast<Workspace*>(opaque);
    if(!correct_device(w)||levels<0||max_moves<0||(levels&&!days)||!report||!selection
        ||std::isnan(seconds)||seconds<0||!arrivals||!departures
        ||int64_t(levels)*max_moves>INT32_MAX)return 1;
    long double travel=0;
    for(int j=0;j<levels;++j) {
        if(!std::isfinite(days[j])||days[j]<=0)return 1;
        travel+=static_cast<long double>(days[j])*max_moves;
    }
    // Bound every possible repeated finite move before launching the loop.
    for(int j=0;j<w->n;++j)
        if(fabsl(arrivals[j])+travel>std::numeric_limits<double>::max()
            ||fabsl(departures[j])+travel>std::numeric_limits<double>::max())return 1;
    const int64_t count=10*int64_t(w->n)-14;
    if(count>INT32_MAX)return 1;
    return geometry_host_impl(opaque,int32_t(count),policy,visits,stages,arrivals,departures,
        elements,records,record_count,0,nullptr,selection,masses,inflations,proxies,collected,
        stats,true,0,output_arrivals,output_departures,&config);
}
