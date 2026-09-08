#include "spacepdhcg/cuda/gtoc12_completion_c_api.h"

#include <cuda_runtime.h>
#include <cmath>
#include <cstddef>
#include <limits>
#include <mutex>
#include <new>

namespace {
using Policy = spacepdhcg_gtoc12_completion_policy;
using Candidate = spacepdhcg_gtoc12_completion_candidate;
using Deploy = spacepdhcg_gtoc12_completion_deploy;
using Leg = spacepdhcg_gtoc12_completion_leg;
using Result = spacepdhcg_gtoc12_completion_result;
using LegResult = spacepdhcg_gtoc12_completion_leg_result;
using Stats = spacepdhcg_gtoc12_completion_stats;
using CompactCandidate = spacepdhcg_gtoc12_completion_compact_candidate;
using CompactDeploy = spacepdhcg_gtoc12_completion_compact_deploy;
using CompactLeg = spacepdhcg_gtoc12_completion_compact_leg;
static_assert(sizeof(Policy) == 80 && sizeof(Candidate) == 24);
static_assert(sizeof(Deploy) == 24 && sizeof(Leg) == 128);
static_assert(sizeof(Result) == 48 && sizeof(LegResult) == 72 && sizeof(Stats) == 48);

struct Workspace {
    int device = -1, candidate_capacity = 0, deploy_capacity = 0, leg_capacity = 0;
    cudaStream_t stream{};
    cudaEvent_t events[4]{};
    Candidate* candidates{};
    Deploy* deploys{};
    Leg* legs{};
    Result* results{};
    LegResult* leg_results{};
    double* collected{};
    int32_t* seen{};
    CompactCandidate* compact_candidates{};
    CompactDeploy* compact_deploys{};
    CompactLeg* compact_legs{};
    std::mutex mutex;
};

// CPython 3.12 float sum: compensated accumulation in dictionary insertion order.
// Do not replace the final condition by an unconditional hi+lo: nonfinite input
// can leave a NaN compensation while the ordinary high part is infinite.
struct FloatSum {
    double hi = 0.0, lo = 0.0;
    __device__ void add(double x) {
        const double t = hi + x;
        if (fabs(hi) >= fabs(x)) lo += (hi - t) + x;
        else lo += (x - t) + hi;
        hi = t;
    }
    __device__ double value() const { return lo != 0.0 && isfinite(lo) ? hi + lo : hi; }
};

__device__ double maximum(double a, double b) {
    // NumPy maximum propagates NaN; CUDA fmax would hide it.
    if (isnan(a) || isnan(b)) return NAN;
    return a > b ? a : b;
}
__device__ double return_base(double tof) {
    constexpr double days[]{352, 420, 450, 480, 510, 540, 578, 630, 690, 810};
    constexpr double values[]{1.323, 1.383, 1.295, 1.195, 1.099, .977, .885, .930, .932, 1.014};
    if (isnan(tof)) return NAN;
    if (tof <= days[0]) return values[0];
    for (int i = 1; i < 10; ++i) {
        if (tof == days[i]) return values[i];
        if (tof < days[i])
            return values[i - 1] + (values[i] - values[i - 1]) /
                (days[i] - days[i - 1]) * (tof - days[i - 1]);
    }
    return values[9];
}
__device__ double inflation(const Leg& leg, double authority, double tof, double year) {
    if (leg.model == SPACEPDHCG_COMPLETION_FLAT ||
        leg.model == SPACEPDHCG_COMPLETION_CERTIFIED_FLAT) return leg.flat;
    const double dv = (leg.model == SPACEPDHCG_COMPLETION_FIT5 ||
        leg.model == SPACEPDHCG_COMPLETION_TABLE_RETURN) && !isfinite(leg.dv) ? 0.0 : leg.dv;
    const double ratio = dv / maximum(authority, 1e-12);
    if (leg.model == SPACEPDHCG_COMPLETION_RATIO) return leg.floor + leg.slope * ratio;
    if (leg.model == SPACEPDHCG_COMPLETION_FIT5) {
        const double features[]{1.0, ratio, tof / year, fabs(leg.delta_a_au) / .1,
            fabs(leg.delta_longitude_rad) / 3.141592653589793238462643383279502884};
        double value = 0.0;
        for (int k = 0; k < 5; ++k) value += features[k] * leg.fit[k];
        return maximum(value, leg.floor);
    }
    const double correction = 1.0 + .6 * (ratio - .33);
    const double clipped = isnan(correction) ? NAN :
        (correction < .85 ? .85 : (correction > 1.2 ? 1.2 : correction));
    return maximum(return_base(tof) * clipped, .85);
}

__global__ void finish_candidates(int count, Policy policy, const Candidate* candidates,
    const Deploy* deploys, const Leg* legs, Result* results, LegResult* leg_results,
    double* collected, int32_t* seen) {
    const int index = int(blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= count) return;
    const Candidate candidate = candidates[index];
    Result result{};
    result.failed_leg = result.failed_deploy = -1;
    result.final_mass = candidate.partial_mass;
    result.propellant = (policy.initial_mass - candidate.partial_mass) -
        policy.miner_mass * candidate.deploy_count;
    for (int k = 0; k < candidate.deploy_count; ++k) {
        collected[candidate.deploy_begin + k] = 0.0;
        seen[candidate.deploy_begin + k] = 0;
    }
    for (int k = 0; k < candidate.leg_count; ++k)
        leg_results[candidate.leg_begin + k] = {};
    for (int k = 0; k < candidate.deploy_count; ++k) {
        const Deploy d = deploys[candidate.deploy_begin + k];
        if (!d.has_collect) {
            result.failure = SPACEPDHCG_COMPLETION_UNCOLLECTED;
            result.failed_deploy = k;
            break;
        }
        if (d.collect_epoch - d.deploy_epoch < policy.minimum_stay - 1e-6) {
            result.failure = SPACEPDHCG_COMPLETION_STAY_TOO_SHORT;
            result.failed_deploy = k;
            break;
        }
    }
    FloatSum cargo;
    for (int k = 0; !result.failure && k < candidate.leg_count; ++k) {
        const Leg leg = legs[candidate.leg_begin + k];
        LegResult detail{};
        detail.mass_before = detail.departure_mass = detail.mass_after = result.final_mass;
        detail.tof = leg.arrival - leg.departure;
        if (leg.role == SPACEPDHCG_COMPLETION_CAMP) {
            detail.stage = SPACEPDHCG_COMPLETION_CAMP_PASSTHROUGH;
            detail.inflation = leg.flat;
            ++result.processed_legs;
            leg_results[candidate.leg_begin + k] = detail;
            continue;
        }
        if (leg.role == SPACEPDHCG_COMPLETION_COLLECT_HOP ||
            leg.role == SPACEPDHCG_COMPLETION_EARTH_RETURN) {
            const int slot = candidate.deploy_begin + leg.source_deploy;
            const Deploy d = deploys[slot];
            if (fabs(d.collect_epoch - leg.departure) < 1e-6) {
                detail.pickup = 1;
                const double stay = d.collect_epoch - d.deploy_epoch;
                if (!isfinite(stay) || stay < 0.0) {
                    detail.stage = SPACEPDHCG_COMPLETION_MINING_REJECTED;
                    result.failure = SPACEPDHCG_COMPLETION_INVALID_MINING_STAY;
                    result.failed_deploy = leg.source_deploy;
                } else {
                    detail.gained = (policy.mining_rate * stay) / policy.year_days;
                    collected[slot] = detail.gained;
                    if (!seen[slot]) { cargo.add(detail.gained); seen[slot] = 1; }
                    result.final_mass += detail.gained;
                    detail.departure_mass = detail.mass_after = result.final_mass;
                }
            }
        }
        if (!result.failure) {
            detail.authority = ((policy.thrust / result.final_mass * 1e-3) * detail.tof) * 86400.0;
            if (!(leg.dv <= leg.authority_ratio * detail.authority)) {
                detail.stage = SPACEPDHCG_COMPLETION_AUTHORITY_REJECTED;
                result.failure = SPACEPDHCG_COMPLETION_LEG_AUTHORITY;
            } else {
                detail.inflation = inflation(leg, detail.authority, detail.tof, policy.year_days);
                if (!isfinite(detail.inflation) || detail.inflation < 0.0) {
                    detail.stage = SPACEPDHCG_COMPLETION_INFLATION_REJECTED;
                    result.failure = SPACEPDHCG_COMPLETION_INVALID_INFLATION;
                } else {
                    detail.propellant = result.final_mass *
                        (1.0 - exp(-(leg.dv * detail.inflation) / policy.exhaust));
                    result.propellant += detail.propellant;
                    result.final_mass -= detail.propellant;
                    detail.mass_after = result.final_mass;
                    detail.stage = SPACEPDHCG_COMPLETION_COSTED;
                    ++result.processed_legs;
                }
            }
        }
        if (result.failure) result.failed_leg = k;
        leg_results[candidate.leg_begin + k] = detail;
    }
    result.collected = cargo.value();
    const double required = policy.dry_mass + result.collected;
    result.margin = result.final_mass - required;
    if (!result.failure && result.final_mass < required)
        result.failure = SPACEPDHCG_COMPLETION_MASS_BELOW_DRY_PLUS_COLLECTED;
    results[index] = result;
}

bool correct_device(Workspace* w) {
    int device = -1;
    return w && cudaGetDevice(&device) == cudaSuccess && device == w->device;
}
bool nonnegative(double x) { return std::isfinite(x) && x >= 0.0; }
int validate_policy(const Policy* policy) {
    if (!policy) return 1;
    const Policy& p = *policy;
    if (p.abi_version != 1 || p.sum_mode != 1 || p.reserved0 || p.reserved1) return 4;
    if (!nonnegative(p.initial_mass) || !nonnegative(p.dry_mass) || !nonnegative(p.miner_mass) ||
        !nonnegative(p.mining_rate) || !nonnegative(p.minimum_stay) ||
        !std::isfinite(p.thrust) || p.thrust <= 0 || !std::isfinite(p.exhaust) || p.exhaust <= 0 ||
        !std::isfinite(p.year_days) || p.year_days <= 0) return 1;
    return 0;
}
int validate(int count, int deploy_count, int leg_count, const Policy* policy,
    const Candidate* candidates, const Deploy* deploys, const Leg* legs, const Result* results) {
    if (!policy || !candidates || !results || (deploy_count && !deploys) || (leg_count && !legs)) return 1;
    const int policy_status = validate_policy(policy);
    if (policy_status) return policy_status;
    int64_t next_deploy = 0, next_leg = 0;
    for (int i = 0; i < count; ++i) {
        const Candidate& c = candidates[i];
        if (c.deploy_begin != next_deploy || c.leg_begin != next_leg ||
            c.deploy_count < 0 || c.leg_count < 0) return 1;
        next_deploy += c.deploy_count;
        next_leg += c.leg_count;
        if (next_deploy > deploy_count || next_leg > leg_count) return 1;
        for (int k = 0; k < c.deploy_count; ++k) {
            const Deploy& d = deploys[c.deploy_begin + k];
            if ((d.has_collect != 0 && d.has_collect != 1) || d.reserved) return 4;
        }
        for (int k = 0; k < c.leg_count; ++k) {
            const Leg& leg = legs[c.leg_begin + k];
            if (leg.reserved || leg.role < 0 || leg.role > SPACEPDHCG_COMPLETION_EARTH_RETURN ||
                leg.model < 0 || leg.model > SPACEPDHCG_COMPLETION_TABLE_RETURN) return 4;
            const bool pickup_role = leg.role == SPACEPDHCG_COMPLETION_COLLECT_HOP ||
                leg.role == SPACEPDHCG_COMPLETION_EARTH_RETURN;
            if ((pickup_role && (leg.source_deploy < 0 || leg.source_deploy >= c.deploy_count)) ||
                (!pickup_role && leg.source_deploy != -1)) return 1;
            if ((!pickup_role && leg.model != SPACEPDHCG_COMPLETION_FLAT) ||
                (leg.role == SPACEPDHCG_COMPLETION_COLLECT_HOP && leg.model > SPACEPDHCG_COMPLETION_FIT5) ||
                (leg.role == SPACEPDHCG_COMPLETION_EARTH_RETURN &&
                    (leg.model == SPACEPDHCG_COMPLETION_RATIO || leg.model == SPACEPDHCG_COMPLETION_FIT5))) return 4;
        }
    }
    return next_deploy == deploy_count && next_leg == leg_count ? 0 : 1;
}
template<class T> bool allocate(T*& output, int count) {
    return count == 0 || (size_t(count) <= std::numeric_limits<size_t>::max() / sizeof(T) &&
        cudaMalloc(&output, size_t(count) * sizeof(T)) == cudaSuccess);
}
template<class T> bool transfer(T* destination, const T* source, int count,
    cudaMemcpyKind kind, cudaStream_t stream) {
    return count == 0 || cudaMemcpyAsync(destination, source, size_t(count) * sizeof(T), kind, stream) == cudaSuccess;
}
bool release(Workspace* w) {
    bool ok = !w->stream || cudaStreamSynchronize(w->stream) == cudaSuccess;
    const auto free_buffer = [&ok](void* pointer) {
        if (pointer && cudaFree(pointer) != cudaSuccess) ok = false;
    };
    free_buffer(w->candidates); free_buffer(w->deploys); free_buffer(w->legs);
    free_buffer(w->results); free_buffer(w->leg_results); free_buffer(w->collected); free_buffer(w->seen);
    free_buffer(w->compact_candidates); free_buffer(w->compact_deploys); free_buffer(w->compact_legs);
    for (cudaEvent_t event : w->events) if (event && cudaEventDestroy(event) != cudaSuccess) ok = false;
    if (w->stream && cudaStreamDestroy(w->stream) != cudaSuccess) ok = false;
    return ok;
}
} // namespace

extern "C" int spacepdhcg_gtoc12_completion_create(
    int32_t device, int32_t candidates, int32_t deploys, int32_t legs, void** output) {
    if (!output) return 1;
    *output = nullptr;
    if (device < 0 || candidates < 1 || deploys < 0 || legs < 0) return 1;
    auto* w = new (std::nothrow) Workspace;
    if (!w) return 2;
    int previous = -1;
    if (cudaGetDevice(&previous) != cudaSuccess || cudaSetDevice(device) != cudaSuccess) {
        delete w; return 2;
    }
    w->device = device; w->candidate_capacity = candidates;
    w->deploy_capacity = deploys; w->leg_capacity = legs;
    bool ok = cudaStreamCreateWithFlags(&w->stream, cudaStreamNonBlocking) == cudaSuccess;
    for (auto& event : w->events) if (ok) ok = cudaEventCreate(&event) == cudaSuccess;
    ok = ok && allocate(w->candidates, candidates) && allocate(w->deploys, deploys) &&
        allocate(w->legs, legs) && allocate(w->results, candidates) &&
        allocate(w->leg_results, legs) && allocate(w->collected, deploys) && allocate(w->seen, deploys) &&
        allocate(w->compact_candidates, candidates) && allocate(w->compact_deploys, deploys) &&
        allocate(w->compact_legs, legs);
    if (!ok) release(w);
    if (previous != device && cudaSetDevice(previous) != cudaSuccess) {
        if (ok) { cudaSetDevice(device); release(w); }
        ok = false;
    }
    if (!ok) { delete w; return 2; }
    *output = w;
    return 0;
}

extern "C" int spacepdhcg_gtoc12_completion_evaluate_host(void* opaque,
    int32_t count, int32_t deploy_count, int32_t leg_count, const Policy* policy,
    const Candidate* candidates, const Deploy* deploys, const Leg* legs, Result* results,
    LegResult* leg_results, double* collected, Stats* stats) {
    auto* w = static_cast<Workspace*>(opaque);
    if (!correct_device(w) || count < 0 || count > w->candidate_capacity ||
        deploy_count < 0 || deploy_count > w->deploy_capacity ||
        leg_count < 0 || leg_count > w->leg_capacity) return 1;
    std::unique_lock<std::mutex> lock(w->mutex, std::try_to_lock);
    if (!lock.owns_lock()) return 3;
    if (count == 0) {
        if (deploy_count || leg_count) return 1;
        if (stats) *stats = {};
        return 0;
    }
    const int status = validate(count, deploy_count, leg_count, policy, candidates, deploys, legs, results);
    if (status) return status;
    const auto failed = [&]() { cudaStreamSynchronize(w->stream); return 2; };
    const auto record = [&](int i) { return !stats || cudaEventRecord(w->events[i], w->stream) == cudaSuccess; };
    if (!record(0) || !transfer(w->candidates, candidates, count, cudaMemcpyHostToDevice, w->stream) ||
        !transfer(w->deploys, deploys, deploy_count, cudaMemcpyHostToDevice, w->stream) ||
        !transfer(w->legs, legs, leg_count, cudaMemcpyHostToDevice, w->stream) || !record(1)) return failed();
    finish_candidates<<<unsigned((int64_t(count) + 127) / 128), 128, 0, w->stream>>>(count, *policy,
        w->candidates, w->deploys, w->legs, w->results, w->leg_results, w->collected, w->seen);
    if (cudaGetLastError() != cudaSuccess || !record(2) ||
        !transfer(results, w->results, count, cudaMemcpyDeviceToHost, w->stream) ||
        (leg_results && !transfer(leg_results, w->leg_results, leg_count, cudaMemcpyDeviceToHost, w->stream)) ||
        (collected && !transfer(collected, w->collected, deploy_count, cudaMemcpyDeviceToHost, w->stream)) ||
        !record(3) || cudaStreamSynchronize(w->stream) != cudaSuccess) return failed();
    if (stats) {
        float upload = 0, kernel = 0, download = 0;
        if (cudaEventElapsedTime(&upload, w->events[0], w->events[1]) != cudaSuccess ||
            cudaEventElapsedTime(&kernel, w->events[1], w->events[2]) != cudaSuccess ||
            cudaEventElapsedTime(&download, w->events[2], w->events[3]) != cudaSuccess) return 2;
        *stats = {double(upload), double(kernel), double(download), uint64_t(count),
            uint64_t(deploy_count), uint64_t(leg_count)};
    }
    return 0;
}

extern "C" int spacepdhcg_gtoc12_completion_destroy(void** opaque) {
    if (!opaque) return 1;
    auto* w = static_cast<Workspace*>(*opaque);
    if (!w) return 0;
    std::unique_lock<std::mutex> lock(w->mutex, std::try_to_lock);
    if (!lock.owns_lock()) return 3;
    int previous = -1;
    if (cudaGetDevice(&previous) != cudaSuccess || cudaSetDevice(w->device) != cudaSuccess) return 2;
    bool ok = release(w);
    if (previous != w->device && cudaSetDevice(previous) != cudaSuccess) ok = false;
    lock.unlock(); delete w; *opaque = nullptr;
    return ok ? 0 : 2;
}

#include "gtoc12_completion_model.cuh"
