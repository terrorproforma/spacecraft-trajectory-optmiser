#include "spacepdhcg/cuda/orbitweaver_gpu_c_api.h"
#include "../internal/gtoc12_collection_options.h"
#include "../internal/gtoc12_joint_geometry.h"

#include <cuda_runtime_api.h>
#include <cub/device/device_merge_sort.cuh>
#include <cub/device/device_select.cuh>

#include <atomic>
#include <algorithm>
#include <climits>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <new>
#include <limits>
#include <mutex>
#include <cstring>
#include <list>
#include <memory>
#include <vector>
#include "gtoc12_fleet.cuh"

namespace {

constexpr double pi = 3.141592653589793238462643383279502884;

struct Geometry {
    double r1;
    double r2;
    double cosine;
    double a;
    double angle;
    bool valid;
};

struct Evaluation {
    double residual;
    double y;
    bool valid;
};

struct Root {
    double parameter;
    std::uint32_t iterations;
};

__device__ double dot3(const double* left, const double* right) {
    return left[0] * right[0] + left[1] * right[1] + left[2] * right[2];
}

__device__ void stumpff(const double z, double& c, double& s) {
    if (z > 1.0e-8) {
        const auto root = sqrt(z);
        c = (1.0 - cos(root)) / z;
        s = (root - sin(root)) / (root * root * root);
    } else if (z < -1.0e-8) {
        const auto root = sqrt(-z);
        c = (cosh(root) - 1.0) / (-z);
        s = (sinh(root) - root) / (root * root * root);
    } else {
        const auto z2 = z * z;
        const auto z3 = z2 * z;
        c = 0.5 - z / 24.0 + z2 / 720.0 - z3 / 40320.0;
        s = 1.0 / 6.0 - z / 120.0 + z2 / 5040.0 - z3 / 362880.0;
    }
}

__device__ Geometry geometry(
    const spacepdhcg_orbitweaver_lambert_request& request,
    const bool long_way
) {
    const auto r1 = sqrt(dot3(request.departure_position, request.departure_position));
    const auto r2 = sqrt(dot3(request.arrival_position, request.arrival_position));
    if (!(r1 > 0.0) || !(r2 > 0.0)) {
        return {r1, r2, 0.0, 0.0, 0.0, false};
    }
    auto cosine = dot3(request.departure_position, request.arrival_position) / (r1 * r2);
    cosine = fmin(1.0, fmax(-1.0, cosine));
    // Cross-product geometry avoids cancellation in 1-cos(theta)^2 near pi.
    const auto* p = request.departure_position;
    const auto* q = request.arrival_position;
    const double cross[3] = {p[1]*q[2]-p[2]*q[1], p[2]*q[0]-p[0]*q[2], p[0]*q[1]-p[1]*q[0]};
    auto sine = fmin(1.0, sqrt(dot3(cross,cross))/(r1*r2));
    sine = long_way ? -sine : sine;
    const auto denominator = 1.0 - cosine;
    if (denominator <= 1.0e-14 || fabs(sine) <= 1.0e-14) {
        return {r1, r2, cosine, 0.0, 0.0, false};
    }
    const double sum[3] = {p[0]/r1+q[0]/r2, p[1]/r1+q[1]/r2, p[2]/r1+q[2]/r2};
    const auto a = (long_way ? -1.0 : 1.0) * sqrt(0.5*r1*r2) * sqrt(dot3(sum,sum));
    auto angle = acos(cosine);
    angle = long_way ? 2.0 * pi - angle : angle;
    return {r1, r2, cosine, a, angle, isfinite(a) && fabs(a) > 1.0e-14};
}

__device__ Evaluation evaluate(
    const double z,
    const Geometry value,
    const spacepdhcg_orbitweaver_lambert_request& request,
    const double* cached = nullptr
) {
    double c = 0.0;
    double s = 0.0;
    if (cached) { c = cached[0]; s = cached[1]; }
    else stumpff(z, c, s);
    if (!isfinite(c) || !isfinite(s) || c <= 0.0) {
        return {0.0, 0.0, false};
    }
    const auto y =
        value.r1 + value.r2 + value.a * (z * s - 1.0) / sqrt(c);
    if (!isfinite(y) || y < 0.0) {
        return {0.0, 0.0, false};
    }
    const auto x = sqrt(y / c);
    const auto computed =
        (x * x * x * s + value.a * sqrt(y))
        / sqrt(request.gravitational_parameter);
    return {computed - request.time_of_flight, y, isfinite(computed)};
}

__device__ bool valid(
    const spacepdhcg_orbitweaver_lambert_request& request
) {
    if (!(request.time_of_flight > 0.0)
        || !(request.gravitational_parameter > 0.0)
        || !(request.time_tolerance > 0.0)
        || request.maximum_iterations == 0U
        || (!request.include_short_way && !request.include_long_way)) {
        return false;
    }
    if (!isfinite(request.time_of_flight) || !isfinite(request.gravitational_parameter)
        || !isfinite(request.time_tolerance)) return false;
    for (int component = 0; component < 3; ++component) {
        if (!isfinite(request.departure_position[component])
            || !isfinite(request.arrival_position[component])) {
            return false;
        }
    }
    return true;
}

__device__ bool bisect(
    double lower,
    double upper,
    const Geometry value,
    const spacepdhcg_orbitweaver_lambert_request& request,
    Root& root
) {
    auto low = evaluate(lower, value, request);
    const auto high = evaluate(upper, value, request);
    if (!low.valid || !high.valid || low.residual * high.residual > 0.0) {
        return false;
    }
    for (std::uint32_t iteration = 0U;
         iteration < request.maximum_iterations;
         ++iteration) {
        const auto middle_parameter = 0.5 * (lower + upper);
        const auto middle = evaluate(middle_parameter, value, request);
        if (!middle.valid) {
            return false;
        }
        if (fabs(middle.residual) <= request.time_tolerance
            || fabs(upper - lower) <= 1.0e-13) {
            root = {middle_parameter, iteration + 1U};
            return true;
        }
        if (low.residual * middle.residual <= 0.0) {
            upper = middle_parameter;
        } else {
            lower = middle_parameter;
            low = middle;
        }
    }
    return false;
}

// Zero-revolution hop brackets have width <= 8*pi*pi/16. For budgets >=64,
// sixteen safeguarded interpolation steps leave enough bisections to reach
// the existing 1e-13 width gate, even if interpolation makes no progress.
__device__ bool interpolate_bracket(
    double lower,
    double upper,
    const Geometry value,
    const spacepdhcg_orbitweaver_lambert_request& request,
    Root& root
) {
    if (request.maximum_iterations < 64U) {
        return bisect(lower, upper, value, request, root);
    }
    auto left = evaluate(lower, value, request);
    auto right = evaluate(upper, value, request);
    if (!left.valid || !right.valid || left.residual * right.residual > 0.0) {
        return false;
    }
    // Illinois weights affect interpolation only; true residuals retain the bracket.
    double fl = left.residual, fr = right.residual;
    int last = 0;
    std::uint32_t used = 0U;
    for (; used < 16U; ++used) {
        double middle = lower + (upper - lower) * (fl / (fl - fr));
        if (!isfinite(middle) || middle <= lower || middle >= upper || (used & 3U) == 3U) {
            middle = 0.5 * (lower + upper);
        }
        const auto current = evaluate(middle, value, request);
        if (!current.valid) {
            ++used;
            break;
        }
        if (fabs(current.residual) <= request.time_tolerance
            || fabs(upper - lower) <= 1.0e-13) {
            root = {middle, used + 1U};
            return true;
        }
        if (left.residual * current.residual <= 0.0) {
            upper = middle;
            right = current;
            fr = current.residual;
            fl = last == 1 ? fl * 0.5 : left.residual;
            last = 1;
        } else {
            lower = middle;
            left = current;
            fl = current.residual;
            fr = last == -1 ? fr * 0.5 : right.residual;
            last = -1;
        }
    }
    auto fallback = request;
    fallback.maximum_iterations -= used;
    const bool found = bisect(lower, upper, value, fallback, root);
    if (found) root.iterations += used;
    return found;
}

__device__ std::uint32_t scan(
    const double lower,
    const double upper,
    const std::uint32_t samples,
    const Geometry value,
    const spacepdhcg_orbitweaver_lambert_request& request,
    Root roots[2],
    const std::uint32_t root_limit = 2U,
    const double* cached = nullptr,
    const bool fast_root = false
) {
    bool has_previous = false;
    double previous_parameter = 0.0;
    Evaluation previous{};
    std::uint32_t count = 0U;
    for (std::uint32_t sample = 0U; sample <= samples; ++sample) {
        const auto parameter =
            lower + static_cast<double>(sample) / static_cast<double>(samples)
                        * (upper - lower);
        const auto current = evaluate(parameter, value, request,
                                      cached ? cached + 2U*static_cast<size_t>(sample) : nullptr);
        if (!current.valid) {
            has_previous = false;
            continue;
        }
        Root root{};
        const auto exact = fabs(current.residual) <= request.time_tolerance;
        const auto bracket =
            has_previous && previous.residual * current.residual < 0.0;
        const auto found = exact
                               ? (root = Root{parameter, 0U}, true)
                               : bracket
                                     && (fast_root ? interpolate_bracket(previous_parameter,parameter,value,request,root) : bisect(
                                         previous_parameter,
                                         parameter,
                                         value,
                                         request,
                                         root
                                     ));
        if (found && count < 2U
            && (count == 0U
                || fabs(root.parameter - roots[count - 1U].parameter)
                       > 1.0e-9
                             * fmax(
                                 1.0,
                                 fmax(
                                     fabs(root.parameter),
                                     fabs(roots[count - 1U].parameter)
                                 )
                             ))) {
            roots[count++] = root;
            if (count == root_limit) return count;
        }
        previous_parameter = parameter;
        previous = current;
        has_previous = true;
    }
    return count;
}

__device__ bool solution(
    const spacepdhcg_orbitweaver_lambert_request& request,
    const Geometry value,
    const Root root,
    spacepdhcg_orbitweaver_lambert_result& result
) {
    const auto state = evaluate(root.parameter, value, request);
    const auto g =
        value.a * sqrt(state.y / request.gravitational_parameter);
    if (!state.valid || !isfinite(g) || fabs(g) <= 1.0e-14) {
        result.status = SPACEPDHCG_ORBITWEAVER_ARC_NUMERICAL_FAILURE;
        return false;
    }
    const auto f = 1.0 - state.y / value.r1;
    const auto g_dot = 1.0 - state.y / value.r2;
    for (int component = 0; component < 3; ++component) {
        result.departure_velocity[component] =
            (request.arrival_position[component]
             - f * request.departure_position[component])
            / g;
        result.arrival_velocity[component] =
            (g_dot * request.arrival_position[component]
             - request.departure_position[component])
            / g;
    }
    result.universal_parameter = root.parameter;
    result.transfer_angle_radians = value.angle;
    result.iterations = root.iterations;
    result.time_of_flight_residual = state.residual;
    result.status = SPACEPDHCG_ORBITWEAVER_ARC_FEASIBLE;
    return true;
}

__global__ void initialize_scan_grid(double* grid, const std::uint32_t samples) {
    const auto sample = static_cast<size_t>(blockIdx.x)*blockDim.x + threadIdx.x;
    if (sample > samples) return;
    const double lower = -4.0*pi*pi, upper = 4.0*pi*pi-1e-8;
    const double parameter = lower + static_cast<double>(sample)/samples*(upper-lower);
    stumpff(parameter, grid[2*sample], grid[2*sample+1]);
}

__global__ void kernel(
    const spacepdhcg_orbitweaver_lambert_request* requests,
    const std::size_t count,
    const std::uint32_t supported_revolutions,
    const std::uint32_t samples,
    spacepdhcg_orbitweaver_lambert_result* results,
    unsigned long long* counters,
    const int* cancelled,
    const double* cached
) {
    const auto input =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (input >= count) {
        return;
    }
    const auto stride =
        static_cast<std::size_t>(2U * (1U + 2U * supported_revolutions));
    auto* output = results + input * stride;
    const auto& request = requests[input];
    for (std::size_t slot = 0U; slot < stride; ++slot) {
        output[slot] = {};
        output[slot].deterministic_id = request.deterministic_id;
        output[slot].input_index = static_cast<std::uint32_t>(input);
        output[slot].family_index = static_cast<std::uint32_t>(slot);
        output[slot].status = SPACEPDHCG_ORBITWEAVER_ARC_UNSUPPORTED;
    }
    const auto invalid = !valid(request) || !geometry(request, false).valid;
    if ((cancelled && *cancelled != 0) || invalid) {
        const auto status = cancelled && *cancelled != 0
                                ? SPACEPDHCG_ORBITWEAVER_ARC_CANCELLED
                                : SPACEPDHCG_ORBITWEAVER_ARC_INVALID_INPUT;
        for (std::size_t slot = 0U; slot < stride; ++slot) {
            output[slot].status = status;
        }
        if (counters) atomicAdd(counters + 1U, static_cast<unsigned long long>(stride));
        return;
    }
    for (std::uint32_t direction = 0U; direction < 2U; ++direction) {
        const auto included =
            direction == 0U ? request.include_short_way : request.include_long_way;
        if (included == 0) {
            continue;
        }
        const auto value = geometry(request, direction != 0U);
        const auto base = static_cast<std::size_t>(direction)
                          * (1U + 2U * supported_revolutions);
        Root roots[2]{};
        auto& zero = output[base];
        zero.long_way = static_cast<int32_t>(direction);
        zero.status = SPACEPDHCG_ORBITWEAVER_ARC_NO_SOLUTION;
        if (scan(
                -4.0 * pi * pi,
                4.0 * pi * pi - 1.0e-8,
                samples,
                value,
                request,
                roots,
                1U,
                cached
            )
            > 0U) {
            static_cast<void>(solution(request, value, roots[0], zero));
        }
        for (std::uint32_t revolution = 1U;
             revolution <= supported_revolutions;
             ++revolution) {
            const auto first =
                base + 1U + 2U * static_cast<std::size_t>(revolution - 1U);
            auto& lower = output[first];
            auto& higher = output[first + 1U];
            lower.long_way = higher.long_way = static_cast<int32_t>(direction);
            lower.revolutions = higher.revolutions = revolution;
            lower.branch = SPACEPDHCG_ORBITWEAVER_LAMBERT_LOWER_PARAMETER;
            higher.branch = SPACEPDHCG_ORBITWEAVER_LAMBERT_HIGHER_PARAMETER;
            if (revolution > request.maximum_revolutions) {
                continue;
            }
            lower.status = higher.status = SPACEPDHCG_ORBITWEAVER_ARC_NO_SOLUTION;
            const auto lower_singularity =
                4.0 * static_cast<double>(revolution) * revolution * pi * pi;
            const auto next = revolution + 1U;
            const auto upper_singularity =
                4.0 * static_cast<double>(next) * next * pi * pi;
            const auto margin = 1.0e-9 * fmax(1.0, upper_singularity);
            const auto root_count = scan(
                lower_singularity + margin,
                upper_singularity - margin,
                samples,
                value,
                request,
                roots
            );
            if (root_count == 1U) {
                lower.branch = SPACEPDHCG_ORBITWEAVER_LAMBERT_UNIQUE;
                static_cast<void>(solution(request, value, roots[0], lower));
            } else if (root_count == 2U) {
                static_cast<void>(solution(request, value, roots[0], lower));
                static_cast<void>(solution(request, value, roots[1], higher));
            }
        }
    }
    unsigned long long feasible = 0U;
    for (std::size_t slot = 0U; slot < stride; ++slot) {
        feasible += output[slot].status == SPACEPDHCG_ORBITWEAVER_ARC_FEASIBLE
                        ? 1U
                        : 0U;
    }
    if (counters) {
        atomicAdd(counters, feasible);
        atomicAdd(counters + 1U, static_cast<unsigned long long>(stride) - feasible);
    }
}

__global__ void hop_kernel(
    const spacepdhcg_orbitweaver_hop_request* requests, const size_t count,
    const uint32_t samples, spacepdhcg_orbitweaver_hop_result* results,
    const double* cached, const bool fast_root
) {
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const auto& hop = requests[i];
    auto& output = results[i];
    output = {};
    output.departure_delta_v = output.arrival_delta_v = INFINITY;
    for (int k = 0; k < 3; ++k) {
        output.departure_velocity[k] = output.arrival_velocity[k] = NAN;
    }
    auto request = hop.lambert;
    request.include_short_way = request.include_long_way = 1;
    request.maximum_revolutions = 0;
    if (!valid(request) || !isfinite(hop.departure_allowance)
        || !isfinite(hop.arrival_allowance) || hop.departure_allowance < 0
        || hop.arrival_allowance < 0) return;
    for (int k = 0; k < 3; ++k) {
        if (!isfinite(hop.departure_body_velocity[k])
            || !isfinite(hop.arrival_body_velocity[k])) return;
    }
    for (int direction = 0; direction < 2; ++direction) {
        const auto value = geometry(request, direction != 0);
        if (!value.valid) continue;
        Root roots[2]{};
        if (!scan(-4*pi*pi, 4*pi*pi-1e-8, samples, value, request, roots, 1, cached, fast_root)) continue;
        spacepdhcg_orbitweaver_lambert_result candidate{};
        if (!solution(request, value, roots[0], candidate)) continue;
        double dep2 = 0, arr2 = 0;
        for (int k = 0; k < 3; ++k) {
            const double dep = candidate.departure_velocity[k] - hop.departure_body_velocity[k];
            const double arr = candidate.arrival_velocity[k] - hop.arrival_body_velocity[k];
            dep2 += dep * dep;
            arr2 += arr * arr;
        }
        const double dep = fmax(sqrt(dep2) - hop.departure_allowance, 0.0);
        const double arr = fmax(sqrt(arr2) - hop.arrival_allowance, 0.0);
        if (!isfinite(dep) || !isfinite(arr)) continue;
        if (dep + arr < output.departure_delta_v + output.arrival_delta_v) {
            output.feasible = 1;
            output.long_way = direction;
            output.departure_delta_v = dep;
            output.arrival_delta_v = arr;
            for (int k = 0; k < 3; ++k) {
                output.departure_velocity[k] = candidate.departure_velocity[k];
                output.arrival_velocity[k] = candidate.arrival_velocity[k];
            }
        }
    }
}

// Experimental small-batch hop operator. Preserve original scan order and bisection.
__device__ bool scan_warp(const double lower,const double upper,const uint32_t samples,
    const Geometry value,const spacepdhcg_orbitweaver_lambert_request& request,
    Root& root,const double* cached,const bool fast_root) {
    constexpr unsigned mask=0xffffffffU;
    const int lane=threadIdx.x&31;
    double prior_parameter=0,prior_residual=0;int prior_valid=0;
    for(uint64_t base=0;base<=samples;base+=32) {
        const uint64_t sample=base+lane;
        const double parameter=lower+static_cast<double>(sample)/static_cast<double>(samples)*(upper-lower);
        Evaluation current{};
        if(sample<=samples)current=evaluate(parameter,value,request,cached?cached+2*sample:nullptr);
        const double left_parameter=__shfl_up_sync(mask,parameter,1);
        const double left_residual=__shfl_up_sync(mask,current.residual,1);
        const int left_valid=__shfl_up_sync(mask,int(current.valid),1);
        const double previous_parameter=lane?left_parameter:prior_parameter;
        const double previous_residual=lane?left_residual:prior_residual;
        const int previous_valid=lane?left_valid:prior_valid;
        const bool exact=current.valid&&fabs(current.residual)<=request.time_tolerance;
        const bool bracket=current.valid&&previous_valid&&previous_residual*current.residual<0;
        unsigned candidates=__ballot_sync(mask,exact||bracket);
        while(candidates) {
            const int leader=__ffs(candidates)-1;
            Root local{};bool found=false;
            if(lane==leader) {
                if(exact){local={parameter,0U};found=true;}
                else found=fast_root ? interpolate_bracket(previous_parameter,parameter,value,request,local)
                    : bisect(previous_parameter,parameter,value,request,local);
            }
            const int accepted=__shfl_sync(mask,int(found),leader);
            const double selected=__shfl_sync(mask,local.parameter,leader);
            const unsigned iterations=__shfl_sync(mask,local.iterations,leader);
            if(accepted){root={selected,iterations};return true;}
            candidates&=candidates-1;
        }
        prior_parameter=__shfl_sync(mask,parameter,31);
        prior_residual=__shfl_sync(mask,current.residual,31);
        prior_valid=__shfl_sync(mask,int(current.valid),31);
    }
    return false;
}

__global__ void hop_warp_kernel(const spacepdhcg_orbitweaver_hop_request* requests,
    size_t count,uint32_t samples,spacepdhcg_orbitweaver_hop_result* results,const double* cached,bool fast_root) {
    const size_t i=(size_t(blockIdx.x)*blockDim.x+threadIdx.x)/32;
    const int lane=threadIdx.x&31;
    if(i>=count)return;
    const auto& hop=requests[i];auto& output=results[i];
    if(lane==0) {
        output={};output.departure_delta_v=output.arrival_delta_v=INFINITY;
        for(int k=0;k<3;++k)output.departure_velocity[k]=output.arrival_velocity[k]=NAN;
    }
    auto request=hop.lambert;
    request.include_short_way=request.include_long_way=1;request.maximum_revolutions=0;
    if(!valid(request)||!isfinite(hop.departure_allowance)||!isfinite(hop.arrival_allowance)
        ||hop.departure_allowance<0||hop.arrival_allowance<0)return;
    for(int k=0;k<3;++k)
        if(!isfinite(hop.departure_body_velocity[k])||!isfinite(hop.arrival_body_velocity[k]))return;
    for(int direction=0;direction<2;++direction) {
        const auto value=geometry(request,direction!=0);
        if(!value.valid)continue;
        Root root{};
        if(!scan_warp(-4*pi*pi,4*pi*pi-1e-8,samples,value,request,root,cached,fast_root))continue;
        if(lane==0) {
            spacepdhcg_orbitweaver_lambert_result candidate{};
            if(solution(request,value,root,candidate)) {
                double dep2=0,arr2=0;
                for(int k=0;k<3;++k) {
                    const double dep=candidate.departure_velocity[k]-hop.departure_body_velocity[k];
                    const double arr=candidate.arrival_velocity[k]-hop.arrival_body_velocity[k];
                    dep2+=dep*dep;arr2+=arr*arr;
                }
                const double dep=fmax(sqrt(dep2)-hop.departure_allowance,0.0);
                const double arr=fmax(sqrt(arr2)-hop.arrival_allowance,0.0);
                if(isfinite(dep)&&isfinite(arr)&&dep+arr<output.departure_delta_v+output.arrival_delta_v) {
                    output.feasible=1;output.long_way=direction;
                    output.departure_delta_v=dep;output.arrival_delta_v=arr;
                    for(int k=0;k<3;++k){output.departure_velocity[k]=candidate.departure_velocity[k];output.arrival_velocity[k]=candidate.arrival_velocity[k];}
                }
            }
        }
    }
}
// Separate warps solve the two directions, then share their candidate results.
__global__ void hop_parallel_kernel(const spacepdhcg_orbitweaver_hop_request* requests,
    size_t count,uint32_t samples,spacepdhcg_orbitweaver_hop_result* results,
    const double* cached,bool fast_root) {
    const size_t i=(size_t(blockIdx.x)*blockDim.x+threadIdx.x)/64;
    const int lane=threadIdx.x&63;
    // Full-warp groups include padding in the final block's shared-memory barrier.
    const spacepdhcg_orbitweaver_hop_request empty{};
    const auto& hop=i<count?requests[i]:empty;
    spacepdhcg_orbitweaver_hop_result local{};
    local.departure_delta_v=local.arrival_delta_v=INFINITY;
    for(int k=0;k<3;++k)local.departure_velocity[k]=local.arrival_velocity[k]=NAN;
    auto request=hop.lambert;
    request.include_short_way=request.include_long_way=1;request.maximum_revolutions=0;
    bool input_valid=valid(request)&&isfinite(hop.departure_allowance)
        &&isfinite(hop.arrival_allowance)&&hop.departure_allowance>=0&&hop.arrival_allowance>=0;
    for(int k=0;k<3;++k)input_valid=input_valid&&isfinite(hop.departure_body_velocity[k])
        &&isfinite(hop.arrival_body_velocity[k]);
    if(input_valid) {
        const auto value=geometry(request,lane>=32);
        Root root{};
        if(value.valid&&scan_warp(-4*pi*pi,4*pi*pi-1e-8,samples,value,request,root,cached,fast_root)) {
            if((lane&31)==0) {
                spacepdhcg_orbitweaver_lambert_result candidate{};
                if(solution(request,value,root,candidate)) {
                    double dep2=0,arr2=0;
                    for(int k=0;k<3;++k) {
                        const double dep=candidate.departure_velocity[k]-hop.departure_body_velocity[k];
                        const double arr=candidate.arrival_velocity[k]-hop.arrival_body_velocity[k];
                        dep2+=dep*dep;arr2+=arr*arr;
                    }
                    const double dep=fmax(sqrt(dep2)-hop.departure_allowance,0.0);
                    const double arr=fmax(sqrt(arr2)-hop.arrival_allowance,0.0);
                    if(isfinite(dep)&&isfinite(arr)&&isfinite(dep+arr)) {
                        local.feasible=1;local.long_way=lane>=32;
                        local.departure_delta_v=dep;local.arrival_delta_v=arr;
                        for(int k=0;k<3;++k) {
                            local.departure_velocity[k]=candidate.departure_velocity[k];
                            local.arrival_velocity[k]=candidate.arrival_velocity[k];
                        }
                    }
                }
            }
        }
    }
    // The 128-thread launch has four warps and two hops per block.
    __shared__ spacepdhcg_orbitweaver_hop_result branches[4];
    const int warp=threadIdx.x/32;
    if((lane&31)==0)branches[warp]=local;
    __syncthreads();
    if(lane==0&&i<count) {
        const auto& shorter=branches[warp];
        const auto& longer=branches[warp+1];
        // Strict comparison preserves short-way ties, including two invalid costs.
        results[i]=longer.departure_delta_v+longer.arrival_delta_v
            <shorter.departure_delta_v+shorter.arrival_delta_v?longer:shorter;
    }
}
void launch_hops(const spacepdhcg_orbitweaver_hop_request* requests,size_t count,
    uint32_t samples,spacepdhcg_orbitweaver_hop_result* results,const double* cached,cudaStream_t stream) {
    const auto* setting=std::getenv("SPACEPDHCG_TEST_GTOC12_FAST_LAMBERT_ROOT");
    const bool fast_root=!setting||setting[0]=='1';
    const auto* parallel=std::getenv("SPACEPDHCG_TEST_GTOC12_PARALLEL_DIRECTIONS");
    if(parallel&&parallel[0]=='1'&&count<=1024)
        hop_parallel_kernel<<<static_cast<unsigned>((count+1)/2),128,0,stream>>>(requests,count,samples,results,cached,fast_root);
    else if(count<=16384)hop_warp_kernel<<<static_cast<unsigned>((count+3)/4),128,0,stream>>>(requests,count,samples,results,cached,fast_root);
    else hop_kernel<<<static_cast<unsigned>((count+63)/64),64,0,stream>>>(requests,count,samples,results,cached,fast_root);
}

__device__ bool element_state(const spacepdhcg_orbitweaver_elements& b,
    double epoch, double mu, double* r, double* v) {
    double mean = fmod(b.mean + sqrt(mu / (b.a*b.a*b.a)) * ((epoch-b.epoch)*86400.0), 2*pi);
    if (mean < 0) mean += 2*pi;
    double eccentric = b.e > 0.8 ? pi : mean;
    bool converged = false;
    for (int k=0; k<64; ++k) {
        const double step = (eccentric-b.e*sin(eccentric)-mean)/(1-b.e*cos(eccentric));
        eccentric -= step;
        if (fabs(step)<1e-14) { converged=true; break; }
    }
    const double f=2*atan2(sqrt(1+b.e)*sin(eccentric/2),sqrt(1-b.e)*cos(eccentric/2));
    const double p=b.a*(1-b.e*b.e), radius=p/(1+b.e*cos(f)), scale=sqrt(mu/p);
    const double cn=cos(b.node),sn=sin(b.node),cp=cos(b.perihelion),sp=sin(b.perihelion);
    const double ci=cos(b.inclination),si=sin(b.inclination);
    const double pv[3]={cp*cn-sp*sn*ci,cp*sn+sp*cn*ci,sp*si};
    const double qv[3]={-sp*cn-cp*sn*ci,-sp*sn+cp*cn*ci,cp*si};
    for (int k=0;k<3;++k) {
        r[k]=radius*(pv[k]*cos(f)+qv[k]*sin(f));
        v[k]=scale*(-pv[k]*sin(f)+qv[k]*(b.e+cos(f)));
        if (!isfinite(r[k]) || !isfinite(v[k])) converged=false;
    }
    return converged;
}

__global__ void element_hops(spacepdhcg_orbitweaver_hop_elements elements,
    const double* times, size_t count, spacepdhcg_orbitweaver_hop_request* hops) {
    const size_t i=static_cast<size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if (i>=count) return;
    auto& h=hops[i];h={};
    auto& q=h.lambert;
    q.deterministic_id=i;q.gravitational_parameter=elements.gravitational_parameter;
    q.time_tolerance=1e-8;q.maximum_iterations=256;q.time_of_flight=times[2*i+1]*86400.0;
    h.departure_allowance=elements.departure_allowance;h.arrival_allowance=elements.arrival_allowance;
    const bool departure_ok=element_state(elements.departure,times[2*i],q.gravitational_parameter,
        q.departure_position,h.departure_body_velocity);
    const bool arrival_ok=element_state(elements.arrival,times[2*i]+times[2*i+1],q.gravitational_parameter,
        q.arrival_position,h.arrival_body_velocity);
    if (!departure_ok || !arrival_ok) q.time_of_flight=NAN;
}

bool valid_elements(const spacepdhcg_orbitweaver_elements& b) {
    return std::isfinite(b.epoch) && std::isfinite(b.a) && b.a>0
        && std::isfinite(b.e) && b.e>=0 && b.e<1 && std::isfinite(b.inclination)
        && std::isfinite(b.node) && std::isfinite(b.perihelion) && std::isfinite(b.mean);
}

__global__ void joint_hops_prepare(int count, int n,
    const spacepdhcg_orbitweaver_hop_elements* elements, const double* arrivals,
    const double* departures, const spacepdhcg_gtoc12_joint_result* preflight,
    const spacepdhcg_gtoc12_joint_cached_cost* records, int record_count,
    spacepdhcg_gtoc12_joint_cost* costs, spacepdhcg_orbitweaver_hop_request* requests,
    spacepdhcg_gtoc12_joint_geometry_stats* stats) {
    const size_t index=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if (index>=size_t(count)*(n-1)) return;
    const int row=int(index/(n-1)), leg=int(index%(n-1));
    auto& hop=requests[index];hop={};hop.lambert.time_of_flight=NAN;
    hop.lambert.deterministic_id=UINT64_MAX;
    if (preflight[row].failure!=SPACEPDHCG_JOINT_LEG_INFEASIBLE) {
        atomicAdd(reinterpret_cast<unsigned long long*>(&stats->rejected_hops),1ULL);return;
    }
    const double departure=departures[size_t(row)*n+leg];
    const double arrival=arrivals[size_t(row)*n+leg+1];
    int lo=0,hi=record_count;
    while (lo<hi) {
        const int mid=lo+(hi-lo)/2;const auto& r=records[mid];
        if (r.leg<leg || (r.leg==leg && (r.departure<departure
            || (r.departure==departure && r.arrival<arrival)))) lo=mid+1;
        else hi=mid;
    }
    if (lo<record_count) {
        const auto& r=records[lo];
        if (r.leg==leg && r.departure==departure && r.arrival==arrival) {
            costs[index]=r.value;
            if (r.cached) {
                atomicAdd(reinterpret_cast<unsigned long long*>(&stats->cached_hops),1ULL);return;
            }
        }
    }
    atomicAdd(reinterpret_cast<unsigned long long*>(&stats->computed_hops),1ULL);
    const auto& e=elements[leg];auto& q=hop.lambert;
    const double tof=arrival-departure;
    q.deterministic_id=index;q.gravitational_parameter=e.gravitational_parameter;
    q.time_tolerance=1e-8;q.maximum_iterations=256;q.time_of_flight=tof*86400.0;
    hop.departure_allowance=e.departure_allowance;hop.arrival_allowance=e.arrival_allowance;
    const bool d=element_state(e.departure,departure,e.gravitational_parameter,q.departure_position,hop.departure_body_velocity);
    // Preserve paired_hops' subtraction/addition order, including binary64 rounding.
    const bool a=element_state(e.arrival,departure+tof,e.gravitational_parameter,q.arrival_position,hop.arrival_body_velocity);
    if (!d || !a) q.time_of_flight=NAN;
}

__global__ void joint_hops_finish(size_t count,
    const spacepdhcg_orbitweaver_hop_request* requests,
    const spacepdhcg_orbitweaver_hop_result* results,
    spacepdhcg_gtoc12_joint_cost* costs) {
    const size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if (i<count && requests[i].lambert.deterministic_id!=UINT64_MAX) {
        const double total=results[i].departure_delta_v+results[i].arrival_delta_v;
        costs[i].lambert=results[i].feasible && isfinite(total)?total:INFINITY;
    }
}

__global__ void grid_times(const double* epochs,const double* tofs,size_t nt,
    size_t start,size_t count,double* times) {
    const size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i<count){times[2*i]=epochs[(start+i)/nt];times[2*i+1]=tofs[(start+i)%nt];}
}
__global__ void grid_costs(const spacepdhcg_orbitweaver_hop_result* results,
    size_t count,double* dv,uint8_t* ok) {
    const size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i>=count)return;
    const double cost=results[i].departure_delta_v+results[i].arrival_delta_v;
    const bool good=results[i].feasible&&isfinite(cost);
    dv[i]=good?cost:INFINITY;ok[i]=good;
}

__global__ void collection_grid_costs(const spacepdhcg_orbitweaver_hop_result* results,
    const double* times,size_t count,double end,float* output) {
    const size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i>=count)return;
    const double cost=results[i].departure_delta_v+results[i].arrival_delta_v;
    const bool good=results[i].feasible&&isfinite(cost)&&times[2*i]+times[2*i+1]<=end+1e-9;
    output[i]=good?static_cast<float>(cost):INFINITY;
}

spacepdhcg_cuda_status mapped(const cudaError_t status) {
    if (status == cudaSuccess) {
        return SPACEPDHCG_CUDA_SUCCESS;
    }
    return status == cudaErrorMemoryAllocation ? SPACEPDHCG_CUDA_OUT_OF_MEMORY
                                               : SPACEPDHCG_CUDA_RUNTIME_ERROR;
}

}  // namespace

cudaError_t spacepdhcg_joint_geometry_launch(int count, int n,
    const spacepdhcg_orbitweaver_hop_elements* elements, const double* arrivals,
    const double* departures, const spacepdhcg_gtoc12_joint_result* preflight,
    const spacepdhcg_gtoc12_joint_cached_cost* records, int record_count,
    spacepdhcg_gtoc12_joint_cost* costs, spacepdhcg_orbitweaver_hop_request* requests,
    spacepdhcg_orbitweaver_hop_result* results,
    spacepdhcg_gtoc12_joint_geometry_stats* stats, cudaStream_t stream) {
    const size_t size=size_t(count)*(n-1);
    joint_hops_prepare<<<unsigned((size+127)/128),128,0,stream>>>(count,n,elements,
        arrivals,departures,preflight,records,record_count,costs,requests,stats);
    auto status=cudaGetLastError();if(status!=cudaSuccess)return status;
    launch_hops(requests,size,256,results,nullptr,stream);
    status=cudaGetLastError();if(status!=cudaSuccess)return status;
    joint_hops_finish<<<unsigned((size+127)/128),128,0,stream>>>(size,requests,results,costs);
    return cudaGetLastError();
}

struct HopGridCacheEntry {
    spacepdhcg_orbitweaver_hop_elements elements{};
    std::vector<double> epochs, tofs;
    double* dv{};
    uint8_t* ok{};
    size_t bytes{};
    ~HopGridCacheEntry() { cudaFree(dv); cudaFree(ok); }
};

struct RankedHopOption {
    spacepdhcg_orbitweaver_hop_option value;
    size_t index;
};
struct HopOptionLess {
    __host__ __device__ bool operator()(const RankedHopOption& a,const RankedHopOption& b) const {
        if(a.value.delta_v!=b.value.delta_v)return a.value.delta_v<b.value.delta_v;
        if(a.value.departure!=b.value.departure)return a.value.departure>b.value.departure;
        return a.index<b.index;
    }
};
struct ValidHopOption {
    __host__ __device__ bool operator()(const RankedHopOption& a) const {
        return isfinite(a.value.delta_v);
    }
};
struct HopOptionScratch {
    RankedHopOption *rows{},*compact{};
    spacepdhcg_orbitweaver_hop_option* packed{};
    int* selected{};
    void* temporary{};
    size_t capacity{},temporary_bytes{};
    size_t bytes() const { return capacity*(2*sizeof(RankedHopOption)+sizeof(*packed))+sizeof(int)+temporary_bytes; }
    ~HopOptionScratch(){cudaFree(rows);cudaFree(compact);cudaFree(packed);cudaFree(selected);cudaFree(temporary);}
};

__global__ void build_hop_options(const spacepdhcg_orbitweaver_hop_result* hops,
    const double* times,size_t n,size_t offset,RankedHopOption* rows) {
    const size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i>=n)return;
    const double cost=hops[i].departure_delta_v+hops[i].arrival_delta_v;
    const bool valid=hops[i].feasible&&isfinite(cost);
    rows[i]={{valid?cost:INFINITY,valid?times[2*i]:0.,valid?times[2*i+1]:0.},offset+i};
}
__global__ void pack_hop_options(const RankedHopOption* rows,const int* selected,
    spacepdhcg_orbitweaver_hop_option* packed) {
    const size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i<size_t(*selected))packed[i]=rows[i].value;
}

#include "orbitweaver_beam.cuh"

struct GridAxesScratch {
    double* values{};
    size_t capacity{};
    ~GridAxesScratch(){cudaFree(values);}
};

struct spacepdhcg_orbitweaver_lambert_workspace {
    // Protect host API entry, including submission before its event is recorded.
    mutable std::mutex api_mutex;
    spacepdhcg_orbitweaver_lambert_config config{};
    spacepdhcg_orbitweaver_lambert_request* requests{nullptr};
    spacepdhcg_orbitweaver_lambert_result* results{nullptr};
    spacepdhcg_orbitweaver_hop_request* hops{nullptr};
    spacepdhcg_orbitweaver_hop_result* hop_results{nullptr};
    unsigned long long* counters{nullptr};
    int* cancelled{nullptr};
    double* scan_grid{nullptr};
    cudaStream_t stream{nullptr};
    cudaEvent_t completion{nullptr};
    bool owns_stream{false};
    std::atomic<bool> busy{false};
    std::uint64_t batches{0U};
    std::uint64_t request_count{0U};
    std::uint64_t result_count{0U};
    std::uint64_t feasible{0U};
    std::uint64_t failed{0U};
    std::uint64_t input_bytes{0U};
    std::uint64_t output_bytes{0U};
    // Immutable numerical tables only; masses, prices and sweep corrections
    // belong to the caller's retiming workspace and are never cached here.
    std::list<std::unique_ptr<HopGridCacheEntry>> grid_cache;
    size_t grid_cache_bytes{};
    uint64_t grid_cache_hits{}, grid_cache_misses{}, grid_cache_evictions{};
    std::unique_ptr<HopOptionScratch> option_scratch;
    std::unique_ptr<BeamScratch> beam_scratch;
    std::unique_ptr<GridAxesScratch> collection_axes;
};

static spacepdhcg_cuda_status hop_grid_locked(
    spacepdhcg_orbitweaver_lambert_workspace* w,
    const spacepdhcg_orbitweaver_hop_elements* elements,
    const double* epochs,size_t n,const double* tofs,size_t nt,double* dv,uint8_t* ok
) {
    auto status=cudaSuccess;
    auto* times=reinterpret_cast<double*>(w->requests);
    const size_t total=n*nt;
    for(size_t start=0;status==cudaSuccess&&start<total;) {
        const size_t count=total-start<w->config.maximum_batch_size?total-start:w->config.maximum_batch_size;
        grid_times<<<static_cast<unsigned>((count+127)/128),128,0,w->stream>>>(epochs,tofs,nt,start,count,times);
        status=cudaGetLastError();if(status!=cudaSuccess)break;
        element_hops<<<static_cast<unsigned>((count+127)/128),128,0,w->stream>>>(*elements,times,count,w->hops);
        status=cudaGetLastError();if(status!=cudaSuccess)break;
        launch_hops(w->hops,count,w->config.scan_samples_per_band,w->hop_results,w->scan_grid,w->stream);
        status=cudaGetLastError();if(status!=cudaSuccess)break;
        grid_costs<<<static_cast<unsigned>((count+127)/128),128,0,w->stream>>>(w->hop_results,count,dv+start,ok+start);
        status=cudaGetLastError();start+=count;
    }
    const auto done=cudaStreamSynchronize(w->stream);
    return mapped(status==cudaSuccess?done:status);
}

extern "C" {

size_t spacepdhcg_orbitweaver_lambert_result_stride(
    const uint32_t revolutions
) {
    return 2U * (1U + 2U * static_cast<size_t>(revolutions));
}

spacepdhcg_cuda_status spacepdhcg_orbitweaver_lambert_launch_device(
    const spacepdhcg_orbitweaver_lambert_request* requests,
    const size_t count, const uint32_t revolutions, const uint32_t samples,
    spacepdhcg_orbitweaver_lambert_result* results, const size_t capacity,
    const spacepdhcg_accelerator_stream stream
) {
    if (revolutions > (UINT32_MAX - 2U) / 4U || samples < 16U || samples == UINT32_MAX
        || count > UINT32_MAX || stream.device.type != SPACEPDHCG_DEVICE_CUDA)
        return SPACEPDHCG_CUDA_INVALID_ARGUMENT;
    const auto stride = spacepdhcg_orbitweaver_lambert_result_stride(revolutions);
    if (count > capacity / stride || (count && (!requests || !results))
        || count > std::numeric_limits<size_t>::max() / stride / sizeof(*results))
        return SPACEPDHCG_CUDA_INVALID_ARGUMENT;
    int device = -1;
    auto status = cudaGetDevice(&device);
    if (status != cudaSuccess) return mapped(status);
    if (device != stream.device.id) return SPACEPDHCG_CUDA_INVALID_ARGUMENT;
    if (!count) return SPACEPDHCG_CUDA_SUCCESS;
    const auto native = reinterpret_cast<cudaStream_t>(stream.native_handle);
    status = cudaMemsetAsync(results, 0, count * stride * sizeof(*results), native);
    if (status != cudaSuccess) return mapped(status);
    kernel<<<static_cast<unsigned>((count + 127U) / 128U), 128, 0, native>>>(
        requests, count, revolutions, samples, results, nullptr, nullptr, nullptr);
    return mapped(cudaGetLastError());
}

spacepdhcg_cuda_status spacepdhcg_orbitweaver_lambert_workspace_create(
    const spacepdhcg_orbitweaver_lambert_config* config,
    const spacepdhcg_accelerator_stream stream,
    spacepdhcg_orbitweaver_lambert_workspace** workspace
) {
    if (config == nullptr || workspace == nullptr || *workspace != nullptr
        || config->abi_version != SPACEPDHCG_ORBITWEAVER_GPU_ABI_VERSION
        || config->maximum_batch_size == 0U || config->scan_samples_per_band < 16U
        || config->scan_samples_per_band == UINT32_MAX
        || config->supported_maximum_revolutions > (UINT32_MAX - 2U) / 4U
        || config->maximum_batch_size > UINT32_MAX
        || config->maximum_batch_size > std::numeric_limits<size_t>::max()
            / spacepdhcg_orbitweaver_lambert_result_stride(config->supported_maximum_revolutions)
            / sizeof(spacepdhcg_orbitweaver_lambert_result)
        || stream.device.type != SPACEPDHCG_DEVICE_CUDA
        || stream.device.id != static_cast<int32_t>(config->device_id)) {
        return SPACEPDHCG_CUDA_INVALID_ARGUMENT;
    }
    auto* created = new (std::nothrow) spacepdhcg_orbitweaver_lambert_workspace{};
    if (created == nullptr) {
        return SPACEPDHCG_CUDA_OUT_OF_MEMORY;
    }
    created->config = *config;
    auto status = cudaSetDevice(static_cast<int>(config->device_id));
    if (status == cudaSuccess) {
        status = stream.native_handle == 0U
                     ? cudaStreamCreateWithFlags(&created->stream, cudaStreamNonBlocking)
                     : cudaSuccess;
        created->owns_stream = stream.native_handle == 0U;
        if (stream.native_handle != 0U) {
            created->stream = reinterpret_cast<cudaStream_t>(stream.native_handle);
        }
    }
    const auto stride =
        spacepdhcg_orbitweaver_lambert_result_stride(
            config->supported_maximum_revolutions
        );
    if (status == cudaSuccess) {
        status = cudaMalloc(
            reinterpret_cast<void**>(&created->requests),
            config->maximum_batch_size * sizeof(*created->requests)
        );
    }
    if (status == cudaSuccess) {
        status = cudaMalloc(
            reinterpret_cast<void**>(&created->results),
            config->maximum_batch_size * stride * sizeof(*created->results)
        );
    }
    if (status == cudaSuccess) {
        status = cudaMalloc(reinterpret_cast<void**>(&created->hops),
                            config->maximum_batch_size * sizeof(*created->hops));
    }
    if (status == cudaSuccess) {
        status = cudaMalloc(reinterpret_cast<void**>(&created->hop_results),
                            config->maximum_batch_size * sizeof(*created->hop_results));
    }
    if (status == cudaSuccess) {
        status = cudaMalloc(reinterpret_cast<void**>(&created->scan_grid),
                            (static_cast<size_t>(config->scan_samples_per_band)+1U)*2U*sizeof(double));
    }
    if (status == cudaSuccess) {
        initialize_scan_grid<<<static_cast<unsigned>((static_cast<size_t>(config->scan_samples_per_band)+128U)/128U),128,0,created->stream>>>(
            created->scan_grid, config->scan_samples_per_band);
        status = cudaGetLastError();
        // On devices without concurrent managed access (including WSL), host
        // initialization of the control page must not overlap this kernel.
        if (status == cudaSuccess) status = cudaStreamSynchronize(created->stream);
    }
    if (status == cudaSuccess) {
        status = cudaMallocManaged(
            reinterpret_cast<void**>(&created->counters),
            2U * sizeof(*created->counters)
        );
    }
    if (status == cudaSuccess) {
        status = cudaMallocManaged(
            reinterpret_cast<void**>(&created->cancelled),
            sizeof(*created->cancelled)
        );
    }
    if (status == cudaSuccess) {
        status = cudaEventCreateWithFlags(&created->completion, cudaEventDisableTiming);
    }
    if (status != cudaSuccess) {
        cudaFree(created->requests);
        cudaFree(created->results);
        cudaFree(created->counters);
        cudaFree(created->cancelled);
        cudaFree(created->scan_grid);
        cudaFree(created->hops);
        cudaFree(created->hop_results);
        if (created->owns_stream) {
            cudaStreamDestroy(created->stream);
        }
        delete created;
        return mapped(status);
    }
    *created->cancelled = 0;
    *workspace = created;
    return SPACEPDHCG_CUDA_SUCCESS;
}

spacepdhcg_cuda_status spacepdhcg_orbitweaver_lambert_evaluate_async(
    spacepdhcg_orbitweaver_lambert_workspace* workspace,
    const spacepdhcg_orbitweaver_lambert_request* requests,
    const size_t request_count,
    spacepdhcg_orbitweaver_lambert_result* results,
    const size_t result_capacity,
    const spacepdhcg_accelerator_stream stream
) {
    if (workspace == nullptr || requests == nullptr || results == nullptr
        || request_count == 0U || request_count > workspace->config.maximum_batch_size
        || stream.device.type != SPACEPDHCG_DEVICE_CUDA
        || stream.device.id != static_cast<int32_t>(workspace->config.device_id)) {
        return SPACEPDHCG_CUDA_INVALID_ARGUMENT;
    }
    std::unique_lock<std::mutex> api_lock(workspace->api_mutex, std::try_to_lock);
    if (!api_lock.owns_lock()) return SPACEPDHCG_CUDA_BUSY;
    bool expected = false;
    if (!workspace->busy.compare_exchange_strong(expected, true)) {
        return SPACEPDHCG_CUDA_BUSY;
    }
    const auto stride = spacepdhcg_orbitweaver_lambert_result_stride(
        workspace->config.supported_maximum_revolutions
    );
    const auto output_count = request_count * stride;
    if (result_capacity < output_count) {
        workspace->busy.store(false);
        return SPACEPDHCG_CUDA_INVALID_ARGUMENT;
    }
    *workspace->cancelled = 0;
    workspace->counters[0] = workspace->counters[1] = 0U;
    // Value-initialising each result in the kernel does not define ABI padding bytes.
    // Clear the full transfer region so device-to-host copies are initcheck-clean.
    auto status = cudaMemsetAsync(
        workspace->results,
        0,
        output_count * sizeof(*results),
        workspace->stream
    );
    if (status == cudaSuccess) {
        status = cudaMemcpyAsync(
        workspace->requests,
        requests,
        request_count * sizeof(*requests),
        cudaMemcpyHostToDevice,
        workspace->stream
        );
    }
    if (status == cudaSuccess) {
        constexpr std::uint32_t threads = 128U;
        const auto blocks =
            static_cast<std::uint32_t>((request_count + threads - 1U) / threads);
        kernel<<<blocks, threads, 0U, workspace->stream>>>(
            workspace->requests,
            request_count,
            workspace->config.supported_maximum_revolutions,
            workspace->config.scan_samples_per_band,
            workspace->results,
            workspace->counters,
            workspace->cancelled,
            workspace->scan_grid
        );
        status = cudaGetLastError();
    }
    if (status == cudaSuccess) {
        status = cudaMemcpyAsync(
            results,
            workspace->results,
            output_count * sizeof(*results),
            cudaMemcpyDeviceToHost,
            workspace->stream
        );
    }
    if (status == cudaSuccess) {
        status = cudaEventRecord(workspace->completion, workspace->stream);
    }
    if (status != cudaSuccess) {
        // Retire queued buffer accesses before making the workspace reusable.
        cudaStreamSynchronize(workspace->stream);
        workspace->busy.store(false);
        return mapped(status);
    }
    ++workspace->batches;
    workspace->request_count += request_count;
    workspace->result_count += output_count;
    workspace->input_bytes += request_count * sizeof(*requests);
    workspace->output_bytes += output_count * sizeof(*results);
    return SPACEPDHCG_CUDA_SUCCESS;
}

spacepdhcg_cuda_status spacepdhcg_orbitweaver_lambert_workspace_telemetry(
    const spacepdhcg_orbitweaver_lambert_workspace* workspace,
    spacepdhcg_orbitweaver_batch_telemetry* telemetry
) {
    if (workspace == nullptr || telemetry == nullptr
        || telemetry->abi_version != SPACEPDHCG_ORBITWEAVER_GPU_ABI_VERSION) {
        return SPACEPDHCG_CUDA_INVALID_ARGUMENT;
    }
    std::unique_lock<std::mutex> api_lock(workspace->api_mutex, std::try_to_lock);
    if (!api_lock.owns_lock()) return SPACEPDHCG_CUDA_BUSY;
    auto* mutable_workspace =
        const_cast<spacepdhcg_orbitweaver_lambert_workspace*>(workspace);
    if (workspace->busy.load()) {
        const auto status = cudaEventQuery(workspace->completion);
        if (status == cudaErrorNotReady) {
            return SPACEPDHCG_CUDA_BUSY;
        }
        if (status != cudaSuccess) {
            return mapped(status);
        }
        mutable_workspace->feasible += workspace->counters[0];
        mutable_workspace->failed += workspace->counters[1];
        mutable_workspace->busy.store(false);
    }
    const auto stride = spacepdhcg_orbitweaver_lambert_result_stride(
        workspace->config.supported_maximum_revolutions
    );
    telemetry->batches_submitted = workspace->batches;
    telemetry->requests_submitted = workspace->request_count;
    telemetry->results_emitted = workspace->result_count;
    telemetry->feasible_results = workspace->feasible;
    telemetry->failed_results = workspace->failed;
    telemetry->input_bytes = workspace->input_bytes;
    telemetry->output_bytes = workspace->output_bytes;
    telemetry->workspace_bytes =
        workspace->config.maximum_batch_size
            * (sizeof(spacepdhcg_orbitweaver_lambert_request)
               + stride * sizeof(spacepdhcg_orbitweaver_lambert_result))
        + 2U * sizeof(unsigned long long) + sizeof(int)
        + (static_cast<size_t>(workspace->config.scan_samples_per_band)+1U)*2U*sizeof(double);
    telemetry->workspace_bytes += workspace->config.maximum_batch_size
        * (sizeof(*workspace->hops) + sizeof(*workspace->hop_results));
    telemetry->workspace_bytes += workspace->grid_cache_bytes;
    if(workspace->option_scratch)telemetry->workspace_bytes+=workspace->option_scratch->bytes();
    if(workspace->beam_scratch)telemetry->workspace_bytes+=workspace->beam_scratch->bytes();
    if(workspace->collection_axes)telemetry->workspace_bytes+=workspace->collection_axes->capacity*sizeof(double);
    telemetry->maximum_batch_size = workspace->config.maximum_batch_size;
    telemetry->device_id = static_cast<int32_t>(workspace->config.device_id);
    return SPACEPDHCG_CUDA_SUCCESS;
}

spacepdhcg_cuda_status spacepdhcg_orbitweaver_lambert_workspace_cancel(
    spacepdhcg_orbitweaver_lambert_workspace* workspace
) {
    if (workspace == nullptr) {
        return SPACEPDHCG_CUDA_INVALID_ARGUMENT;
    }
    std::unique_lock<std::mutex> api_lock(workspace->api_mutex, std::try_to_lock);
    if (!api_lock.owns_lock()) return SPACEPDHCG_CUDA_BUSY;
    *workspace->cancelled = 1;
    return SPACEPDHCG_CUDA_SUCCESS;
}

spacepdhcg_cuda_status spacepdhcg_orbitweaver_lambert_screening_host(
    spacepdhcg_orbitweaver_lambert_workspace* w,
    const spacepdhcg_orbitweaver_lambert_request* requests, const size_t count,
    spacepdhcg_orbitweaver_lambert_result* results, const size_t capacity
) {
    if (!w || !requests || !results || !count || count > w->config.maximum_batch_size)
        return SPACEPDHCG_CUDA_INVALID_ARGUMENT;
    const auto stride = spacepdhcg_orbitweaver_lambert_result_stride(w->config.supported_maximum_revolutions);
    if (count > capacity/stride) return SPACEPDHCG_CUDA_INVALID_ARGUMENT;
    int device = -1;
    auto status = cudaGetDevice(&device);
    if (status != cudaSuccess) return mapped(status);
    if (device != static_cast<int>(w->config.device_id)) return SPACEPDHCG_CUDA_INVALID_ARGUMENT;
    std::unique_lock<std::mutex> api_lock(w->api_mutex, std::try_to_lock);
    if (!api_lock.owns_lock()) return SPACEPDHCG_CUDA_BUSY;
    bool expected = false;
    if (!w->busy.compare_exchange_strong(expected,true)) return SPACEPDHCG_CUDA_BUSY;
    status = cudaMemcpyAsync(w->requests,requests,count*sizeof(*requests),cudaMemcpyHostToDevice,w->stream);
    if (status == cudaSuccess) status = cudaMemsetAsync(w->results,0,count*stride*sizeof(*results),w->stream);
    if (status == cudaSuccess) {
        kernel<<<static_cast<unsigned>((count+127U)/128U),128,0,w->stream>>>(
            w->requests,count,w->config.supported_maximum_revolutions,w->config.scan_samples_per_band,
            w->results,nullptr,nullptr,w->scan_grid);
        status = cudaGetLastError();
    }
    if (status == cudaSuccess) status = cudaMemcpyAsync(results,w->results,count*stride*sizeof(*results),cudaMemcpyDeviceToHost,w->stream);
    const auto completed = cudaStreamSynchronize(w->stream);
    if (status == cudaSuccess) status = completed;
    if (status == cudaSuccess) {
        ++w->batches; w->request_count += count; w->result_count += count*stride;
        w->input_bytes += count*sizeof(*requests);
        w->output_bytes += count*stride*sizeof(*results);
        for (size_t i=0;i<count*stride;++i) {
            if (results[i].status == SPACEPDHCG_ORBITWEAVER_ARC_FEASIBLE) ++w->feasible;
            else ++w->failed;
        }
    }
    w->busy.store(false);
    return mapped(status);
}

spacepdhcg_cuda_status spacepdhcg_orbitweaver_hop_launch_device(
    const spacepdhcg_orbitweaver_hop_request* requests, const size_t count,
    const uint32_t samples, spacepdhcg_orbitweaver_hop_result* results,
    const size_t capacity, const spacepdhcg_accelerator_stream stream
) {
    if (count > UINT32_MAX || count > capacity || samples < 16 || samples == UINT32_MAX
        || (count && (!requests || !results)) || stream.device.type != SPACEPDHCG_DEVICE_CUDA)
        return SPACEPDHCG_CUDA_INVALID_ARGUMENT;
    int device = -1;
    const auto status = cudaGetDevice(&device);
    if (status != cudaSuccess) return mapped(status);
    if (device != stream.device.id) return SPACEPDHCG_CUDA_INVALID_ARGUMENT;
    if (!count) return SPACEPDHCG_CUDA_SUCCESS;
    launch_hops(requests,count,samples,results,nullptr,reinterpret_cast<cudaStream_t>(stream.native_handle));
    return mapped(cudaGetLastError());
}

spacepdhcg_cuda_status spacepdhcg_orbitweaver_hop_screening_host(
    spacepdhcg_orbitweaver_lambert_workspace* w,
    const spacepdhcg_orbitweaver_hop_request* requests, const size_t count,
    spacepdhcg_orbitweaver_hop_result* results, const size_t capacity
) {
    if (!w || !requests || !results || !count || count > capacity
        || count > w->config.maximum_batch_size) return SPACEPDHCG_CUDA_INVALID_ARGUMENT;
    std::unique_lock<std::mutex> api_lock(w->api_mutex,std::try_to_lock);
    if (!api_lock.owns_lock() || w->busy.load()) return SPACEPDHCG_CUDA_BUSY;
    int device = -1;
    auto status = cudaGetDevice(&device);
    if (status != cudaSuccess) return mapped(status);
    if (device != static_cast<int>(w->config.device_id)) return SPACEPDHCG_CUDA_INVALID_ARGUMENT;
    status = cudaMemcpyAsync(w->hops,requests,count*sizeof(*requests),cudaMemcpyHostToDevice,w->stream);
    if (status == cudaSuccess) {
        launch_hops(w->hops,count,w->config.scan_samples_per_band,w->hop_results,w->scan_grid,w->stream);
        status = cudaGetLastError();
    }
    if (status == cudaSuccess) status = cudaMemcpyAsync(results,w->hop_results,count*sizeof(*results),cudaMemcpyDeviceToHost,w->stream);
    const auto completed = cudaStreamSynchronize(w->stream);
    if (status == cudaSuccess) status = completed;
    if (status == cudaSuccess) {
        ++w->batches; w->request_count += count; w->result_count += count;
        w->input_bytes += count*sizeof(*requests); w->output_bytes += count*sizeof(*results);
        for (size_t i=0;i<count;++i) {
            if (results[i].feasible) ++w->feasible;
            else ++w->failed;
        }
    }
    return mapped(status);
}

spacepdhcg_cuda_status spacepdhcg_orbitweaver_hop_elements_host(
    spacepdhcg_orbitweaver_lambert_workspace* w,
    const spacepdhcg_orbitweaver_hop_elements* elements,
    const double* times, const size_t count,
    spacepdhcg_orbitweaver_hop_result* results, const size_t capacity
) {
    if (!w || !elements || !times || !results || !count || count>capacity
        || count>w->config.maximum_batch_size || !valid_elements(elements->departure)
        || !valid_elements(elements->arrival) || !std::isfinite(elements->gravitational_parameter)
        || elements->gravitational_parameter<=0 || !std::isfinite(elements->departure_allowance)
        || elements->departure_allowance<0 || !std::isfinite(elements->arrival_allowance)
        || elements->arrival_allowance<0) return SPACEPDHCG_CUDA_INVALID_ARGUMENT;
    std::unique_lock<std::mutex> lock(w->api_mutex,std::try_to_lock);
    if (!lock.owns_lock() || w->busy.load()) return SPACEPDHCG_CUDA_BUSY;
    int device=-1;auto status=cudaGetDevice(&device);
    if (status!=cudaSuccess) return mapped(status);
    if (device!=static_cast<int>(w->config.device_id)) return SPACEPDHCG_CUDA_INVALID_ARGUMENT;
    // The ordinary Lambert request buffer is idle while the hop path owns the
    // workspace. Reuse its allocation for compact epoch/duration pairs.
    static_assert(sizeof(spacepdhcg_orbitweaver_lambert_request)>=2*sizeof(double));
    auto* device_times=reinterpret_cast<double*>(w->requests);
    status=cudaMemcpyAsync(device_times,times,count*2*sizeof(double),cudaMemcpyHostToDevice,w->stream);
    if (status==cudaSuccess) {
        element_hops<<<static_cast<unsigned>((count+127)/128),128,0,w->stream>>>(
            *elements,device_times,count,w->hops);
        status=cudaGetLastError();
    }
    if (status==cudaSuccess) {
        launch_hops(w->hops,count,w->config.scan_samples_per_band,w->hop_results,w->scan_grid,w->stream);
        status=cudaGetLastError();
    }
    if (status==cudaSuccess) status=cudaMemcpyAsync(results,w->hop_results,count*sizeof(*results),cudaMemcpyDeviceToHost,w->stream);
    const auto completed=cudaStreamSynchronize(w->stream);
    if (status==cudaSuccess) status=completed;
    if (status==cudaSuccess) {
        ++w->batches;w->request_count+=count;w->result_count+=count;
        w->input_bytes+=count*2*sizeof(double)+sizeof(*elements);w->output_bytes+=count*sizeof(*results);
        for (size_t i=0;i<count;++i) {
            if (results[i].feasible) ++w->feasible; else ++w->failed;
        }
    }
    return mapped(status);
}

spacepdhcg_cuda_status spacepdhcg_orbitweaver_earth_beam_host(
    spacepdhcg_orbitweaver_lambert_workspace* w,const spacepdhcg_orbitweaver_beam_config* config,
    const spacepdhcg_orbitweaver_beam_target* targets,size_t na,const double* epochs,size_t ne,
    const double* tofs,size_t nt,size_t block,size_t limit,
    spacepdhcg_orbitweaver_beam_option* options,size_t capacity,size_t* selected) {
    return earth_beam_host(w,config,targets,na,epochs,ne,tofs,nt,block,limit,options,capacity,selected);
}

static spacepdhcg_cuda_status hop_options_impl(
    spacepdhcg_orbitweaver_lambert_workspace* w,
    const spacepdhcg_orbitweaver_hop_elements* elements,
    const double* times,size_t count,int32_t sort_returns,
    spacepdhcg_orbitweaver_hop_option* options,size_t capacity,size_t* selected,
    spacepdhcg_gtoc12_collection_options** resident
) {
    if(!w||!elements||!selected||!times||(!options&&!resident)||(resident&&*resident)||!count||count>capacity
        ||count>INT_MAX||count>SIZE_MAX/(2*sizeof(RankedHopOption)+sizeof(*options))
        ||(sort_returns!=0&&sort_returns!=1)||!valid_elements(elements->departure)
        ||!valid_elements(elements->arrival)||!std::isfinite(elements->gravitational_parameter)
        ||elements->gravitational_parameter<=0||!std::isfinite(elements->departure_allowance)
        ||elements->departure_allowance<0||!std::isfinite(elements->arrival_allowance)
        ||elements->arrival_allowance<0)return SPACEPDHCG_CUDA_INVALID_ARGUMENT;
    *selected=0;
    std::unique_lock<std::mutex> lock(w->api_mutex,std::try_to_lock);
    if(!lock.owns_lock()||w->busy.load())return SPACEPDHCG_CUDA_BUSY;
    int device=-1;auto status=cudaGetDevice(&device);
    if(status!=cudaSuccess)return mapped(status);
    if(device!=static_cast<int>(w->config.device_id))return SPACEPDHCG_CUDA_INVALID_ARGUMENT;
    if(!w->option_scratch||w->option_scratch->capacity<count) {
        std::unique_ptr<HopOptionScratch> scratch(new(std::nothrow) HopOptionScratch);
        if(!scratch)return SPACEPDHCG_CUDA_OUT_OF_MEMORY;
        scratch->capacity=count;
        status=cudaMalloc(&scratch->rows,count*sizeof(*scratch->rows));
        if(status==cudaSuccess)status=cudaMalloc(&scratch->compact,count*sizeof(*scratch->compact));
        if(status==cudaSuccess)status=cudaMalloc(&scratch->packed,count*sizeof(*scratch->packed));
        if(status==cudaSuccess)status=cudaMalloc(&scratch->selected,sizeof(int));
        size_t sorting=0,filtering=0;
        if(status==cudaSuccess)status=cub::DeviceMergeSort::SortKeys(nullptr,sorting,scratch->rows,int(count),HopOptionLess{},w->stream);
        if(status==cudaSuccess)status=cub::DeviceSelect::If(nullptr,filtering,scratch->rows,scratch->compact,scratch->selected,int(count),ValidHopOption{},w->stream);
        scratch->temporary_bytes=std::max(sorting,filtering);
        if(status==cudaSuccess)status=cudaMalloc(&scratch->temporary,scratch->temporary_bytes);
        if(status!=cudaSuccess)return mapped(status);
        w->option_scratch=std::move(scratch);
    }
    auto& scratch=*w->option_scratch;
    auto* device_times=reinterpret_cast<double*>(w->requests);
    for(size_t start=0;status==cudaSuccess&&start<count;) {
        const size_t n=std::min(count-start,w->config.maximum_batch_size);
        status=cudaMemcpyAsync(device_times,times+2*start,n*2*sizeof(double),cudaMemcpyHostToDevice,w->stream);
        if(status==cudaSuccess){element_hops<<<(n+127)/128,128,0,w->stream>>>(*elements,device_times,n,w->hops);status=cudaGetLastError();}
        if(status==cudaSuccess){launch_hops(w->hops,n,w->config.scan_samples_per_band,w->hop_results,w->scan_grid,w->stream);status=cudaGetLastError();}
        if(status==cudaSuccess){build_hop_options<<<(n+127)/128,128,0,w->stream>>>(w->hop_results,device_times,n,start,scratch.rows+start);status=cudaGetLastError();}
        start+=n;
    }
    auto bytes=scratch.temporary_bytes;
    if(status==cudaSuccess&&sort_returns)status=cub::DeviceMergeSort::SortKeys(scratch.temporary,bytes,scratch.rows,int(count),HopOptionLess{},w->stream);
    bytes=scratch.temporary_bytes;
    if(status==cudaSuccess)status=cub::DeviceSelect::If(scratch.temporary,bytes,scratch.rows,scratch.compact,scratch.selected,int(count),ValidHopOption{},w->stream);
    if(status==cudaSuccess){pack_hop_options<<<(count+127)/128,128,0,w->stream>>>(scratch.compact,scratch.selected,scratch.packed);status=cudaGetLastError();}
    int valid_count=0;
    if(status==cudaSuccess)status=cudaMemcpyAsync(&valid_count,scratch.selected,sizeof(int),cudaMemcpyDeviceToHost,w->stream);
    auto done=cudaStreamSynchronize(w->stream);
    if(status==cudaSuccess)status=done;
    if(status!=cudaSuccess)return mapped(status);
    if(valid_count<0||size_t(valid_count)>count)return SPACEPDHCG_CUDA_RUNTIME_ERROR;
    if(resident) {
        static_assert(sizeof(spacepdhcg_orbitweaver_hop_option)==sizeof(spacepdhcg_gtoc12_collection_option));
        const auto copied=gtoc12_collection_options_copy_device(
            reinterpret_cast<const spacepdhcg_gtoc12_collection_option*>(scratch.packed),valid_count,w->stream,resident);
        if(copied!=SPACEPDHCG_CUDA_SUCCESS)return copied;
    } else if(valid_count)status=cudaMemcpyAsync(options,scratch.packed,size_t(valid_count)*sizeof(*options),cudaMemcpyDeviceToHost,w->stream);
    done=resident?cudaSuccess:cudaStreamSynchronize(w->stream);
    if(status==cudaSuccess)status=done;
    if(status==cudaSuccess){
        *selected=size_t(valid_count);
        const size_t batches=(count-1)/w->config.maximum_batch_size+1;
        w->batches+=batches;w->request_count+=count;w->result_count+=size_t(valid_count);
        w->feasible+=valid_count;w->failed+=count-size_t(valid_count);
        w->input_bytes+=count*2*sizeof(double)+batches*sizeof(*elements);
        w->output_bytes+=(resident?0:size_t(valid_count)*sizeof(*options))+sizeof(int);
    }
    return mapped(status);
}

spacepdhcg_cuda_status spacepdhcg_orbitweaver_hop_options_host(
    spacepdhcg_orbitweaver_lambert_workspace* w,const spacepdhcg_orbitweaver_hop_elements* elements,
    const double* times,size_t count,int32_t sort_returns,spacepdhcg_orbitweaver_hop_option* options,
    size_t capacity,size_t* selected) {
    return hop_options_impl(w,elements,times,count,sort_returns,options,capacity,selected,nullptr);
}

spacepdhcg_cuda_status spacepdhcg_orbitweaver_hop_options_resident(
    spacepdhcg_orbitweaver_lambert_workspace* w,const spacepdhcg_orbitweaver_hop_elements* elements,
    const double* times,size_t count,int32_t sort_returns,spacepdhcg_gtoc12_collection_options** table,
    size_t* selected) {
    if(!table)return SPACEPDHCG_CUDA_INVALID_ARGUMENT;
    return hop_options_impl(w,elements,times,count,sort_returns,nullptr,count,selected,table);
}

spacepdhcg_cuda_status spacepdhcg_orbitweaver_hop_grid_device(
    spacepdhcg_orbitweaver_lambert_workspace* w,
    const spacepdhcg_orbitweaver_hop_elements* elements,
    const double* epochs,size_t n,const double* tofs,size_t nt,double* dv,uint8_t* ok
) {
    if(!w||!elements||!epochs||!tofs||!dv||!ok||!n||!nt||n>SIZE_MAX/nt
        ||!valid_elements(elements->departure)||!valid_elements(elements->arrival)
        ||!std::isfinite(elements->gravitational_parameter)||elements->gravitational_parameter<=0
        ||!std::isfinite(elements->departure_allowance)||elements->departure_allowance<0
        ||!std::isfinite(elements->arrival_allowance)||elements->arrival_allowance<0)
        return SPACEPDHCG_CUDA_INVALID_ARGUMENT;
    std::unique_lock<std::mutex> lock(w->api_mutex,std::try_to_lock);
    if(!lock.owns_lock()||w->busy.load())return SPACEPDHCG_CUDA_BUSY;
    int device=-1;auto status=cudaGetDevice(&device);
    if(status!=cudaSuccess)return mapped(status);
    if(device!=static_cast<int>(w->config.device_id))return SPACEPDHCG_CUDA_INVALID_ARGUMENT;
    return hop_grid_locked(w,elements,epochs,n,tofs,nt,dv,ok);
}

spacepdhcg_cuda_status spacepdhcg_orbitweaver_collection_grid_host(
    spacepdhcg_orbitweaver_lambert_workspace* w,
    const spacepdhcg_orbitweaver_hop_elements* elements,
    const double* epochs,size_t n,const double* tofs,size_t nt,double end,float* output
) {
    if(!w||!elements||!epochs||!tofs||!output||!n||!nt||n>SIZE_MAX/nt
        ||n*nt>SIZE_MAX/sizeof(float)||nt>SIZE_MAX/sizeof(double)||n>SIZE_MAX/sizeof(double)-nt
        ||!std::isfinite(end)||!valid_elements(elements->departure)||!valid_elements(elements->arrival)
        ||!std::isfinite(elements->gravitational_parameter)||elements->gravitational_parameter<=0
        ||!std::isfinite(elements->departure_allowance)||elements->departure_allowance<0
        ||!std::isfinite(elements->arrival_allowance)||elements->arrival_allowance<0)
        return SPACEPDHCG_CUDA_INVALID_ARGUMENT;
    std::unique_lock<std::mutex> lock(w->api_mutex,std::try_to_lock);
    if(!lock.owns_lock()||w->busy.load())return SPACEPDHCG_CUDA_BUSY;
    int device=-1;auto status=cudaGetDevice(&device);
    if(status!=cudaSuccess)return mapped(status);
    if(device!=static_cast<int>(w->config.device_id))return SPACEPDHCG_CUDA_INVALID_ARGUMENT;
    if(!w->collection_axes||w->collection_axes->capacity<n+nt){
        std::unique_ptr<GridAxesScratch> scratch(new(std::nothrow) GridAxesScratch);
        if(!scratch)return SPACEPDHCG_CUDA_OUT_OF_MEMORY;
        scratch->capacity=n+nt;status=cudaMalloc(&scratch->values,scratch->capacity*sizeof(double));
        if(status!=cudaSuccess)return mapped(status);w->collection_axes=std::move(scratch);
    }
    auto* de=w->collection_axes->values;auto* dt=de+n;
    status=cudaMemcpyAsync(de,epochs,n*sizeof(double),cudaMemcpyHostToDevice,w->stream);
    if(status==cudaSuccess)status=cudaMemcpyAsync(dt,tofs,nt*sizeof(double),cudaMemcpyHostToDevice,w->stream);
    auto* times=reinterpret_cast<double*>(w->requests);
    const size_t total=n*nt;
    for(size_t start=0;status==cudaSuccess&&start<total;){
        const size_t count=std::min(total-start,w->config.maximum_batch_size);
        grid_times<<<(count+127)/128,128,0,w->stream>>>(de,dt,nt,start,count,times);status=cudaGetLastError();
        if(status==cudaSuccess){element_hops<<<(count+127)/128,128,0,w->stream>>>(*elements,times,count,w->hops);status=cudaGetLastError();}
        if(status==cudaSuccess){launch_hops(w->hops,count,w->config.scan_samples_per_band,w->hop_results,w->scan_grid,w->stream);status=cudaGetLastError();}
        if(status==cudaSuccess){collection_grid_costs<<<(count+127)/128,128,0,w->stream>>>(w->hop_results,times,count,end,output+start);status=cudaGetLastError();}
        start+=count;
    }
    const auto done=cudaStreamSynchronize(w->stream);return mapped(status==cudaSuccess?done:status);
}

spacepdhcg_cuda_status spacepdhcg_orbitweaver_hop_grid_cached_host(
    spacepdhcg_orbitweaver_lambert_workspace* w,
    const spacepdhcg_orbitweaver_hop_elements* elements,
    const double* epochs,size_t n,const double* tofs,size_t nt,double* dv,uint8_t* ok
) {
    constexpr size_t limit=64U*1024U*1024U;
    if(!w||!elements||!epochs||!tofs||!dv||!ok||!n||!nt||n>SIZE_MAX/nt
        ||n*nt>SIZE_MAX/(sizeof(double)+sizeof(uint8_t))
        ||n>SIZE_MAX/sizeof(double)||nt>SIZE_MAX/sizeof(double)
        ||!valid_elements(elements->departure)||!valid_elements(elements->arrival)
        ||!std::isfinite(elements->gravitational_parameter)||elements->gravitational_parameter<=0
        ||!std::isfinite(elements->departure_allowance)||elements->departure_allowance<0
        ||!std::isfinite(elements->arrival_allowance)||elements->arrival_allowance<0)
        return SPACEPDHCG_CUDA_INVALID_ARGUMENT;
    for(size_t i=0;i<n;++i)if(!std::isfinite(epochs[i]))return SPACEPDHCG_CUDA_INVALID_ARGUMENT;
    for(size_t i=0;i<nt;++i)if(!std::isfinite(tofs[i])||tofs[i]<=0)return SPACEPDHCG_CUDA_INVALID_ARGUMENT;
    std::unique_lock<std::mutex> lock(w->api_mutex,std::try_to_lock);
    if(!lock.owns_lock()||w->busy.load())return SPACEPDHCG_CUDA_BUSY;
    int device=-1;auto status=cudaGetDevice(&device);
    if(status!=cudaSuccess)return mapped(status);
    if(device!=static_cast<int>(w->config.device_id))return SPACEPDHCG_CUDA_INVALID_ARGUMENT;
    const size_t cells=n*nt,bytes=cells*(sizeof(double)+sizeof(uint8_t));
    const auto copy=[&](const HopGridCacheEntry& entry) {
        auto code=cudaMemcpyAsync(dv,entry.dv,cells*sizeof(double),cudaMemcpyDeviceToDevice,w->stream);
        if(code==cudaSuccess)code=cudaMemcpyAsync(ok,entry.ok,cells,cudaMemcpyDeviceToDevice,w->stream);
        const auto done=cudaStreamSynchronize(w->stream);
        return mapped(code==cudaSuccess?done:code);
    };
    for(auto it=w->grid_cache.begin();it!=w->grid_cache.end();++it) {
        const auto& e=**it;
        if(e.epochs.size()==n && e.tofs.size()==nt
            && std::memcmp(&e.elements,elements,sizeof(*elements))==0
            && std::memcmp(e.epochs.data(),epochs,n*sizeof(double))==0
            && std::memcmp(e.tofs.data(),tofs,nt*sizeof(double))==0) {
            const auto code=copy(e);
            if(code==SPACEPDHCG_CUDA_SUCCESS){++w->grid_cache_hits;w->grid_cache.splice(w->grid_cache.begin(),w->grid_cache,it);}
            return code;
        }
    }
    ++w->grid_cache_misses;
    try {
        // Oversize requests are computed normally without retaining an entry.
        // Temporary staging contains only the already-host-resident grid axes.
        struct Axes { double *epochs{},*tofs{}; ~Axes(){cudaFree(epochs);cudaFree(tofs);} } axes;
        status=cudaMalloc(&axes.epochs,n*sizeof(double));
        if(status==cudaSuccess)status=cudaMalloc(&axes.tofs,nt*sizeof(double));
        if(status==cudaSuccess)status=cudaMemcpyAsync(axes.epochs,epochs,n*sizeof(double),cudaMemcpyHostToDevice,w->stream);
        if(status==cudaSuccess)status=cudaMemcpyAsync(axes.tofs,tofs,nt*sizeof(double),cudaMemcpyHostToDevice,w->stream);
        if(status!=cudaSuccess){cudaStreamSynchronize(w->stream);return mapped(status);}
        if(bytes>limit)return hop_grid_locked(w,elements,axes.epochs,n,axes.tofs,nt,dv,ok);
        // Also bound entry count so tiny tables cannot accumulate unbounded
        // allocation and host-key overhead while staying below the byte cap.
        while(w->grid_cache_bytes+bytes>limit || w->grid_cache.size()>=256) {
            w->grid_cache_bytes-=w->grid_cache.back()->bytes;
            w->grid_cache.pop_back();++w->grid_cache_evictions;
        }
        auto entry=std::make_unique<HopGridCacheEntry>();
        entry->elements=*elements;entry->epochs.assign(epochs,epochs+n);entry->tofs.assign(tofs,tofs+nt);entry->bytes=bytes;
        status=cudaMalloc(&entry->dv,cells*sizeof(double));
        if(status==cudaSuccess)status=cudaMalloc(&entry->ok,cells);
        if(status!=cudaSuccess){cudaStreamSynchronize(w->stream);return mapped(status);}
        auto code=hop_grid_locked(w,elements,axes.epochs,n,axes.tofs,nt,entry->dv,entry->ok);
        if(code!=SPACEPDHCG_CUDA_SUCCESS)return code;
        code=copy(*entry);if(code!=SPACEPDHCG_CUDA_SUCCESS)return code;
        w->grid_cache.push_front(std::move(entry));w->grid_cache_bytes+=bytes;
        return SPACEPDHCG_CUDA_SUCCESS;
    } catch(const std::bad_alloc&) {
        cudaStreamSynchronize(w->stream);return SPACEPDHCG_CUDA_OUT_OF_MEMORY;
    }
}

spacepdhcg_cuda_status spacepdhcg_orbitweaver_hop_grid_cache_stats(
    spacepdhcg_orbitweaver_lambert_workspace* w,uint64_t* hits,uint64_t* misses,
    uint64_t* evictions,uint64_t* bytes
) {
    if(!w||!hits||!misses||!evictions||!bytes)return SPACEPDHCG_CUDA_INVALID_ARGUMENT;
    std::unique_lock<std::mutex> lock(w->api_mutex,std::try_to_lock);
    if(!lock.owns_lock())return SPACEPDHCG_CUDA_BUSY;
    *hits=w->grid_cache_hits;*misses=w->grid_cache_misses;
    *evictions=w->grid_cache_evictions;*bytes=w->grid_cache_bytes;
    return SPACEPDHCG_CUDA_SUCCESS;
}

spacepdhcg_cuda_status spacepdhcg_orbitweaver_lambert_workspace_finish(
    spacepdhcg_orbitweaver_lambert_workspace* workspace
) {
    if (!workspace) return SPACEPDHCG_CUDA_INVALID_ARGUMENT;
    std::unique_lock<std::mutex> api_lock(workspace->api_mutex, std::try_to_lock);
    if (!api_lock.owns_lock()) return SPACEPDHCG_CUDA_BUSY;
    if (workspace->busy.load()) {
        const auto status = cudaEventSynchronize(workspace->completion);
        if (status != cudaSuccess) return mapped(status);
        workspace->feasible += workspace->counters[0];
        workspace->failed += workspace->counters[1];
        workspace->busy.store(false);
    }
    return SPACEPDHCG_CUDA_SUCCESS;
}

spacepdhcg_cuda_status spacepdhcg_orbitweaver_lambert_workspace_destroy(
    spacepdhcg_orbitweaver_lambert_workspace** workspace
) {
    if (workspace == nullptr || *workspace == nullptr) {
        return SPACEPDHCG_CUDA_INVALID_ARGUMENT;
    }
    auto* owned = *workspace;
    std::unique_lock<std::mutex> api_lock(owned->api_mutex, std::try_to_lock);
    if (!api_lock.owns_lock()) return SPACEPDHCG_CUDA_BUSY;
    if (owned->busy.load()) {
        const auto status = cudaEventSynchronize(owned->completion);
        if (status != cudaSuccess) {
            return mapped(status);
        }
    }
    cudaEventDestroy(owned->completion);
    cudaFree(owned->cancelled);
    cudaFree(owned->counters);
    cudaFree(owned->results);
    cudaFree(owned->requests);
    cudaFree(owned->scan_grid);
    cudaFree(owned->hops);
    cudaFree(owned->hop_results);
    owned->grid_cache.clear();
    owned->option_scratch.reset();
    owned->beam_scratch.reset();
    owned->collection_axes.reset();
    if (owned->owns_stream) {
        cudaStreamDestroy(owned->stream);
    }
    api_lock.unlock();
    delete owned;
    *workspace = nullptr;
    return SPACEPDHCG_CUDA_SUCCESS;
}

}  // extern "C"
