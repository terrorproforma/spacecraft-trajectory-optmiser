#include "spacepdhcg/cuda/orbitweaver_gpu_c_api.h"

#include <cuda_runtime_api.h>

#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <new>
#include <limits>
#include <mutex>

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

__device__ std::uint32_t scan(
    const double lower,
    const double upper,
    const std::uint32_t samples,
    const Geometry value,
    const spacepdhcg_orbitweaver_lambert_request& request,
    Root roots[2],
    const std::uint32_t root_limit = 2U,
    const double* cached = nullptr
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
                                     && bisect(
                                         previous_parameter,
                                         parameter,
                                         value,
                                         request,
                                         root
                                     );
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
    const double* cached
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
        if (!scan(-4*pi*pi, 4*pi*pi-1e-8, samples, value, request, roots, 1, cached)) continue;
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
    Root& root,const double* cached) {
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
                else found=bisect(previous_parameter,parameter,value,request,local);
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
    size_t count,uint32_t samples,spacepdhcg_orbitweaver_hop_result* results,const double* cached) {
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
        if(!scan_warp(-4*pi*pi,4*pi*pi-1e-8,samples,value,request,root,cached))continue;
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
void launch_hops(const spacepdhcg_orbitweaver_hop_request* requests,size_t count,
    uint32_t samples,spacepdhcg_orbitweaver_hop_result* results,const double* cached,cudaStream_t stream) {
    if(count<=16384)hop_warp_kernel<<<static_cast<unsigned>((count+3)/4),128,0,stream>>>(requests,count,samples,results,cached);
    else hop_kernel<<<static_cast<unsigned>((count+63)/64),64,0,stream>>>(requests,count,samples,results,cached);
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

spacepdhcg_cuda_status mapped(const cudaError_t status) {
    if (status == cudaSuccess) {
        return SPACEPDHCG_CUDA_SUCCESS;
    }
    return status == cudaErrorMemoryAllocation ? SPACEPDHCG_CUDA_OUT_OF_MEMORY
                                               : SPACEPDHCG_CUDA_RUNTIME_ERROR;
}

}  // namespace

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
};

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
    if (owned->owns_stream) {
        cudaStreamDestroy(owned->stream);
    }
    api_lock.unlock();
    delete owned;
    *workspace = nullptr;
    return SPACEPDHCG_CUDA_SUCCESS;
}

}  // extern "C"
