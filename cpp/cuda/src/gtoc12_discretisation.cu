#include "spacepdhcg/cuda/gtoc12_discretisation_c_api.h"
#include "../internal/gtoc12_workspace_reuse.h"

#include <cuda_runtime.h>
#include <cmath>
#include <limits>
#include <new>

struct spacepdhcg_gtoc12_discretisation {
    int intervals{}, stencil{}, device{};
    double kappa{}, mass_flow{};
    double *times{}, *states{}, *controls{}, *a{}, *b{}, *c{}, *propagated{};
    int* invalid{};
    cudaStream_t stream{};
};

namespace {
constexpr int dimension = 7 + 49 + 4 * 28;

template <bool Linearise, bool Controlled = false>
__global__ void interval_kernel(
    int intervals, int stencil, int substeps, double kappa, double mass_flow,
    const double* times, const double* states, const double* controls,
    double* out_a, double* out_b, double* out_c, double* propagated, int* invalid,
    const int* device_substeps = nullptr, const int* enabled = nullptr
) {
    if constexpr (Controlled) {
        // Uniform across the block, before any barrier or numerical read.
        if (enabled && !*enabled) return;
        substeps = *device_substeps;
        if (substeps < 1) return;  // The preceding reset kernel marked invalid.
    }
    const int interval = static_cast<int>(blockIdx.x);
    const int tid = static_cast<int>(threadIdx.x);
    const int width = Linearise ? 56 + stencil * 28 : 7;
    const int first = stencil == 1 ? interval : min(max(interval - 1, 0), intervals - 3);
    __shared__ double y[dimension], stage[dimension], slope[4][dimension];
    __shared__ double a[49], b[28], u[4], weights[4], inverse_r3, inverse_r5;
    const double h = (times[interval + 1] - times[interval]) / substeps;
    double t = times[interval];
    for (int i = tid; i < width; i += blockDim.x) {
        y[i] = i < 7 ? states[static_cast<size_t>(interval) * 7 + i]
            : (i < 56 && (i - 7) / 7 == (i - 7) % 7 ? 1.0 : 0.0);
    }
    __syncthreads();
    for (int step = 0; step < substeps; ++step) {
        for (int rk = 0; rk < 4; ++rk) {
            const double fraction = rk == 0 ? 0.0 : (rk == 3 ? 1.0 : 0.5);
            for (int i = tid; i < width; i += blockDim.x)
                stage[i] = rk == 0 ? y[i] : y[i] + fraction * h * slope[rk - 1][i];
            __syncthreads();
            if (tid == 0) {
                const double tau = t + fraction * h;
                for (int j = 0; j < stencil; ++j) {
                    weights[j] = 1.0;
                    for (int k = 0; k < stencil; ++k)
                        if (k != j) weights[j] *= (tau - times[first + k])
                            / (times[first + j] - times[first + k]);
                }
                for (int j = 0; j < 4; ++j) {
                    u[j] = 0.0;
                    for (int k = 0; k < stencil; ++k)
                        u[j] += weights[k] * controls[static_cast<size_t>(first + k) * 4 + j];
                    if (!isfinite(u[j])) atomicExch(invalid, 1);
                }
                const double r2 = stage[0]*stage[0] + stage[1]*stage[1] + stage[2]*stage[2];
                const double radius = sqrt(r2);
                inverse_r3 = 1.0 / (radius * radius * radius);
                inverse_r5 = 1.0 / (radius * radius * radius * radius * radius);
                if (!(radius > 0.0) || !(stage[6] > 0.0)) atomicExch(invalid, 1);
            }
            __syncthreads();
            if constexpr (Linearise) {
                for (int i = tid; i < 49; i += blockDim.x) {
                    const int row = i / 7, col = i % 7;
                    double value = 0.0;
                    if (row < 3 && col == row + 3) value = 1.0;
                    if (row >= 3 && row < 6 && col < 3)
                        value = -(row - 3 == col ? 1.0 : 0.0) * inverse_r3
                            + 3.0 * stage[row - 3] * stage[col] * inverse_r5;
                    if (row >= 3 && row < 6 && col == 6)
                        value = -kappa * u[row - 3] / (stage[6] * stage[6]);
                    a[i] = value;
                }
                for (int i = tid; i < 28; i += blockDim.x) {
                    const int row = i / 4, col = i % 4;
                    b[i] = row >= 3 && row < 6 && col == row - 3 ? kappa / stage[6]
                        : (row == 6 && col == 3 ? -mass_flow : 0.0);
                }
                __syncthreads();
            }
            for (int i = tid; i < width; i += blockDim.x) {
                double value = 0.0;
                if (i < 3) value = stage[i + 3];
                else if (i < 6) value = -stage[i - 3] * inverse_r3 + kappa * u[i - 3] / stage[6];
                else if (i == 6) value = -mass_flow * sqrt(u[0]*u[0] + u[1]*u[1] + u[2]*u[2]);
                else if constexpr (Linearise) {
                    if (i < 56) {
                        const int row = (i - 7) / 7, col = (i - 7) % 7;
                        for (int j = 0; j < 7; ++j) value += a[row * 7 + j] * stage[7 + j * 7 + col];
                    } else {
                        const int offset = i - 56, s = offset / 28;
                        const int row = (offset % 28) / 4, col = offset % 4;
                        for (int j = 0; j < 7; ++j) value += a[row * 7 + j] * stage[56 + s * 28 + j * 4 + col];
                        value += weights[s] * b[row * 4 + col];
                    }
                }
                slope[rk][i] = value;
            }
            __syncthreads();
        }
        for (int i = tid; i < width; i += blockDim.x)
            y[i] += h / 6.0 * (slope[0][i] + 2.0*slope[1][i] + 2.0*slope[2][i] + slope[3][i]);
        t += h;
        __syncthreads();
    }
    for (int i = tid; i < width; i += blockDim.x) {
        if (!isfinite(y[i])) atomicExch(invalid, 1);
        if (i < 7) {
            const auto index = static_cast<size_t>(interval) * 7 + i;
            propagated[index] = y[i];
            if constexpr (Linearise) {
                double ax = 0.0, bu = 0.0;
                for (int j = 0; j < 7; ++j) ax += y[7 + i * 7 + j] * states[static_cast<size_t>(interval) * 7 + j];
                for (int s = 0; s < stencil; ++s)
                    for (int j = 0; j < 4; ++j)
                        bu += y[56 + s * 28 + i * 4 + j] * controls[static_cast<size_t>(first + s) * 4 + j];
                out_c[index] = y[i] - ax - bu;
                if (!isfinite(out_c[index])) atomicExch(invalid, 1);
            }
        } else if constexpr (Linearise) {
            if (i < 56) out_a[static_cast<size_t>(interval) * 49 + i - 7] = y[i];
            else out_b[static_cast<size_t>(interval) * stencil * 28 + i - 56] = y[i];
        }
    }
}

bool correct_device(const spacepdhcg_gtoc12_discretisation* w) {
    int device = -1;
    return w && cudaGetDevice(&device) == cudaSuccess && device == w->device;
}
__global__ void reset_controlled_invalid(const int* substeps, const int* enabled, int* invalid) {
    if (!enabled || *enabled) *invalid = *substeps < 1;
}
}

extern "C" void spacepdhcg_gtoc12_discretisation_destroy(spacepdhcg_gtoc12_discretisation* w) {
    if (!w) return;
    int previous = -1;
    cudaGetDevice(&previous);
    cudaSetDevice(w->device);
    if (w->stream) cudaStreamSynchronize(w->stream);
    cudaFree(w->times); cudaFree(w->states); cudaFree(w->controls);
    cudaFree(w->a); cudaFree(w->b); cudaFree(w->c); cudaFree(w->propagated); cudaFree(w->invalid);
    if (w->stream) cudaStreamDestroy(w->stream);
    if (previous >= 0 && previous != w->device) cudaSetDevice(previous);
    delete w;
}

extern "C" int spacepdhcg_gtoc12_discretisation_create(
    int intervals, int hold, double kappa, double mass_flow, const double* times,
    spacepdhcg_gtoc12_discretisation** output
) {
    if (!output) return 1;
    *output = nullptr;
    if (intervals < 1 || intervals == std::numeric_limits<int>::max() || !times
        || (hold != 0 && hold != 1) || (hold == 1 && intervals < 3)
        || !std::isfinite(kappa) || kappa <= 0.0 || !std::isfinite(mass_flow) || mass_flow <= 0.0)
        return 1;
    for (int i = 0; i <= intervals; ++i)
        if (!std::isfinite(times[i]) || (i && !(times[i] > times[i - 1]))) return 1;
    const auto n = static_cast<size_t>(intervals);
    if (n > std::numeric_limits<size_t>::max() / (112 * sizeof(double))) return 1;
    auto* w = new (std::nothrow) spacepdhcg_gtoc12_discretisation;
    if (!w) return 2;
    w->intervals = intervals; w->stencil = hold ? 4 : 1; w->kappa = kappa; w->mass_flow = mass_flow;
    const auto failed = [&]() { spacepdhcg_gtoc12_discretisation_destroy(w); return 2; };
    if (cudaGetDevice(&w->device) != cudaSuccess) { delete w; return 2; }
    if (cudaStreamCreateWithFlags(&w->stream, cudaStreamNonBlocking) != cudaSuccess) return failed();
    if (cudaMalloc(&w->times, (n + 1) * sizeof(double)) != cudaSuccess
        || cudaMalloc(&w->states, (n + 1) * 7 * sizeof(double)) != cudaSuccess
        || cudaMalloc(&w->controls, (n + 1) * 4 * sizeof(double)) != cudaSuccess
        || cudaMalloc(&w->a, n * 49 * sizeof(double)) != cudaSuccess
        || cudaMalloc(&w->b, n * w->stencil * 28 * sizeof(double)) != cudaSuccess
        || cudaMalloc(&w->c, n * 7 * sizeof(double)) != cudaSuccess
        || cudaMalloc(&w->propagated, n * 7 * sizeof(double)) != cudaSuccess
        || cudaMalloc(&w->invalid, sizeof(int)) != cudaSuccess) return failed();
    if (cudaMemcpyAsync(w->times, times, (n + 1) * sizeof(double), cudaMemcpyHostToDevice, w->stream) != cudaSuccess
        || cudaStreamSynchronize(w->stream) != cudaSuccess) return failed();
    *output = w;
    return 0;
}

int gtoc12_discretisation_rebind(spacepdhcg_gtoc12_discretisation* w,
    double kappa,double mass_flow,const double* times) {
    if(!correct_device(w) || !times || !std::isfinite(kappa) || kappa<=0.0
        || !std::isfinite(mass_flow) || mass_flow<=0.0) return 1;
    for(int i=0;i<=w->intervals;++i)
        if(!std::isfinite(times[i]) || (i && !(times[i]>times[i-1]))) return 1;
    if(cudaMemcpyAsync(w->times,times,(w->intervals+1)*sizeof(double),cudaMemcpyHostToDevice,w->stream)!=cudaSuccess
        || cudaStreamSynchronize(w->stream)!=cudaSuccess) return 2;
    w->kappa=kappa;w->mass_flow=mass_flow;
    return 0;
}

extern "C" int spacepdhcg_gtoc12_discretisation_launch_device(
    spacepdhcg_gtoc12_discretisation* w, const double* states, const double* controls,
    int substeps, int linearise, void* stream_pointer
) {
    if (!correct_device(w) || !states || !controls || substeps < 1 || (linearise != 0 && linearise != 1)) return 1;
    auto stream = static_cast<cudaStream_t>(stream_pointer);
    if (cudaMemsetAsync(w->invalid, 0, sizeof(int), stream) != cudaSuccess) return 2;
    if (linearise)
        interval_kernel<true><<<w->intervals, 128, 0, stream>>>(w->intervals, w->stencil, substeps,
            w->kappa, w->mass_flow, w->times, states, controls, w->a, w->b, w->c, w->propagated, w->invalid);
    else
        interval_kernel<false><<<w->intervals, 32, 0, stream>>>(w->intervals, w->stencil, substeps,
            w->kappa, w->mass_flow, w->times, states, controls, w->a, w->b, w->c, w->propagated, w->invalid);
    return cudaGetLastError() == cudaSuccess ? 0 : 2;
}

extern "C" int spacepdhcg_gtoc12_discretisation_launch_controlled_device(
    spacepdhcg_gtoc12_discretisation* w, const double* states, const double* controls,
    const int* substeps, const int* enabled, int linearise, void* stream_pointer
) {
    if (!correct_device(w) || !states || !controls || !substeps || (linearise != 0 && linearise != 1)) return 1;
    auto stream = static_cast<cudaStream_t>(stream_pointer);
    reset_controlled_invalid<<<1, 1, 0, stream>>>(substeps, enabled, w->invalid);
    if (linearise)
        interval_kernel<true, true><<<w->intervals, 128, 0, stream>>>(w->intervals, w->stencil, 0,
            w->kappa, w->mass_flow, w->times, states, controls, w->a, w->b, w->c, w->propagated,
            w->invalid, substeps, enabled);
    else
        interval_kernel<false, true><<<w->intervals, 32, 0, stream>>>(w->intervals, w->stencil, 0,
            w->kappa, w->mass_flow, w->times, states, controls, w->a, w->b, w->c, w->propagated,
            w->invalid, substeps, enabled);
    return cudaGetLastError() == cudaSuccess ? 0 : 2;
}

extern "C" int spacepdhcg_gtoc12_discretisation_outputs(
    spacepdhcg_gtoc12_discretisation* w, const double** a, const double** b,
    const double** c, const double** propagated, const int** invalid
) {
    if (!w || !a || !b || !c || !propagated || !invalid) return 1;
    *a = w->a; *b = w->b; *c = w->c; *propagated = w->propagated; *invalid = w->invalid;
    return 0;
}

extern "C" int spacepdhcg_gtoc12_discretisation_evaluate_host(
    spacepdhcg_gtoc12_discretisation* w, const double* states, const double* controls,
    int substeps, int linearise, double* a, double* b, double* c, double* propagated
) {
    if (!correct_device(w) || !states || !controls || !propagated || substeps < 1
        || (linearise != 0 && linearise != 1) || (linearise && (!a || !b || !c))) return 1;
    const auto n = static_cast<size_t>(w->intervals);
    int invalid = 0;
    const auto stream = w->stream;
    // On an enqueue error, drain pending copies before the caller can free its
    // host arrays. Errors never fall back to CPU propagation.
    const auto failed = [&]() { cudaStreamSynchronize(stream); return 2; };
    if (cudaMemcpyAsync(w->states, states, (n + 1) * 7 * sizeof(double), cudaMemcpyHostToDevice, stream) != cudaSuccess
        || cudaMemcpyAsync(w->controls, controls, (n + 1) * 4 * sizeof(double), cudaMemcpyHostToDevice, stream) != cudaSuccess)
        return failed();
    const auto status = spacepdhcg_gtoc12_discretisation_launch_device(w, w->states, w->controls, substeps, linearise, stream);
    if (status) { cudaStreamSynchronize(stream); return status; }
    if (linearise && (cudaMemcpyAsync(a, w->a, n * 49 * sizeof(double), cudaMemcpyDeviceToHost, stream) != cudaSuccess
        || cudaMemcpyAsync(b, w->b, n * w->stencil * 28 * sizeof(double), cudaMemcpyDeviceToHost, stream) != cudaSuccess
        || cudaMemcpyAsync(c, w->c, n * 7 * sizeof(double), cudaMemcpyDeviceToHost, stream) != cudaSuccess)) return failed();
    if (cudaMemcpyAsync(propagated, w->propagated, n * 7 * sizeof(double), cudaMemcpyDeviceToHost, stream) != cudaSuccess
        || cudaMemcpyAsync(&invalid, w->invalid, sizeof(int), cudaMemcpyDeviceToHost, stream) != cudaSuccess) return failed();
    if (cudaStreamSynchronize(stream) != cudaSuccess) return 2;
    return invalid ? 3 : 0;
}
