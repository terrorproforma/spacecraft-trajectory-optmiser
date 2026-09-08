#include "spacepdhcg/cuda/gtoc12_conic_c_api.h"
#include "spacepdhcg/cuda/gtoc12_discretisation_c_api.h"
#include "../internal/gtoc12_workspace_reuse.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <climits>
#include <new>
#include <vector>

namespace {
enum Kind { Constant, Phi, Psi, Affine, Boundary, Radial, State, Control,
            MinimumMass, Radius, Vinf, Fuel, Virtual, Smooth };
struct Term { int kind, index; double scale; };
struct Entry { int row, column; Term term; };
struct Pattern { std::vector<int> offsets, indices; std::vector<Term> terms; };
Pattern compile(std::vector<Entry>& entries, int columns) {
    std::sort(entries.begin(), entries.end(), [](const Entry& a, const Entry& b) {
        return a.column < b.column || (a.column == b.column && a.row < b.row);
    });
    Pattern p;
    p.offsets.resize(columns + 1);
    for (const auto& e : entries) {
        ++p.offsets[e.column + 1]; p.indices.push_back(e.row); p.terms.push_back(e.term);
    }
    for (int i = 0; i < columns; ++i) p.offsets[i + 1] += p.offsets[i];
    return p;
}
}

struct spacepdhcg_gtoc12_conic {
    int intervals{}, device{}, count{};
    spacepdhcg_gtoc12_conic_dimensions dimensions{};
    Pattern a_pattern, p_pattern;
    int *a_offsets{}, *a_indices{}, *p_offsets{}, *p_indices{}, *invalid{};
    double *packed{}, *states{}, *controls{}, *boundary{}, *fuel{};
    Term* terms{};
    spacepdhcg_gtoc12_conic_parameters* parameters{};
    spacepdhcg_gtoc12_discretisation* dynamics{};
    cudaStream_t stream{};
};

namespace {
template<bool Controlled=false>
__global__ void assemble(int count, const Term* terms, const double* phi,
    const double* psi, const double* affine, const double* states, const double* controls,
    const double* boundary, const double* fuel, const spacepdhcg_gtoc12_conic_parameters* params,
    const int* dynamics_invalid, double* output, int* invalid, const int* enabled=nullptr) {
    if constexpr (Controlled) {
        if (enabled && !*enabled) return;
        if (*dynamics_invalid) {
            if (blockIdx.x==0 && threadIdx.x==0) *invalid=1;
            return; // Includes invalid step counts on the first-ever launch.
        }
    }
    const auto p = *params;
    const bool valid = isfinite(p.trust_state) && p.trust_state >= 0.0
        && isfinite(p.trust_control) && p.trust_control >= 0.0
        && isfinite(p.virtual_weight) && p.virtual_weight >= 0.0
        && isfinite(p.minimum_mass) && p.minimum_mass > 0.0
        && isfinite(p.radius_floor) && p.radius_floor >= 0.0
        && isfinite(p.vinf_max) && p.vinf_max >= 0.0
        && isfinite(p.smoothness_weight) && p.smoothness_weight >= 0.0;
    if (blockIdx.x == 0 && threadIdx.x == 0 && (!valid || *dynamics_invalid)) atomicExch(invalid, 1);
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x) {
        const auto t = terms[i];
        double value = 0.0;
        switch (t.kind) {
        case Constant: value = t.scale; break;
        case Phi: value = t.scale * phi[t.index]; break;
        case Psi: value = t.scale * psi[t.index]; break;
        case Affine: value = t.scale * affine[t.index]; break;
        case Boundary: value = t.scale * boundary[t.index]; break;
        case Radial: {
            const int base = (t.index / 7) * 7;
            const double norm = sqrt(states[base]*states[base] + states[base+1]*states[base+1]
                + states[base+2]*states[base+2]);
            value = t.scale * states[t.index] / norm;
            break;
        }
        case State: value = t.scale * states[t.index] + p.trust_state; break;
        case Control: value = t.scale * controls[t.index] + p.trust_control; break;
        case MinimumMass: value = t.scale * p.minimum_mass; break;
        case Radius: value = t.scale * p.radius_floor; break;
        case Vinf: value = t.scale * p.vinf_max; break;
        case Fuel: value = t.scale * fuel[t.index]; break;
        case Virtual: value = t.scale * p.virtual_weight; break;
        case Smooth: value = t.scale * p.smoothness_weight; break;
        }
        output[i] = value;
        if (!isfinite(value)) atomicExch(invalid, 1);
    }
}
bool correct_device(const spacepdhcg_gtoc12_conic* w) {
    int current = -1;
    return w && cudaGetDevice(&current) == cudaSuccess && current == w->device;
}
__global__ void reset_controlled_invalid(const int* enabled,int* invalid) {
    if (!enabled || *enabled) *invalid=0;
}
template<class T> bool allocate(T** p, size_t count) {
    return cudaMalloc(p, count * sizeof(T)) == cudaSuccess;
}
template<class T> bool upload(T* destination, const T* source, size_t count, cudaStream_t stream) {
    return cudaMemcpyAsync(destination, source, count * sizeof(T), cudaMemcpyHostToDevice, stream) == cudaSuccess;
}
}

extern "C" void spacepdhcg_gtoc12_conic_destroy(spacepdhcg_gtoc12_conic* w) {
    if (!w) return;
    int previous = -1; cudaGetDevice(&previous); cudaSetDevice(w->device);
    if (w->stream) cudaStreamSynchronize(w->stream);
    spacepdhcg_gtoc12_discretisation_destroy(w->dynamics);
    cudaFree(w->a_offsets); cudaFree(w->a_indices); cudaFree(w->p_offsets); cudaFree(w->p_indices);
    cudaFree(w->invalid); cudaFree(w->packed); cudaFree(w->states); cudaFree(w->controls);
    cudaFree(w->boundary); cudaFree(w->fuel); cudaFree(w->terms); cudaFree(w->parameters);
    if (w->stream) cudaStreamDestroy(w->stream);
    if (previous >= 0 && previous != w->device) cudaSetDevice(previous);
    delete w;
}

extern "C" int spacepdhcg_gtoc12_conic_create(int intervals, int hold, int free_dep,
    int free_arr, double kappa, double mass_flow, const double* times,
    const double* boundary, const double* fuel, spacepdhcg_gtoc12_conic** output) {
    if (!output) return 1;
    *output = nullptr;
    // Counts and all term indices must fit the int32 sparse API, including
    // the maximum Lagrange pattern and packed RHS/objective/upper-P layout.
    if (intervals < 1 || intervals > (INT_MAX - 1024) / 512
        || !times || !boundary || !fuel || (hold != 0 && hold != 1)
        || (hold && intervals < 3) || (free_dep != 0 && free_dep != 1)
        || (free_arr != 0 && free_arr != 1)) return 1;
    for (int i = 0; i < 12; ++i) if (!std::isfinite(boundary[i])) return 1;
    for (int i = 0; i <= intervals; ++i) if (!std::isfinite(fuel[i]) || fuel[i] < 0.0) return 1;
    auto* w = new (std::nothrow) spacepdhcg_gtoc12_conic;
    if (!w) return 2;
    if (cudaGetDevice(&w->device) != cudaSuccess) { delete w; return 2; }
    const auto failed = [&](int status) { spacepdhcg_gtoc12_conic_destroy(w); return status; };
    w->intervals = intervals;
    try {
        const int n = intervals + 1, k = intervals, stencil = hold ? 4 : 1;
        const int iu = 7*n, inu = 11*n, isl = inu + 7*k, ivd = isl + 7*k, iva = ivd + 3*free_dep;
        const int variables = iva + 3*free_arr;
        int row = 0;
        std::vector<Entry> entries, quadratic;
        std::vector<Term> rhs, objective(variables, {Constant, 0, 0.0});
        const auto add = [&](int column, double scale, int kind = Constant, int index = 0) {
            entries.push_back({row, column, {kind, index, scale}});
        };
        const auto right = [&](double scale = 0.0, int kind = Constant, int index = 0) {
            rhs.push_back({kind, index, scale}); ++row;
        };
        for (int interval = 0; interval < k; ++interval) {
            const int first = hold ? std::min(std::max(interval - 1, 0), k - 3) : interval;
            for (int i = 0; i < 7; ++i) {
                add(7*(interval+1)+i, 1.0);
                // dm/dx is zero and dm/du has only the Gamma surrogate. The
                // variational mass row therefore has Phi[6,6] and Psi[6,3]
                // only, for every trajectory. Omit structural zeros once;
                // retain all potentially nonzero entries even during coast.
                for (int j = 0; j < 7; ++j)
                    if (i != 6 || j == 6) add(7*interval+j, -1.0, Phi, 49*interval+7*i+j);
                for (int s = 0; s < stencil; ++s)
                    for (int j = 0; j < 4; ++j)
                        if (i != 6 || j == 3)
                            add(iu+4*(first+s)+j, -1.0, Psi, (interval*stencil+s)*28+4*i+j);
                add(inu+7*interval+i, -1.0); right(1.0, Affine, 7*interval+i);
            }
        }
        for (int i = 0; i < 3; ++i) { add(i, 1.0); right(1.0, Boundary, i); }
        for (int i = 0; i < 3; ++i) {
            add(3+i, 1.0); if (free_dep) add(ivd+i, -1.0); right(1.0, Boundary, 3+i);
        }
        add(6, 1.0); right(1.0);
        for (int i = 0; i < 3; ++i) { add(7*k+i, 1.0); right(1.0, Boundary, 6+i); }
        for (int i = 0; i < 3; ++i) {
            add(7*k+3+i, 1.0); if (free_arr) add(iva+i, -1.0); right(1.0, Boundary, 9+i);
        }
        if (!hold) for (int i = 0; i < 4; ++i) { add(iu+4*k+i, 1.0); right(); }
        const int equalities = row;
        for (int node = 0; node < n; ++node) {
            add(iu+4*node+3, -1.0); right();
            add(iu+4*node+3, 1.0); right(1.0);
            add(7*node+6, -1.0); right(-1.0, MinimumMass);
            for (int i = 0; i < 3; ++i) add(7*node+i, -1.0, Radial, 7*node+i);
            right(-1.0, Radius);
            for (int i = 0; i < 7; ++i) {
                add(7*node+i, 1.0); right(1.0, State, 7*node+i);
                add(7*node+i, -1.0); right(-1.0, State, 7*node+i);
            }
            for (int i = 0; i < 4; ++i) {
                add(iu+4*node+i, 1.0); right(1.0, Control, 4*node+i);
                add(iu+4*node+i, -1.0); right(-1.0, Control, 4*node+i);
            }
        }
        for (int i = 0; i < 7*k; ++i) {
            add(inu+i, 1.0); add(isl+i, -1.0); right();
            add(inu+i, -1.0); add(isl+i, -1.0); right();
        }
        const int inequalities = row - equalities;
        for (int node = 0; node < n; ++node) {
            add(iu+4*node+3, -1.0); right();
            for (int i = 0; i < 3; ++i) { add(iu+4*node+i, -1.0); right(); }
        }
        for (int end = 0; end < 2; ++end) if (end ? free_arr : free_dep) {
            right(1.0, Vinf);
            for (int i = 0; i < 3; ++i) { add((end ? iva : ivd)+i, -1.0); right(); }
        }
        for (int node = 0; node < n; ++node) {
            objective[iu+4*node+3] = {Fuel, node, 1.0};
            for (int i = 0; i < 3; ++i) {
                const int index = iu+4*node+i;
                quadratic.push_back({index, index, {Smooth, 0, node == 0 || node == k ? 2.0 : 4.0}});
                if (node < k) quadratic.push_back({index, index+4, {Smooth, 0, -2.0}});
            }
        }
        for (int i = 0; i < 7*k; ++i) objective[isl+i] = {Virtual, 0, 1.0};
        w->a_pattern = compile(entries, variables); w->p_pattern = compile(quadratic, variables);
        w->dimensions = {variables, row, equalities, inequalities, n+free_dep+free_arr,
            static_cast<int>(entries.size()), static_cast<int>(quadratic.size())};
        std::vector<Term> terms = w->a_pattern.terms;
        terms.insert(terms.end(), rhs.begin(), rhs.end());
        terms.insert(terms.end(), objective.begin(), objective.end());
        terms.insert(terms.end(), w->p_pattern.terms.begin(), w->p_pattern.terms.end());
        w->count = static_cast<int>(terms.size());
        const int dynamic_status = spacepdhcg_gtoc12_discretisation_create(k, hold, kappa, mass_flow, times, &w->dynamics);
        if (dynamic_status) return failed(dynamic_status);
        if (cudaStreamCreateWithFlags(&w->stream, cudaStreamNonBlocking) != cudaSuccess
            || !allocate(&w->a_offsets, variables+1) || !allocate(&w->a_indices, entries.size())
            || !allocate(&w->p_offsets, variables+1) || !allocate(&w->p_indices, quadratic.size())
            || !allocate(&w->invalid, 1) || !allocate(&w->packed, terms.size())
            || !allocate(&w->terms, terms.size()) || !allocate(&w->states, n*7)
            || !allocate(&w->controls, n*4) || !allocate(&w->parameters, 1)
            || !allocate(&w->boundary, 12) || !allocate(&w->fuel, n)) return failed(2);
        const bool uploaded = upload(w->a_offsets, w->a_pattern.offsets.data(), variables+1, w->stream)
            && upload(w->a_indices, w->a_pattern.indices.data(), entries.size(), w->stream)
            && upload(w->p_offsets, w->p_pattern.offsets.data(), variables+1, w->stream)
            && upload(w->p_indices, w->p_pattern.indices.data(), quadratic.size(), w->stream)
            && upload(w->terms, terms.data(), terms.size(), w->stream)
            && upload(w->boundary, boundary, 12, w->stream) && upload(w->fuel, fuel, n, w->stream);
        // Drain even on enqueue failure before local term vectors are destroyed.
        const auto synced = cudaStreamSynchronize(w->stream);
        if (!uploaded || synced != cudaSuccess) return failed(2);
        *output = w;
        return 0;
    } catch (...) { return failed(2); }
}

int gtoc12_conic_rebind(spacepdhcg_gtoc12_conic* w,double kappa,double mass_flow,
    const double* times,const double* boundary,const double* fuel) {
    if(!correct_device(w) || !boundary || !fuel) return 1;
    for(int i=0;i<12;++i) if(!std::isfinite(boundary[i])) return 1;
    for(int i=0;i<=w->intervals;++i) if(!std::isfinite(fuel[i]) || fuel[i]<0.0) return 1;
    const int status=gtoc12_discretisation_rebind(w->dynamics,kappa,mass_flow,times);
    if(status) return status;
    if(!upload(w->boundary,boundary,12,w->stream) || !upload(w->fuel,fuel,w->intervals+1,w->stream)
        || cudaStreamSynchronize(w->stream)!=cudaSuccess) return 2;
    return 0;
}

extern "C" int spacepdhcg_gtoc12_conic_get_dimensions(spacepdhcg_gtoc12_conic* w,
    spacepdhcg_gtoc12_conic_dimensions* d) {
    if (!w || !d) return 1;
    *d = w->dimensions; return 0;
}
extern "C" int spacepdhcg_gtoc12_conic_copy_topology_host(spacepdhcg_gtoc12_conic* w,
    int* ao, int* ai, int* po, int* pi) {
    if (!w || !ao || !ai || !po || !pi) return 1;
    std::copy(w->a_pattern.offsets.begin(), w->a_pattern.offsets.end(), ao);
    std::copy(w->a_pattern.indices.begin(), w->a_pattern.indices.end(), ai);
    std::copy(w->p_pattern.offsets.begin(), w->p_pattern.offsets.end(), po);
    std::copy(w->p_pattern.indices.begin(), w->p_pattern.indices.end(), pi);
    return 0;
}
extern "C" int spacepdhcg_gtoc12_conic_outputs(spacepdhcg_gtoc12_conic* w,
    spacepdhcg_gtoc12_conic_device_outputs* out) {
    if (!w || !out) return 1;
    const auto& d = w->dimensions;
    *out = {w->a_offsets, w->a_indices, w->p_offsets, w->p_indices,
        w->packed, w->packed+d.a_nonzeros, w->packed+d.a_nonzeros+d.rows,
        w->packed+d.a_nonzeros+d.rows+d.variables, w->invalid};
    return 0;
}
extern "C" int spacepdhcg_gtoc12_conic_launch_device(spacepdhcg_gtoc12_conic* w,
    const double* states, const double* controls,
    const spacepdhcg_gtoc12_conic_parameters* params, int substeps, void* stream_pointer) {
    if (!correct_device(w) || !states || !controls || !params || substeps < 1) return 1;
    auto stream = static_cast<cudaStream_t>(stream_pointer);
    const int status = spacepdhcg_gtoc12_discretisation_launch_device(w->dynamics, states, controls, substeps, 1, stream);
    if (status) return status;
    const double *phi{}, *psi{}, *affine{}, *propagated{}; const int* dynamics_invalid{};
    spacepdhcg_gtoc12_discretisation_outputs(w->dynamics, &phi, &psi, &affine, &propagated, &dynamics_invalid);
    if (cudaMemsetAsync(w->invalid, 0, sizeof(int), stream) != cudaSuccess) return 2;
    assemble<false><<<std::min(1024, (w->count+255)/256), 256, 0, stream>>>(w->count, w->terms,
        phi, psi, affine, states, controls, w->boundary, w->fuel, params, dynamics_invalid, w->packed, w->invalid);
    return cudaGetLastError() == cudaSuccess ? 0 : 2;
}
extern "C" int spacepdhcg_gtoc12_conic_launch_controlled_device(spacepdhcg_gtoc12_conic* w,
    const double* states,const double* controls,const spacepdhcg_gtoc12_conic_parameters* params,
    const int* substeps,const int* enabled,void* stream_pointer) {
    if (!correct_device(w) || !states || !controls || !params || !substeps) return 1;
    auto stream=static_cast<cudaStream_t>(stream_pointer);
    const int status=spacepdhcg_gtoc12_discretisation_launch_controlled_device(
        w->dynamics,states,controls,substeps,enabled,1,stream);
    if (status) return status;
    const double *phi{},*psi{},*affine{},*propagated{}; const int* dynamics_invalid{};
    spacepdhcg_gtoc12_discretisation_outputs(w->dynamics,&phi,&psi,&affine,&propagated,&dynamics_invalid);
    reset_controlled_invalid<<<1,1,0,stream>>>(enabled,w->invalid);
    assemble<true><<<std::min(1024,(w->count+255)/256),256,0,stream>>>(w->count,w->terms,
        phi,psi,affine,states,controls,w->boundary,w->fuel,params,dynamics_invalid,w->packed,w->invalid,enabled);
    return cudaGetLastError()==cudaSuccess ? 0 : 2;
}
extern "C" int spacepdhcg_gtoc12_conic_evaluate_host(spacepdhcg_gtoc12_conic* w,
    const double* states, const double* controls, const spacepdhcg_gtoc12_conic_parameters* params,
    int substeps, double* output) {
    if (!correct_device(w) || !states || !controls || !params || !output || substeps < 1) return 1;
    const auto failed = [&](int status) { cudaStreamSynchronize(w->stream); return status; };
    const size_t n = w->intervals+1;
    if (!upload(w->states, states, n*7, w->stream) || !upload(w->controls, controls, n*4, w->stream)
        || !upload(w->parameters, params, 1, w->stream)) return failed(2);
    const int status = spacepdhcg_gtoc12_conic_launch_device(w, w->states, w->controls, w->parameters, substeps, w->stream);
    if (status) return failed(status);
    int invalid = 0;
    if (cudaMemcpyAsync(output, w->packed, w->count*sizeof(double), cudaMemcpyDeviceToHost, w->stream) != cudaSuccess
        || cudaMemcpyAsync(&invalid, w->invalid, sizeof(int), cudaMemcpyDeviceToHost, w->stream) != cudaSuccess) return failed(2);
    if (cudaStreamSynchronize(w->stream) != cudaSuccess) return 2;
    return invalid ? 3 : 0;
}
