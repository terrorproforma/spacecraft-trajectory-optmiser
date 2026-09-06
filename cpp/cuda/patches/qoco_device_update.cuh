// SPDX-License-Identifier: Apache-2.0
// Included by the prepared CUDA algebra after qoco_gather.cuh. This interface
// owns numerical updates for its solver; do not mix it with host update calls.
#include "qoco.h"
#include <algorithm>
#include <cstdlib>
#include <memory>
#include <vector>

namespace qoco_device_update {
struct Failure { cudaError_t code; };
inline void check(cudaError_t code) { if (code != cudaSuccess) throw Failure{code}; }
template<class T> struct Buffer {
    T* data{};
    ~Buffer() { cudaFree(data); }
    void allocate(size_t count) { if (count) check(cudaMalloc(&data, count * sizeof(T))); }
    void upload(const T* values, size_t count) {
        allocate(count); if (count) check(cudaMemcpy(data, values, count * sizeof(T), cudaMemcpyHostToDevice));
    }
};
struct Matrix {
    int rows{}, columns{}, nonzeros{};
    const int *offsets{}, *indices{}, *row_offsets{}, *entries{}, *columns_by_entry{};
    double* values{};
};
inline Matrix matrix(QOCOMatrix* m) {
#ifdef SPACEPDHCG_QOCO_DEFERRED_TRANSPOSES
    // Compatibility transposes are refreshed only on explicit access. Do not
    // capture their optional payload in the numerical-update context.
    if (m && m->transpose_source) return {};
#endif
    if (!m || !m->d_csc_host) return {};
    const auto* c = m->d_csc_host;
    const auto* g = m->gather;
    return {c->m, c->n, c->nnz, c->p, c->i, g->offsets, g->entries, g->columns, c->x};
}
struct Scales { double *d, *e, *f, *di, *ei, *fi, *delta; };
struct Pair { double sum, maximum; };
struct Ranges { double low[3], high[3]; };
struct Context {
    QOCOSolver* solver{};
    Matrix p, a, g, at, gt;
    Scales scales{};
    Buffer<int> p_source, diagonal, cone_starts, invalid;
    // Borrow the matrix-owned stable row ordering. Destroy this context before
    // its solver; numerical updates preserve the solver's matrix topology.
    const int *at_source{}, *gt_source{};
    Buffer<double> p_norm, factors, result;
    Buffer<Pair> partial;
    Buffer<Ranges> ranges;
    int input_p{}, blocks{}, cone_count{}, tasks{};
    cudaEvent_t ready{};
    ~Context() { if (ready) cudaEventDestroy(ready); }
};
__device__ double reciprocal(double x) { return fabs(x) > 1e-15 ? 1.0 / x : DBL_MAX; }
// An empty row/column or zero objective has nothing to equilibrate. Its
// identity scale preserves the problem; DBL_MAX can overflow on the next
// pass or turn an explicitly stored zero into NaN during matrix scaling.
__device__ double equilibration_reciprocal(double x) { return x > 1e-15 ? 1.0 / x : 1.0; }
__device__ double column_norm(Matrix a, int col) {
    double result = 0;
    if (a.nonzeros) for (int k = a.offsets[col]; k < a.offsets[col + 1]; ++k) result = fmax(result, fabs(a.values[k]));
    return result;
}
__device__ double row_norm(Matrix a, int row) {
    double result = 0;
    if (a.nonzeros) for (int k = a.row_offsets[row]; k < a.row_offsets[row + 1]; ++k)
        result = fmax(result, fabs(a.values[a.entries[k]]));
    return result;
}
__global__ void load_p(Matrix p, const int* sources, const double* input) {
    for (long long k = blockIdx.x * blockDim.x + threadIdx.x; k < p.nonzeros; k += gridDim.x * blockDim.x)
        p.values[k] = sources[k] < 0 ? 0.0 : input[sources[k]];
}
__global__ void initialize(double* c, Scales s, int n, int p, int m, double kinv, double* factors, int* invalid) {
    for (long long i = blockIdx.x * blockDim.x + threadIdx.x; i < static_cast<long long>(n) + p + m; i += gridDim.x * blockDim.x) {
        if (i < n) {
            // Match the old host matrix-update order: its Ruiz cost is the prior
            // unscaled objective; the new objective is supplied after equilibration.
            c[i] = __dmul_rn(__dmul_rn(kinv, c[i]), s.di[i]);
            s.d[i] = 1;
        } else if (i < n + p) s.e[i - n] = 1;
        else s.f[i - n - p] = 1;
    }
    if (blockIdx.x == 0 && threadIdx.x == 0) { factors[0] = 1; factors[1] = 1; *invalid = 0; }
}
__global__ void norms(Matrix a, Matrix p, Matrix g, Scales scales, double* p_norm, int n, int eq, int cone) {
    for (long long i = blockIdx.x * blockDim.x + threadIdx.x; i < static_cast<long long>(n) + eq + cone; i += gridDim.x * blockDim.x) {
        double norm = 0;
        if (i < n) {
            p_norm[i] = fmax(column_norm(p, i), row_norm(p, i));
            norm = fmax(p_norm[i], fmax(column_norm(a, i), column_norm(g, i)));
        } else if (i < n + eq) norm = row_norm(a, i - n);
        else norm = row_norm(g, i - n - eq);
        scales.delta[i] = equilibration_reciprocal(sqrt(norm));
    }
}
__global__ void cost_partial(const double* p_norm, const double* c, int n, Pair* out) {
    __shared__ Pair shared[128];
    Pair value{};
    for (long long i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        value.sum += p_norm[i]; value.maximum = fmax(value.maximum, fabs(c[i]));
    }
    shared[threadIdx.x] = value; __syncthreads();
    for (int stride = 64; stride; stride >>= 1) {
        if (threadIdx.x < stride) { shared[threadIdx.x].sum += shared[threadIdx.x + stride].sum;
            shared[threadIdx.x].maximum = fmax(shared[threadIdx.x].maximum, shared[threadIdx.x + stride].maximum); }
        __syncthreads();
    }
    if (!threadIdx.x) out[blockIdx.x] = shared[0];
}
__global__ void cost_finish(const Pair* partial, int count, int n, double* factors) {
    __shared__ Pair shared[256];
    shared[threadIdx.x] = threadIdx.x < count ? partial[threadIdx.x] : Pair{};
    __syncthreads();
    for (int stride = 128; stride; stride >>= 1) {
        if (threadIdx.x < stride) { shared[threadIdx.x].sum += shared[threadIdx.x + stride].sum;
            shared[threadIdx.x].maximum = fmax(shared[threadIdx.x].maximum, shared[threadIdx.x + stride].maximum); }
        __syncthreads();
    }
    if (!threadIdx.x) { factors[0] = equilibration_reciprocal(fmax(shared[0].sum / n, shared[0].maximum));
        factors[1] = __dmul_rn(factors[1], factors[0]); }
}
__global__ void cone_scales(double* f, const int* starts, int count) {
    const int cone = blockIdx.x;
    if (cone >= count) return;
    const double value = f[starts[cone]];
    for (int i = starts[cone] + 1 + threadIdx.x; i < starts[cone + 1]; i += blockDim.x) f[i] = value;
}
__global__ void scale_matrix(Matrix a, const double* row, const double* column, const double* cost) {
    for (long long k = blockIdx.x * blockDim.x + threadIdx.x; k < a.nonzeros; k += gridDim.x * blockDim.x) {
        double value = a.values[k];
        if (cost) value = __dmul_rn(value, cost[0]);
        a.values[k] = __dmul_rn(value, __dmul_rn(column[a.columns_by_entry[k]], row[a.indices[k]]));
    }
}
__global__ void accumulate_scales(Scales s, double* c, int n, int p, int m, const double* factors) {
    for (long long i = blockIdx.x * blockDim.x + threadIdx.x; i < static_cast<long long>(n) + p + m; i += gridDim.x * blockDim.x) {
        if (i < n) { s.d[i] = __dmul_rn(s.d[i], s.delta[i]); c[i] = __dmul_rn(__dmul_rn(factors[0], c[i]), s.delta[i]); }
        else if (i < n + p) s.e[i - n] = __dmul_rn(s.e[i - n], s.delta[i]);
        else s.f[i - n - p] = __dmul_rn(s.f[i - n - p], s.delta[i]);
    }
}
__global__ void regularize(Matrix p, const int* diagonal, double reg) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < p.columns; i += gridDim.x * blockDim.x)
        p.values[diagonal[i]] += reg;
}
__global__ void transpose_values(Matrix source, Matrix target, const int* mapping) {
    for (long long k = blockIdx.x * blockDim.x + threadIdx.x; k < target.nonzeros; k += gridDim.x * blockDim.x)
        target.values[k] = source.values[mapping[k]];
}
__global__ void finish_vectors(Scales s, int n, int p, int m, const double* input,
                               double* c, double* b, double* h, const double* factors, int* invalid) {
    for (long long i = blockIdx.x * blockDim.x + threadIdx.x; i < static_cast<long long>(n) + p + m; i += gridDim.x * blockDim.x) {
        double value;
        if (i < n) { s.di[i] = reciprocal(s.d[i]); value = c[i] = __dmul_rn(__dmul_rn(factors[1], s.d[i]), input[i]); }
        else if (i < n + p) { const int k = i - n; s.ei[k] = reciprocal(s.e[k]); value = b[k] = __dmul_rn(s.e[k], input[i]); }
        else { const int k = i - n - p; s.fi[k] = reciprocal(s.f[k]); value = h[k] = __dmul_rn(s.f[k], input[i]); }
        if (!isfinite(value)) atomicOr(invalid, 1);
    }
}
__device__ Ranges empty_ranges() { Ranges value{}; for (int i = 0; i < 3; ++i) value.low[i] = DBL_MAX; return value; }
__device__ void merge_ranges(Ranges& a, const Ranges& b) {
    for (int i = 0; i < 3; ++i) { a.low[i] = fmin(a.low[i], b.low[i]); a.high[i] = fmax(a.high[i], b.high[i]); }
}
__global__ void range_partial(Matrix p, Matrix a, Matrix g, const double* raw_vectors, int n, int eq, int cone,
                              Ranges* output, int* invalid) {
    __shared__ Ranges shared[128];
    Ranges result = empty_ranges();
    const long long count = static_cast<long long>(p.nonzeros) + a.nonzeros + g.nonzeros + n + eq + cone;
    for (long long i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += gridDim.x * blockDim.x) {
        long long k = i; int category; double value;
        if (k < p.nonzeros) { category = 0; value = p.values[k]; }
        else if ((k -= p.nonzeros) < a.nonzeros) { category = 1; value = a.values[k]; }
        else if ((k -= a.nonzeros) < g.nonzeros) { category = 1; value = g.values[k]; }
        else { k -= g.nonzeros; category = k < n ? 0 : 2; value = raw_vectors[k]; }
        if (!isfinite(value)) atomicOr(invalid, 1);
        result.low[category] = fmin(result.low[category], fabs(value));
        result.high[category] = fmax(result.high[category], fabs(value));
    }
    shared[threadIdx.x] = result; __syncthreads();
    for (int stride = 64; stride; stride >>= 1) { if (threadIdx.x < stride) merge_ranges(shared[threadIdx.x], shared[threadIdx.x + stride]); __syncthreads(); }
    if (!threadIdx.x) output[blockIdx.x] = shared[0];
}
__global__ void range_finish(const Ranges* input, int count, const double* factors, const int* invalid, double* output) {
    __shared__ Ranges shared[256];
    shared[threadIdx.x] = threadIdx.x < count ? input[threadIdx.x] : empty_ranges(); __syncthreads();
    for (int stride = 128; stride; stride >>= 1) { if (threadIdx.x < stride) merge_ranges(shared[threadIdx.x], shared[threadIdx.x + stride]); __syncthreads(); }
    if (!threadIdx.x) {
        output[0] = factors[1]; output[1] = reciprocal(factors[1]);
        for (int i = 0; i < 3; ++i) { output[2 + i * 2] = shared[0].low[i]; output[3 + i * 2] = shared[0].high[i]; }
        output[8] = *invalid || !isfinite(output[0]) || !isfinite(output[1]);
    }
}
inline int blocks(int count) { return std::min(256, (count - 1) / 256 + 1); }
__global__ void source_map(int count, int input_count, const int* added, int added_count, int* sources) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += gridDim.x * blockDim.x) {
        int lo = 0, hi = added_count;
        while (lo < hi) {
            const int mid = lo + (hi - lo) / 2;
            if (added[mid] < i) lo = mid + 1; else hi = mid;
        }
        sources[i] = !input_count || (lo < added_count && added[lo] == i) ? -1 : i - lo;
    }
}
__global__ void diagonal_map(Matrix p, int* diagonal, int* invalid) {
    for (int col = blockIdx.x * blockDim.x + threadIdx.x; col < p.columns; col += gridDim.x * blockDim.x) {
        int found = -1;
        for (int k = p.offsets[col]; k < p.offsets[col + 1]; ++k)
            if (p.indices[k] == col) { found = k; break; }
        diagonal[col] = found;
        if (found < 0) atomicOr(invalid, 1);
    }
}
__global__ void copy_cone_starts(const int* input, int count, int m, int* output) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i <= count; i += gridDim.x * blockDim.x)
        output[i] = i < count ? input[i] : m;
}
inline void compare_setup_maps(Context* w, QOCOProblemData* data) {
    const char* flag = std::getenv("SPACEPDHCG_TEST_QOCO_UPDATE_MAPS_COMPARE");
    if (!flag || flag[0] != '1') return;
#ifdef SPACEPDHCG_QOCO_DEFERRED_TRANSPOSES
    qoco_materialize_host_mirror(data->At);
    qoco_materialize_host_mirror(data->Gt);
#endif
    const auto download = [](const int* ptr, int count) {
        std::vector<int> values(count);
        if (count) check(cudaMemcpy(values.data(), ptr, count * sizeof(int), cudaMemcpyDeviceToHost));
        return values;
    };
    auto source = download(w->p_source.data, w->p.nonzeros);
    int next = 0, added = 0;
    for (int i = 0; i < w->p.nonzeros; ++i) {
        int expected = -1;
        if (!w->input_p || (added < data->Pnum_nzadded && data->Pnzadded_idx[added] == i)) ++added;
        else expected = next++;
        if (source[i] != expected) throw Failure{cudaErrorInvalidValue};
    }
    if (next != w->input_p || added != data->Pnum_nzadded) throw Failure{cudaErrorInvalidValue};
    const auto diagonal = download(w->diagonal.data, data->n);
    const auto* p = data->P->csc;
    for (int col = 0; col < data->n; ++col) {
        int expected = -1;
        for (int k = p->p[col]; k < p->p[col + 1]; ++k) if (p->i[k] == col) { expected = k; break; }
        if (diagonal[col] != expected) throw Failure{cudaErrorInvalidValue};
    }
    const auto check_transpose = [&](const int* original, const int* gpu, int count) {
        const auto actual = download(gpu, count);
        for (int i = 0; i < count; ++i)
            if (original[i] < 0 || original[i] >= count || actual[original[i]] != i)
                throw Failure{cudaErrorInvalidValue};
    };
    check_transpose(data->AtoAt, w->at_source, w->a.nonzeros);
    check_transpose(data->GtoGt, w->gt_source, w->g.nonzeros);
    const auto starts = download(w->cone_starts.data, data->nsoc + 1);
    int expected = data->l;
    for (int i = 0; i < data->nsoc; ++i) {
        if (starts[i] != expected) throw Failure{cudaErrorInvalidValue};
        expected += data->q->data[i];
    }
    if (starts.back() != expected || expected != data->m) throw Failure{cudaErrorInvalidValue};
}
} // namespace qoco_device_update

extern "C" int qoco_gpu_create_numeric_update(QOCOSolver* solver, int pnnz, int annz, int gnnz, void** output) {
    using namespace qoco_device_update;
    if (!output) return 1;
    *output = nullptr;
    if (!solver || !solver->work || !solver->work->data || pnnz < 0 || annz < 0 || gnnz < 0) return 1;
    try {
        auto w = std::make_unique<Context>(); w->solver = solver;
        auto* data = solver->work->data; auto* s = solver->work->scaling;
        w->p = matrix(data->P); w->a = matrix(data->A); w->g = matrix(data->G);
        w->at = matrix(data->At); w->gt = matrix(data->Gt);
        if (data->n <= 0 || w->p.nonzeros != pnnz + data->Pnum_nzadded || w->a.nonzeros != annz || w->g.nonzeros != gnnz
            || static_cast<long long>(data->n) + data->p + data->m > INT_MAX) return 1;
        w->scales = {s->Druiz->d_data, s->Eruiz->d_data, s->Fruiz->d_data, s->Dinvruiz->d_data,
                     s->Einvruiz->d_data, s->Finvruiz->d_data, s->delta->d_data};
        w->tasks = data->n + data->p + data->m; w->blocks = std::min(256, (data->n - 1) / 128 + 1); w->input_p = pnnz;
        w->p_source.allocate(w->p.nonzeros); w->diagonal.allocate(data->n);
        w->at_source = w->a.entries; w->gt_source = w->g.entries;
        w->cone_starts.allocate(static_cast<size_t>(data->nsoc) + 1); w->cone_count = data->nsoc;
        w->p_norm.allocate(data->n); w->partial.allocate(w->blocks); w->ranges.allocate(256);
        w->factors.allocate(2); w->result.allocate(9); w->invalid.allocate(1);
        Buffer<int> added;
        const int added_count = pnnz ? data->Pnum_nzadded : 0;
        added.upload(data->Pnzadded_idx, added_count);
        check(cudaMemsetAsync(w->invalid.data, 0, sizeof(int)));
        source_map<<<blocks(w->p.nonzeros), 256>>>(w->p.nonzeros, pnnz, added.data, added_count, w->p_source.data);
        diagonal_map<<<blocks(data->n), 256>>>(w->p, w->diagonal.data, w->invalid.data);
        copy_cone_starts<<<blocks(data->nsoc + 1), 256>>>(solver->work->soc_idx->d_data,
            data->nsoc, data->m, w->cone_starts.data);
        check(cudaGetLastError());
        int invalid = 0;
        check(cudaMemcpyAsync(&invalid, w->invalid.data, sizeof(int), cudaMemcpyDeviceToHost));
        check(cudaStreamSynchronize(nullptr));
        if (invalid) return 1;
        compare_setup_maps(w.get(), data);
        check(cudaEventCreateWithFlags(&w->ready, cudaEventDisableTiming));
        // The pinned CPU Ruiz setup uploads matrices/scales but omits c/b/h.
        // Repair those initial device vectors before any solve or unscaling.
        if (solver->settings->ruiz_iters > 0) {
            for (auto* vector : {data->c, data->b, data->h}) if (vector->len)
                check(cudaMemcpy(vector->d_data, vector->data, vector->len * sizeof(double), cudaMemcpyHostToDevice));
        }
        *output = w.release(); return 0;
    } catch (const Failure&) { return 2; } catch (const std::bad_alloc&) { return 2; }
}

extern "C" int qoco_gpu_update_numeric(void* opaque, const double* packed, cudaStream_t producer) {
    using namespace qoco_device_update;
    auto* w = static_cast<Context*>(opaque);
    if (!w || !packed) return 1;
    auto* solver = w->solver; auto* data = solver->work->data; auto* s = solver->work->scaling;
    const int n = data->n, p = data->p, m = data->m, launch = blocks(w->tasks);
    const double* av = packed + w->input_p;
    const double* gv = av + w->a.nonzeros;
    const double* vectors = gv + w->g.nonzeros;
    const auto scales = w->scales;
    double result[9]{};
#ifdef SPACEPDHCG_QOCO_DEFERRED_TRANSPOSES
    for (auto* matrix : {data->At, data->Gt})
        if (matrix && matrix->transpose_source) matrix->transpose_values_pending = 1;
#endif
    try {
        check(cudaEventRecord(w->ready, producer)); check(cudaStreamWaitEvent(nullptr, w->ready, 0));
        load_p<<<blocks(w->p.nonzeros), 256>>>(w->p, w->p_source.data, packed);
        if (w->a.nonzeros) check(cudaMemcpyAsync(w->a.values, av, w->a.nonzeros * sizeof(double), cudaMemcpyDeviceToDevice));
        if (w->g.nonzeros) check(cudaMemcpyAsync(w->g.values, gv, w->g.nonzeros * sizeof(double), cudaMemcpyDeviceToDevice));
        initialize<<<launch, 256>>>(data->c->d_data, scales, n, p, m, s->kinv, w->factors.data, w->invalid.data);
        for (int iteration = 0; iteration < solver->settings->ruiz_iters; ++iteration) {
            norms<<<launch, 256>>>(w->a, w->p, w->g, scales, w->p_norm.data, n, p, m);
            cost_partial<<<w->blocks, 128>>>(w->p_norm.data, data->c->d_data, n, w->partial.data);
            cost_finish<<<1, 256>>>(w->partial.data, w->blocks, n, w->factors.data);
            if (w->cone_count) cone_scales<<<w->cone_count, 128>>>(scales.delta + n + p, w->cone_starts.data, w->cone_count);
            scale_matrix<<<blocks(w->p.nonzeros), 256>>>(w->p, scales.delta, scales.delta, w->factors.data);
            if (w->a.nonzeros) scale_matrix<<<blocks(w->a.nonzeros), 256>>>(w->a, scales.delta + n, scales.delta, nullptr);
            if (w->g.nonzeros) scale_matrix<<<blocks(w->g.nonzeros), 256>>>(w->g, scales.delta + n + p, scales.delta, nullptr);
            accumulate_scales<<<launch, 256>>>(scales, data->c->d_data, n, p, m, w->factors.data);
        }
        regularize<<<blocks(n), 256>>>(w->p, w->diagonal.data, solver->settings->kkt_static_reg_P);
        if (w->at.nonzeros) transpose_values<<<blocks(w->at.nonzeros), 256>>>(w->a, w->at, w->at_source);
        if (w->gt.nonzeros) transpose_values<<<blocks(w->gt.nonzeros), 256>>>(w->g, w->gt, w->gt_source);
        finish_vectors<<<launch, 256>>>(scales, n, p, m, vectors, data->c->d_data, data->b->d_data, data->h->d_data,
            w->factors.data, w->invalid.data);
        range_partial<<<256, 128>>>(w->p, w->a, w->g, vectors, n, p, m, w->ranges.data, w->invalid.data);
        range_finish<<<1, 256>>>(w->ranges.data, 256, w->factors.data, w->invalid.data, w->result.data);
        check(cudaGetLastError());
        check(cudaMemcpyAsync(result, w->result.data, sizeof(result), cudaMemcpyDeviceToHost));
        check(cudaStreamSynchronize(nullptr));
        if (result[8] != 0) return 3;
        s->k = result[0]; s->kinv = result[1];
        data->obj_range_min = result[2]; data->obj_range_max = result[3];
        data->constraint_range_min = result[4]; data->constraint_range_max = result[5];
        data->rhs_range_min = result[6]; data->rhs_range_max = result[7];
        solver->sol->status = QOCO_UNSOLVED;
        set_cpu_mode(0);
        solver->linsys->linsys_update_data(solver->linsys_data, data);
        check(cudaStreamSynchronize(nullptr));
        return 0;
    } catch (const Failure&) { cudaStreamSynchronize(nullptr); return 2; }
}
extern "C" void qoco_gpu_destroy_numeric_update(void* opaque) {
    delete static_cast<qoco_device_update::Context*>(opaque);
}
