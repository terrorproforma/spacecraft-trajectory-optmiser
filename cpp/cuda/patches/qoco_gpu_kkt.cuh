// SPDX-License-Identifier: Apache-2.0
// Build upper-triangular KKT CSR and update maps from device CSC inputs.
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_scan.cuh>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "cuda_types.h"

namespace qoco_gpu_kkt {
using Key = unsigned long long;
template<class T> struct Buffer {
    T* p{};
    void allocate(size_t n) { if (n) CUDA_CHECK(cudaMalloc(&p, n * sizeof(T))); }
    ~Buffer() { if (p) cudaFree(p); }
};
template<class T> struct Slice { T* p; };
template<class T> Slice<T> take(unsigned char*& cursor, size_t count) {
    auto* result = reinterpret_cast<T*>(cursor);
    // Preserve cudaMalloc alignment for every CUB input/output view.
    cursor += (count * sizeof(T) + 255) & ~size_t(255);
    return {result};
}
__device__ Key key(int row, int col) {
    return (static_cast<Key>(row) << 32) | static_cast<unsigned>(col);
}
__global__ void matrix(const int* offsets, const int* rows, const double* values,
                       int n, int nnz, int base, int shift, bool transpose,
                       Key* keys, int* ids, double* data) {
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= nnz) return;
    int lo = 0, hi = n;
    while (lo < hi) {
        const int mid = lo + (hi - lo) / 2;
        if (offsets[mid + 1] <= e) lo = mid + 1; else hi = mid;
    }
    keys[base + e] = transpose ? key(lo, shift + rows[e]) : key(rows[e], lo);
    ids[base + e] = base + e;
    data[base + e] = values[e];
}
__global__ void equality(int n, int p, int base, double regularization,
                         Key* keys, int* ids, double* values) {
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e < p) {
        keys[base + e] = key(n + e, n + e);
        ids[base + e] = base + e;
        values[base + e] = -regularization;
    }
}
__global__ void cone_sizes(const int* q, int nsoc, int* sizes, int* triangles) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i <= nsoc) {
        const long long size = i < nsoc ? q[i] : 0;
        sizes[i] = static_cast<int>(size);
        triangles[i] = static_cast<int>(size * (size + 1) / 2);
    }
}
__global__ void nt(int n, int p, int l, int nsoc, int count, int base,
                   const int* starts, const int* triangles,
                   Key* keys, int* ids, double* values) {
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= count) return;
    int row = e, col = e;
    if (e >= l) {
        const int index = e - l;
        int lo = 0, hi = nsoc;
        while (lo < hi) {
            const int mid = lo + (hi - lo) / 2;
            if (triangles[mid + 1] <= index) lo = mid + 1; else hi = mid;
        }
        const int cone = lo, local = index - triangles[cone];
        lo = 0; hi = starts[cone + 1] - starts[cone];
        while (lo < hi) {
            const int mid = lo + (hi - lo) / 2;
            if (static_cast<long long>(mid + 1) * (mid + 2) / 2 <= local)
                lo = mid + 1;
            else hi = mid;
        }
        col = l + starts[cone] + lo;
        row = l + starts[cone] + local - static_cast<int>(static_cast<long long>(lo) * (lo + 1) / 2);
    }
    keys[base + e] = key(n + p + row, n + p + col);
    ids[base + e] = base + e;
    values[base + e] = row == col ? -1.0 : 0.0;
}
__global__ void scatter(const Key* keys, const int* ids, const double* input,
                        int count, int pn, int an, int gn, int p, int n,
                        int* columns, double* values, int* pm, int* am, int* gm,
                        int* wm, int* diag) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const int e = ids[i], col = static_cast<int>(keys[i] & 0xffffffffULL);
    const int row = static_cast<int>(keys[i] >> 32);
    columns[i] = col;
    values[i] = input[e];
    if (e < pn) pm[e] = i;
    else if (e < pn + an) am[e - pn] = i;
    else if (e < pn + an + gn) gm[e - pn - an] = i;
    else if (e >= pn + an + gn + p) {
        wm[e - pn - an - gn - p] = i;
        if (row == col) diag[row - n - p] = i;
    }
}
__global__ void row_offsets(const Key* keys, int nnz, int rows, int* offsets) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row > rows) return;
    const Key target = static_cast<Key>(row) << 32;
    int lo = 0, hi = nnz;
    while (lo < hi) {
        const int mid = lo + (hi - lo) / 2;
        if (keys[mid] < target) lo = mid + 1; else hi = mid;
    }
    offsets[row] = lo;
}
template<class T> std::vector<T> download(const T* source, size_t count) {
    std::vector<T> result(count);
    if (count) CUDA_CHECK(cudaMemcpy(result.data(), source, count * sizeof(T), cudaMemcpyDeviceToHost));
    return result;
}
// Independent legacy construction is used only under an explicit test flag.
static void compare(QOCOProblemData* d, QOCOSettings* settings, LinSysData* s, int wn) {
    const int pn = get_nnz(d->P), an = get_nnz(d->A), gn = get_nnz(d->G);
    std::vector<int> pm(pn), am(an), gm(gn), wm(wn), diag(d->m);
    set_cpu_mode(1);
    auto* k = construct_kkt(get_csc_matrix(d->P), get_csc_matrix(d->A), get_csc_matrix(d->G),
        get_csc_matrix(d->At), get_csc_matrix(d->Gt), settings->kkt_static_reg_A,
        d->n, d->m, d->p, d->l, d->nsoc, get_data_vectori(d->q),
        pm.data(), am.data(), gm.data(), wm.data(), diag.data(), wn);
    set_cpu_mode(0);
    std::vector<int> offsets(k->n + 1), columns(k->nnz), inverse(k->nnz);
    std::vector<double> values(k->nnz);
    for (int e = 0; e < k->nnz; ++e) ++offsets[k->i[e] + 1];
    for (int r = 0; r < k->n; ++r) offsets[r + 1] += offsets[r];
    auto cursor = offsets;
    for (int c = 0; c < k->n; ++c) for (int e = k->p[c]; e < k->p[c + 1]; ++e) {
        const int target = cursor[k->i[e]]++;
        columns[target] = c; values[target] = k->x[e]; inverse[e] = target;
    }
    bool ok = offsets == download(s->d_csr_rows, offsets.size())
        && columns == download(s->d_csr_columns, columns.size())
        && values == download(s->d_csr_val, values.size());
    auto check_map = [&](const std::vector<int>& old, const int* gpu, const int* transpose) {
        auto actual = download(gpu, old.size());
        for (size_t i = 0; i < old.size(); ++i)
            if (actual[i] != inverse[old[transpose ? transpose[i] : i]]) ok = false;
    };
    check_map(pm, s->d_PregtoKKTcsr, nullptr); check_map(am, s->d_AttoKKTcsr, d->AtoAt);
    check_map(gm, s->d_GttoKKTcsr, d->GtoGt); check_map(wm, s->d_nt2kktcsr, nullptr);
    check_map(diag, s->d_ntdiag2kktcsr, nullptr);
    free_qoco_csc_matrix(k);
    if (!ok) { std::fprintf(stderr, "GPU KKT assembly/reference mismatch\n"); std::exit(1); }
}
static int build(QOCOProblemData* d, QOCOSettings* settings, LinSysData* s, int wn,
                 bool verify = false) {
    const int pn = get_nnz(d->P), an = get_nnz(d->A), gn = get_nnz(d->G);
    const long long total = static_cast<long long>(pn) + an + gn + d->p + wn;
    if (total <= 0 || total > INT_MAX) { std::fprintf(stderr, "Unsupported GPU KKT size\n"); std::exit(1); }
    const int count = static_cast<int>(total), base = pn + an + gn + d->p;
    Buffer<unsigned char> storage, scratch;
    const size_t cones = static_cast<size_t>(d->nsoc) + 1;
    storage.allocate(32 * static_cast<size_t>(count) + 16 * cones + 9 * 256);
    auto* cursor = storage.p;
    auto keys = take<Key>(cursor, count), sorted = take<Key>(cursor, count);
    auto input = take<double>(cursor, count);
    auto ids = take<int>(cursor, count), ordered = take<int>(cursor, count);
    auto sizes = take<int>(cursor, cones), triangles = take<int>(cursor, cones);
    auto starts = take<int>(cursor, cones), wt = take<int>(cursor, cones);
    size_t sorting = 0, scanning = 0;
    CUDA_CHECK(cub::DeviceRadixSort::SortPairs(nullptr, sorting, keys.p, sorted.p, ids.p, ordered.p, count));
    CUDA_CHECK(cub::DeviceScan::ExclusiveSum(nullptr, scanning, sizes.p, starts.p, d->nsoc + 1));
    scratch.allocate(sorting > scanning ? sorting : scanning);
    auto launch = [&](QOCOMatrix* matrix_data, int nnz, int offset, int shift, bool transpose) {
        if (!nnz) return;
        const auto* c = matrix_data->d_csc_host;
        matrix<<<(nnz + 255) / 256, 256>>>(c->p, c->i, c->x, c->n, nnz, offset, shift,
            transpose, keys.p, ids.p, input.p);
    };
    launch(d->P, pn, 0, 0, false); launch(d->A, an, pn, d->n, true);
    launch(d->G, gn, pn + an, d->n + d->p, true);
    if (d->p) equality<<<(d->p + 255) / 256, 256>>>(d->n, d->p, pn + an + gn,
        settings->kkt_static_reg_A, keys.p, ids.p, input.p);
    cone_sizes<<<(d->nsoc + 256) / 256, 256>>>(d->q->d_data, d->nsoc, sizes.p, triangles.p);
    CUDA_CHECK(cub::DeviceScan::ExclusiveSum(scratch.p, scanning, sizes.p, starts.p, d->nsoc + 1));
    CUDA_CHECK(cub::DeviceScan::ExclusiveSum(scratch.p, scanning, triangles.p, wt.p, d->nsoc + 1));
    if (wn) nt<<<(wn + 255) / 256, 256>>>(d->n, d->p, d->l, d->nsoc, wn, base,
        starts.p, wt.p, keys.p, ids.p, input.p);
    CUDA_CHECK(cub::DeviceRadixSort::SortPairs(scratch.p, sorting, keys.p, sorted.p, ids.p, ordered.p, count));
    auto allocate = [](auto** out, size_t n) {
        *out = nullptr; if (n) CUDA_CHECK(cudaMalloc(out, n * sizeof(**out)));
    };
    allocate(&s->d_csr_rows, static_cast<size_t>(s->Kn) + 1); allocate(&s->d_csr_columns, count);
    allocate(&s->d_csr_val, count); allocate(&s->d_PregtoKKTcsr, pn); allocate(&s->d_AttoKKTcsr, an);
    allocate(&s->d_GttoKKTcsr, gn); allocate(&s->d_nt2kktcsr, wn); allocate(&s->d_ntdiag2kktcsr, d->m);
    allocate(&s->d_WtW, wn);
    scatter<<<(count + 255) / 256, 256>>>(sorted.p, ordered.p, input.p, count, pn, an, gn, d->p, d->n,
        s->d_csr_columns, s->d_csr_val, s->d_PregtoKKTcsr, s->d_AttoKKTcsr, s->d_GttoKKTcsr,
        s->d_nt2kktcsr, s->d_ntdiag2kktcsr);
    row_offsets<<<(s->Kn + 256) / 256, 256>>>(sorted.p, count, s->Kn, s->d_csr_rows);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(nullptr));
    if (const char* oracle = std::getenv("SPACEPDHCG_TEST_QOCO_KKT_COMPARE");
        verify || (oracle && oracle[0] == '1'))
        compare(d, settings, s, wn);
    return count;
}
}

// Test-only entry point isolates assembly from vendor analysis. This also lets
// the exact oracle exercise duplicate entries unsupported by cuDSS analysis.
extern "C" int qoco_test_gpu_kkt(QOCOCscMatrix* P, QOCOCscMatrix* A, QOCOCscMatrix* G,
                                 int n, int p, int m, int l, int nsoc, int* q) {
    QOCOProblemData d{}; QOCOSettings settings{}; LinSysData s{};
    d.n = n; d.p = p; d.m = m; d.l = l; d.nsoc = nsoc; s.Kn = n + p + m;
    settings.kkt_static_reg_A = 1e-8;
    d.P = new_qoco_matrix(P); d.A = new_qoco_matrix(A); d.G = new_qoco_matrix(G);
    d.q = new_qoco_vectori(q, nsoc);
    d.AtoAt = static_cast<int*>(qoco_calloc(get_nnz(d.A), sizeof(int)));
    d.GtoGt = static_cast<int*>(qoco_calloc(get_nnz(d.G), sizeof(int)));
    set_cpu_mode(1);
    auto* at = create_transposed_matrix(get_csc_matrix(d.A), d.AtoAt);
    auto* gt = create_transposed_matrix(get_csc_matrix(d.G), d.GtoGt);
    d.At = new_qoco_matrix(at); d.Gt = new_qoco_matrix(gt);
    free_qoco_csc_matrix(at); free_qoco_csc_matrix(gt);
    set_cpu_mode(0);
    int wn = l; for (int i = 0; i < nsoc; ++i) wn += q[i] * (q[i] + 1) / 2;
    const int count = qoco_gpu_kkt::build(&d, &settings, &s, wn, true);
    void* buffers[]{s.d_csr_rows, s.d_csr_columns, s.d_csr_val, s.d_PregtoKKTcsr,
        s.d_AttoKKTcsr, s.d_GttoKKTcsr, s.d_nt2kktcsr, s.d_ntdiag2kktcsr, s.d_WtW};
    for (void* ptr : buffers) if (ptr) CUDA_CHECK(cudaFree(ptr));
    for (auto* ptr : {d.P, d.A, d.G, d.At, d.Gt}) free_qoco_matrix(ptr);
    free_qoco_vectori(d.q); qoco_free(d.AtoAt); qoco_free(d.GtoGt);
    return count;
}
