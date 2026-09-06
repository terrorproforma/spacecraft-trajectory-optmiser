#include "native_qoco_gpu.h"

#include <cub/device/device_radix_sort.cuh>
#include <algorithm>
#include <cmath>
#include <climits>
#include <memory>
#include <new>
#include <vector>

namespace {
struct Failure { cudaError_t status; };
void check(cudaError_t status) { if (status != cudaSuccess) throw Failure{status}; }
template<class T> struct Buffer {
    T* data{};
    QocoAuditMemory* memory{};
    std::size_t bytes{};
    ~Buffer() { if (data) { cudaFree(data); memory->bytes -= bytes; } }
    void allocate(std::size_t count, QocoAuditMemory& stats) {
        if (!count) return;
        check(cudaMalloc(&data, count * sizeof(T)));
        bytes = count * sizeof(T); memory = &stats;
        ++stats.allocations; stats.bytes += bytes;
        stats.peak_bytes = std::max(stats.peak_bytes, stats.bytes);
    }
    Buffer() = default;
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
};
template<class T> void upload(T* target, const T* source, std::size_t count,
                              cudaStream_t stream, QocoAuditTransfers& transfers) {
    if (!count) return;
    check(cudaMemcpyAsync(target, source, count * sizeof(T), cudaMemcpyHostToDevice, stream));
    ++transfers.h2d_count;
    transfers.h2d_bytes += count * sizeof(T);
}

struct Matrix {
    int rows, columns, nonzeros;
    const int *offsets, *indices, *row_offsets, *entries, *entry_columns;
    const double* values;
};

__global__ void gather_keys(Matrix a, int* keys, int* entries, int* columns) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= a.columns) return;
    for (int k = a.offsets[col]; k < a.offsets[col + 1]; ++k) {
        keys[k] = a.indices[k]; entries[k] = k; columns[k] = col;
    }
}
__global__ void gather_offsets(const int* keys, int nnz, int rows, int* offsets) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row > rows) return;
    int lo = 0, hi = nnz;
    while (lo < hi) { int mid = lo + (hi - lo) / 2;
        if (keys[mid] < row) lo = mid + 1; else hi = mid; }
    offsets[row] = lo;
}

struct StoredMatrix {
    int rows{}, columns{}, nonzeros{};
    Buffer<int> offsets, indices, row_offsets, entries, entry_columns;
    Buffer<double> values;
    Matrix view() const { return {rows, columns, nonzeros, offsets.data, indices.data,
        row_offsets.data, entries.data, entry_columns.data, values.data}; }
    void create(const QocoAuditCsc& a, cudaStream_t stream, QocoAuditTransfers& transfers,
                QocoAuditMemory& memory) {
        if (a.rows < 0 || a.columns < 0 || a.nonzeros < 0 || !a.offsets
            || a.offsets[0] != 0 || a.offsets[a.columns] != a.nonzeros
            || (a.nonzeros && (!a.indices || !a.values))) throw Failure{cudaErrorInvalidValue};
        for (int j = 0; j < a.columns; ++j)
            if (a.offsets[j] < 0 || a.offsets[j + 1] < a.offsets[j]) throw Failure{cudaErrorInvalidValue};
        for (int k = 0; k < a.nonzeros; ++k)
            if (a.indices[k] < 0 || a.indices[k] >= a.rows) throw Failure{cudaErrorInvalidValue};
        rows = a.rows; columns = a.columns; nonzeros = a.nonzeros;
        offsets.allocate(static_cast<std::size_t>(columns) + 1, memory);
        indices.allocate(nonzeros, memory); row_offsets.allocate(static_cast<std::size_t>(rows) + 1, memory);
        entries.allocate(nonzeros, memory); entry_columns.allocate(nonzeros, memory); values.allocate(nonzeros, memory);
        upload(offsets.data, a.offsets, columns + 1, stream, transfers);
        upload(indices.data, a.indices, nonzeros, stream, transfers);
        upload(values.data, a.values, nonzeros, stream, transfers);
        Buffer<int> keys, sorted_keys, ids;
        Buffer<unsigned char> scratch;
        keys.allocate(nonzeros, memory); sorted_keys.allocate(nonzeros, memory); ids.allocate(nonzeros, memory);
        if (nonzeros) {
            gather_keys<<<(columns + 255) / 256, 256, 0, stream>>>(view(), keys.data, ids.data, entry_columns.data);
            check(cudaGetLastError());
            std::size_t bytes = 0;
            check(cub::DeviceRadixSort::SortPairs(nullptr, bytes, keys.data, sorted_keys.data,
                                                 ids.data, entries.data, nonzeros, 0, 32, stream));
            scratch.allocate(bytes, memory);
            check(cub::DeviceRadixSort::SortPairs(scratch.data, bytes, keys.data, sorted_keys.data,
                                                 ids.data, entries.data, nonzeros, 0, 32, stream));
        }
        gather_offsets<<<(rows + 256) / 256, 256, 0, stream>>>(sorted_keys.data, nonzeros, rows, row_offsets.data);
        check(cudaGetLastError());
        check(cudaStreamSynchronize(stream)); // release setup-only sort scratch safely
    }
    void update(const QocoAuditCsc& a, cudaStream_t stream, QocoAuditTransfers& transfers) {
        if (a.rows != rows || a.columns != columns || a.nonzeros != nonzeros)
            throw Failure{cudaErrorInvalidValue};
        upload(values.data, a.values, nonzeros, stream, transfers);
    }
};

// Each task owns a metric record. Reduction is deterministic and uses no atomics.
struct Metric {
    double primal{}, dual{}, dual_cone{}, complementarity{}, rhs{}, ax{}, c{}, px{}, aty{};
    double primal_objective{}, dual_objective{};
    int invalid{};
};
__device__ void observe(Metric& a, double value) { if (!isfinite(value)) a.invalid = 1; }
__device__ Metric combine(Metric a, const Metric& b) {
    a.primal = fmax(a.primal, b.primal); a.dual = fmax(a.dual, b.dual);
    a.dual_cone = fmax(a.dual_cone, b.dual_cone);
    a.complementarity = fmax(a.complementarity, b.complementarity);
    a.rhs = fmax(a.rhs, b.rhs); a.ax = fmax(a.ax, b.ax); a.c = fmax(a.c, b.c);
    a.px = fmax(a.px, b.px); a.aty = fmax(a.aty, b.aty);
    a.primal_objective += b.primal_objective; a.dual_objective += b.dual_objective;
    a.invalid |= b.invalid;
    return a;
}
__device__ double product_row(Matrix a, int row, const double* x, Metric& metric) {
    double sum = 0;
    for (int k = a.row_offsets[row]; k < a.row_offsets[row + 1]; ++k) {
        int entry = a.entries[k];
        double value = a.values[entry], input = x[a.entry_columns[entry]];
        observe(metric, value); observe(metric, input);
        sum += value * input;
    }
    observe(metric, sum);
    return sum;
}
__device__ double transpose_column(Matrix a, int col, const double* y, Metric& metric) {
    double sum = 0;
    for (int k = a.offsets[col]; k < a.offsets[col + 1]; ++k) {
        observe(metric, a.values[k]); observe(metric, y[a.indices[k]]);
        sum += a.values[k] * y[a.indices[k]];
    }
    observe(metric, sum);
    return sum;
}

__global__ void audit_rows(Matrix a, Matrix g, const double* b, const double* h,
                           int nonnegative, const double* x, const double* y, const double* z,
                           double* slack, Metric* metrics) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= a.rows + g.rows) return;
    Metric metric{};
    if (row < a.rows) {
        double ax = product_row(a, row, x, metric);
        observe(metric, b[row]); observe(metric, y[row]);
        metric.primal = fabs(ax - b[row]); metric.rhs = fabs(b[row]); metric.ax = fabs(ax);
        metric.dual_objective = b[row] * y[row];
    } else {
        int i = row - a.rows;
        double gx = product_row(g, i, x, metric);
        observe(metric, h[i]); observe(metric, z[i]);
        slack[i] = h[i] - gx;
        metric.rhs = fabs(h[i]); metric.ax = fabs(gx); metric.dual_objective = h[i] * z[i];
        if (i < nonnegative) {
            metric.primal = fmax(0.0, -slack[i]);
            metric.dual_cone = fmax(0.0, -z[i]);
            metric.complementarity = fabs(slack[i] * z[i]);
        }
    }
    observe(metric, metric.primal); observe(metric, metric.complementarity);
    observe(metric, metric.dual_objective);
    metrics[row] = metric;
}

__global__ void audit_variables(Matrix p, Matrix a, Matrix g, const double* c,
                                const double* x, const double* y, const double* z, Metric* metrics) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= p.columns) return;
    Metric metric{};
    observe(metric, x[i]); observe(metric, c[i]);
    double stationarity = c[i];
    // Upper-triangular CSC: mirrored entries in column i precede the diagonal
    // and later columns, matching the reference accumulation order.
    for (int k = p.offsets[i]; k < p.offsets[i + 1]; ++k) {
        if (p.indices[k] != i) {
            observe(metric, p.values[k]); observe(metric, x[p.indices[k]]);
            stationarity += p.values[k] * x[p.indices[k]];
        }
    }
    for (int k = p.row_offsets[i]; k < p.row_offsets[i + 1]; ++k) {
        int entry = p.entries[k];
        observe(metric, p.values[entry]); observe(metric, x[p.entry_columns[entry]]);
        stationarity += p.values[entry] * x[p.entry_columns[entry]];
    }
    double px = stationarity - c[i];
    double aty = transpose_column(a, i, y, metric) + transpose_column(g, i, z, metric);
    metric.dual = fabs(stationarity + aty); metric.c = fabs(c[i]);
    metric.px = fabs(px); metric.aty = fabs(aty);
    metric.primal_objective = (0.5 * px + c[i]) * x[i];
    observe(metric, metric.dual); observe(metric, metric.primal_objective);
    metrics[i] = metric;
}

__global__ void audit_cones(const int* starts, int count, const double* slack,
                            const double* z, Metric* metrics) {
    int cone = blockIdx.x * blockDim.x + threadIdx.x;
    if (cone >= count) return;
    int begin = starts[cone], end = starts[cone + 1];
    Metric metric{};
    double norm_s = 0, norm_z = 0, inner = slack[begin] * z[begin];
    observe(metric, slack[begin]); observe(metric, z[begin]);
    for (int k = begin + 1; k < end; ++k) {
        observe(metric, slack[k]); observe(metric, z[k]);
        norm_s += slack[k] * slack[k]; norm_z += z[k] * z[k]; inner += slack[k] * z[k];
    }
    metric.primal = fmax(0.0, sqrt(norm_s) - slack[begin]);
    metric.dual_cone = fmax(0.0, sqrt(norm_z) - z[begin]);
    metric.complementarity = fabs(inner);
    observe(metric, norm_s); observe(metric, norm_z); observe(metric, inner);
    metrics[cone] = metric;
}

__global__ void map_dual(Matrix map, int equalities, const double* y, const double* z, double* out) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= map.rows) return;
    double sum = 0;
    for (int k = map.row_offsets[row]; k < map.row_offsets[row + 1]; ++k) {
        int entry = map.entries[k], col = map.entry_columns[entry];
        sum += map.values[entry] * (col < equalities ? y[col] : z[col - equalities]);
    }
    out[row] = sum;
}

__global__ void reduce_metrics(const Metric* in, int count, Metric* out, QocoAuditResult* result) {
    __shared__ Metric partial[128];
    Metric metric{};
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x)
        metric = combine(metric, in[i]);
    partial[threadIdx.x] = metric;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset; offset /= 2) {
        if (threadIdx.x < offset) partial[threadIdx.x] = combine(partial[threadIdx.x], partial[threadIdx.x + offset]);
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        metric = partial[0];
        if (out) out[blockIdx.x] = metric;
        if (result) {
            double ps = 1 + metric.rhs + metric.ax;
            double ds = 1 + metric.c + metric.px + metric.aty;
            double gs = 1 + fabs(metric.primal_objective) + fabs(metric.dual_objective);
            if (metric.invalid || !isfinite(ps) || !isfinite(ds) || !isfinite(gs)) {
                *result = {INFINITY, INFINITY, INFINITY, INFINITY, INFINITY, INFINITY};
            } else {
                *result = {fmax(metric.primal, metric.dual_cone) / ps,
                    fmax(metric.dual / ds, metric.complementarity / gs),
                    metric.primal, metric.dual, metric.dual_cone / ps, metric.complementarity / gs};
            }
        }
    }
}
} // namespace

struct QocoGpuAudit {
    // Declared first so the counters outlive every owned buffer during teardown.
    QocoAuditMemory memory{};
    StoredMatrix p, a, g, dual_map;
    Buffer<double> c, b, h, slack, host_x, host_y, host_z;
    Buffer<int> cone_starts;
    Buffer<Metric> metrics, partial;
    Buffer<QocoAuditResult> result;
    Buffer<QocoReplayStatus> replay_status;
    int nonnegative{}, cone_count{}, tasks{}, blocks{};
    QocoAuditTransfers transfers{};
};

cudaError_t qoco_gpu_audit_create(const QocoAuditInput& in, bool host_solution,
                                cudaStream_t stream, QocoGpuAudit** output) {
    if (!output) return cudaErrorInvalidValue;
    *output = nullptr;
    try {
        auto w = std::make_unique<QocoGpuAudit>();
        int n = in.quadratic.columns;
        const auto tasks = static_cast<std::int64_t>(n) + in.equality.rows + in.conic.rows + in.soc_count;
        if (n <= 0 || in.quadratic.rows != n || in.equality.columns != n || in.conic.columns != n
            || in.equality.rows < 0 || in.conic.rows < 0 || tasks > INT_MAX - 256
            || in.dual_map.rows > INT_MAX - 256
            || (in.equality.rows && !in.equality_rhs) || (in.conic.rows && !in.conic_rhs)
            || !in.objective || in.nonnegative < 0 || in.nonnegative > in.conic.rows
            || in.soc_count < 0 || (in.soc_count && !in.soc_sizes)
            || in.dual_map.columns != in.equality.rows + in.conic.rows) return cudaErrorInvalidValue;
        std::vector<int> starts{in.nonnegative};
        for (int i = 0; i < in.soc_count; ++i) {
            if (in.soc_sizes[i] < 1 || in.soc_sizes[i] > in.conic.rows - starts.back()) return cudaErrorInvalidValue;
            starts.push_back(starts.back() + in.soc_sizes[i]);
        }
        if (starts.back() != in.conic.rows) return cudaErrorInvalidValue;
        w->p.create(in.quadratic, stream, w->transfers, w->memory);
        for (int col = 0; col < n; ++col)
            for (int k = in.quadratic.offsets[col]; k < in.quadratic.offsets[col + 1]; ++k)
                if (in.quadratic.indices[k] > col) return cudaErrorInvalidValue;
        w->a.create(in.equality, stream, w->transfers, w->memory); w->g.create(in.conic, stream, w->transfers, w->memory);
        w->dual_map.create(in.dual_map, stream, w->transfers, w->memory);
        for (int k = 0; k < in.dual_map.nonzeros; ++k)
            if (!std::isfinite(in.dual_map.values[k])) return cudaErrorInvalidValue;
        w->c.allocate(n, w->memory); w->b.allocate(in.equality.rows, w->memory); w->h.allocate(in.conic.rows, w->memory);
        w->slack.allocate(in.conic.rows, w->memory); w->cone_starts.allocate(starts.size(), w->memory);
        upload(w->c.data, in.objective, n, stream, w->transfers);
        upload(w->b.data, in.equality_rhs, in.equality.rows, stream, w->transfers);
        upload(w->h.data, in.conic_rhs, in.conic.rows, stream, w->transfers);
        upload(w->cone_starts.data, starts.data(), starts.size(), stream, w->transfers);
        w->nonnegative = in.nonnegative; w->cone_count = in.soc_count;
        w->tasks = static_cast<int>(tasks);
        w->blocks = std::min(256, (w->tasks - 1) / 128 + 1);
        w->metrics.allocate(w->tasks, w->memory); w->partial.allocate(w->blocks, w->memory); w->result.allocate(1, w->memory);
        w->replay_status.allocate(1,w->memory);
        if (host_solution) { w->host_x.allocate(n, w->memory); w->host_y.allocate(in.equality.rows, w->memory); w->host_z.allocate(in.conic.rows, w->memory); }
        check(cudaStreamSynchronize(stream));
        *output = w.release();
        return cudaSuccess;
    } catch (const Failure& e) { return e.status; }
      catch (const std::bad_alloc&) { return cudaErrorMemoryAllocation; }
}

cudaError_t qoco_gpu_audit_update(QocoGpuAudit* w, const QocoAuditInput& in, cudaStream_t stream) {
    if (!w) return cudaErrorInvalidValue;
    try {
        w->p.update(in.quadratic, stream, w->transfers); w->a.update(in.equality, stream, w->transfers);
        w->g.update(in.conic, stream, w->transfers);
        upload(w->c.data, in.objective, w->p.columns, stream, w->transfers);
        upload(w->b.data, in.equality_rhs, w->a.rows, stream, w->transfers);
        upload(w->h.data, in.conic_rhs, w->g.rows, stream, w->transfers);
        return cudaSuccess;
    } catch (const Failure& e) { return e.status; }
}

cudaError_t qoco_gpu_audit_upload_solution(QocoGpuAudit* w, const double* x, const double* y,
    const double* z, cudaStream_t stream, const double** dx, const double** dy, const double** dz) {
    if (!w || !w->host_x.data || !dx || !dy || !dz) return cudaErrorInvalidValue;
    try {
        upload(w->host_x.data, x, w->p.columns, stream, w->transfers);
        upload(w->host_y.data, y, w->a.rows, stream, w->transfers);
        upload(w->host_z.data, z, w->g.rows, stream, w->transfers);
        *dx = w->host_x.data; *dy = w->host_y.data; *dz = w->host_z.data;
        return cudaSuccess;
    } catch (const Failure& e) { return e.status; }
}

cudaError_t qoco_gpu_audit_update_device(QocoGpuAudit* w, const double* values, cudaStream_t stream) {
    if (!w || !values) return cudaErrorInvalidValue;
    const int counts[]{w->p.nonzeros, w->a.nonzeros, w->g.nonzeros, w->p.columns, w->a.rows, w->g.rows};
    double* destinations[]{w->p.values.data, w->a.values.data, w->g.values.data, w->c.data, w->b.data, w->h.data};
    for (int i = 0; i < 6; ++i) {
        if (counts[i]) {
            const auto status = cudaMemcpyAsync(destinations[i], values, counts[i] * sizeof(double), cudaMemcpyDeviceToDevice, stream);
            if (status != cudaSuccess) return status;
        }
        values += counts[i];
    }
    return cudaSuccess;
}

cudaError_t qoco_gpu_audit_run_device(QocoGpuAudit* w, const double* x, const double* y, const double* z,
                             double* mapped, cudaStream_t stream, const QocoAuditResult** output) {
    if (output) *output = nullptr;
    if (!w || !x || !output || (w->a.rows && !y) || (w->g.rows && !z)
        || (w->dual_map.rows && !mapped)) return cudaErrorInvalidValue;
    int rows = w->a.rows + w->g.rows;
    if (rows) audit_rows<<<(rows + 255) / 256, 256, 0, stream>>>(w->a.view(), w->g.view(), w->b.data,
        w->h.data, w->nonnegative, x, y, z, w->slack.data, w->metrics.data);
    audit_variables<<<(w->p.columns + 255) / 256, 256, 0, stream>>>(w->p.view(), w->a.view(), w->g.view(),
        w->c.data, x, y, z, w->metrics.data + rows);
    if (w->cone_count) audit_cones<<<(w->cone_count + 255) / 256, 256, 0, stream>>>(w->cone_starts.data,
        w->cone_count, w->slack.data, z, w->metrics.data + rows + w->p.columns);
    if (w->dual_map.rows) map_dual<<<(w->dual_map.rows + 255) / 256, 256, 0, stream>>>(w->dual_map.view(), w->a.rows, y, z, mapped);
    reduce_metrics<<<w->blocks, 128, 0, stream>>>(w->metrics.data, w->tasks, w->partial.data, nullptr);
    reduce_metrics<<<1, 128, 0, stream>>>(w->partial.data, w->blocks, nullptr, w->result.data);
    const auto status = cudaGetLastError();
    if (status != cudaSuccess) return status;
    *output = w->result.data;
    return cudaSuccess;
}
cudaError_t qoco_gpu_audit_download_async(QocoGpuAudit* w, cudaStream_t stream, QocoAuditResult* output) {
    if (!w || !output) return cudaErrorInvalidValue;
    const auto status = cudaMemcpyAsync(output, w->result.data, sizeof(*output), cudaMemcpyDeviceToHost, stream);
    if (status != cudaSuccess) return status;
    ++w->transfers.d2h_count; w->transfers.d2h_bytes += sizeof(*output);
    return cudaSuccess;
}
namespace {
__global__ void compact_replay_status(const int* header, QocoReplayStatus* output) {
    *output=header[0]==1 ? QocoReplayStatus{header[1],header[2]} : QocoReplayStatus{-1,0};
}
}
cudaError_t qoco_gpu_audit_replay_status(QocoGpuAudit* w, const int* header,
    cudaStream_t stream, const QocoReplayStatus** output) {
    if (output) *output=nullptr;
    if (!w || !header || !output) return cudaErrorInvalidValue;
    compact_replay_status<<<1,1,0,stream>>>(header,w->replay_status.data);
    const auto status=cudaGetLastError();
    if (status==cudaSuccess) *output=w->replay_status.data;
    return status;
}
cudaError_t qoco_gpu_audit_run(QocoGpuAudit* w, const double* x, const double* y, const double* z,
                             double* mapped, cudaStream_t stream, QocoAuditResult* output) {
    if (!output) return cudaErrorInvalidValue;
    const QocoAuditResult* device = nullptr;
    auto status = qoco_gpu_audit_run_device(w,x,y,z,mapped,stream,&device);
    if (status != cudaSuccess) return status;
    status = qoco_gpu_audit_download_async(w,stream,output);
    if (status != cudaSuccess) return status;
    return cudaStreamSynchronize(stream);
}
QocoAuditTransfers qoco_gpu_audit_transfers(const QocoGpuAudit* w) { return w ? w->transfers : QocoAuditTransfers{}; }
QocoAuditMemory qoco_gpu_audit_memory(const QocoGpuAudit* w) { return w ? w->memory : QocoAuditMemory{}; }
void qoco_gpu_audit_destroy(QocoGpuAudit* w) { delete w; }

struct QocoGpuTopology {
    QocoAuditMemory memory{};
    QocoAuditTransfers transfers{};
    Buffer<int> arrays[6], mismatch;
    QocoTopologyInput expected{};
    int maximum_count{};
};

namespace {
__global__ void validate_topology(QocoTopologyInput expected, QocoTopologyInput actual,
                                  int* mismatch) {
    const int array = blockIdx.y;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < expected.counts[array];
         i += gridDim.x * blockDim.x) {
        if (expected.arrays[array][i] != actual.arrays[array][i]) atomicExch(mismatch, 1);
    }
}
}

cudaError_t qoco_gpu_topology_create(const QocoTopologyInput& input, cudaStream_t stream,
                                    QocoGpuTopology** output) {
    if (!output) return cudaErrorInvalidValue;
    *output = nullptr;
    try {
        auto result = std::make_unique<QocoGpuTopology>();
        for (int i = 0; i < 6; ++i) {
            if (input.counts[i] < 0 || (input.counts[i] && !input.arrays[i]))
                return cudaErrorInvalidValue;
            result->arrays[i].allocate(input.counts[i], result->memory);
            upload(result->arrays[i].data, input.arrays[i], input.counts[i], stream, result->transfers);
            result->expected.counts[i] = input.counts[i];
            result->expected.arrays[i] = result->arrays[i].data;
            result->maximum_count = std::max(result->maximum_count, input.counts[i]);
        }
        result->mismatch.allocate(1, result->memory);
        check(cudaStreamSynchronize(stream));
        *output = result.release();
        return cudaSuccess;
    } catch (const Failure& e) { return e.status; }
      catch (const std::bad_alloc&) { return cudaErrorMemoryAllocation; }
}

cudaError_t qoco_gpu_topology_validate(QocoGpuTopology* cache, const QocoTopologyInput& input,
                                      cudaStream_t stream, bool* match) {
    if (!cache || !match) return cudaErrorInvalidValue;
    *match = false;
    for (int i = 0; i < 6; ++i)
        if (input.counts[i] != cache->expected.counts[i] || (input.counts[i] && !input.arrays[i]))
            return cudaErrorInvalidValue;
    try {
        check(cudaMemsetAsync(cache->mismatch.data, 0, sizeof(int), stream));
        if (cache->maximum_count) {
            const int blocks = std::min(256, (cache->maximum_count - 1) / 256 + 1);
            validate_topology<<<dim3(blocks, 6), 256, 0, stream>>>(cache->expected, input, cache->mismatch.data);
            check(cudaGetLastError());
        }
        int mismatch = 0;
        check(cudaMemcpyAsync(&mismatch, cache->mismatch.data, sizeof(int), cudaMemcpyDeviceToHost, stream));
        ++cache->transfers.d2h_count; cache->transfers.d2h_bytes += sizeof(int);
        check(cudaStreamSynchronize(stream));
        *match = mismatch == 0;
        return cudaSuccess;
    } catch (const Failure& e) { return e.status; }
}
QocoAuditTransfers qoco_gpu_topology_transfers(const QocoGpuTopology* cache) {
    return cache ? cache->transfers : QocoAuditTransfers{};
}
QocoAuditMemory qoco_gpu_topology_memory(const QocoGpuTopology* cache) {
    return cache ? cache->memory : QocoAuditMemory{};
}
void qoco_gpu_topology_destroy(QocoGpuTopology* cache) { delete cache; }

struct QocoGpuConversion {
    QocoAuditMemory memory{};
    QocoAuditTransfers transfers{};
    QocoConversionPlan plan{};
    Buffer<int> offsets, bound_types, invalid;
    Buffer<QocoConversionTerm> entries;
    Buffer<QocoConversionPair> symmetry;
    Buffer<double> values;
    int maximum_input{};
};

namespace {
__global__ void conversion_validate(QocoConversionPlan plan, QocoConversionInputs in, int* invalid) {
    const int array = blockIdx.y;
    for (std::int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < plan.input_counts[array];
         i += gridDim.x * blockDim.x) {
        const double value = in.arrays[array][i];
        const bool bound = array == 4 || array == 5 || array == 7 || array == 8;
        if (isnan(value) || (!bound && !isfinite(value))) atomicOr(invalid, 2);
        if (array == 4 || array == 7) {
            const double upper = in.arrays[array + 1][i];
            const int kind = isfinite(value) && isfinite(upper) && value == upper ? 4
                : (isfinite(value) ? 2 : 0) + (isfinite(upper) ? 1 : 0);
            const int expected = plan.bound_types[(array == 7 ? plan.input_counts[4] : 0) + i];
            if (kind != expected) atomicOr(invalid, 1);
        }
    }
}
__global__ void conversion_symmetry(QocoConversionPlan plan, const double* values, int* invalid) {
    for (std::int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < plan.symmetry_pairs;
         i += gridDim.x * blockDim.x) {
        const auto pair = plan.symmetry[i];
        const double a = values[pair.first], b = values[pair.second];
        if (fabs(a - b) > 1e-12 * fmax(1.0, fmax(fabs(a), fabs(b)))) atomicOr(invalid, 4);
    }
}
__global__ void conversion_gather(QocoConversionPlan plan, QocoConversionInputs in,
                                  double* output, int* invalid) {
    for (std::int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < plan.outputs;
         i += gridDim.x * blockDim.x) {
        double sum = 0;
        for (int k = plan.offsets[i]; k < plan.offsets[i + 1]; ++k) {
            const auto term = plan.entries[k];
            double value = term.scale;
            if (term.input >= 0) {
                value = in.arrays[term.input][term.index];
                if (term.other >= 0)
                    value = __dadd_rn(value, __dmul_rn(term.other_scale, in.arrays[term.input][term.other]));
                value = __dmul_rn(term.scale, value);
            }
            sum = __dadd_rn(sum, value);
        }
        if (!isfinite(sum)) atomicOr(invalid, 2);
        output[i] = sum;
    }
}
}

cudaError_t qoco_gpu_conversion_create(const QocoConversionPlan& in, cudaStream_t stream,
                                        QocoGpuConversion** output) {
    if (!output) return cudaErrorInvalidValue;
    *output = nullptr;
    if (in.outputs < 0 || in.outputs == INT_MAX || in.terms < 0 || in.symmetry_pairs < 0
        || !in.offsets || in.offsets[0] != 0 || in.offsets[in.outputs] != in.terms
        || (in.terms && !in.entries) || (in.symmetry_pairs && !in.symmetry)) return cudaErrorInvalidValue;
    for (int count : in.input_counts) if (count < 0) return cudaErrorInvalidValue;
    if (in.input_counts[4] != in.input_counts[5] || in.input_counts[7] != in.input_counts[8]
        || static_cast<std::int64_t>(in.input_counts[4]) + in.input_counts[7] > INT_MAX
        || ((in.input_counts[4] || in.input_counts[7]) && !in.bound_types)) return cudaErrorInvalidValue;
    for (int i = 0; i < in.outputs; ++i)
        if (in.offsets[i] < 0 || in.offsets[i] > in.offsets[i + 1]) return cudaErrorInvalidValue;
    for (int i = 0; i < in.terms; ++i) {
        const auto& t = in.entries[i];
        if (t.input < -1 || t.input >= 9 || !std::isfinite(t.scale) || !std::isfinite(t.other_scale)
            || (t.input >= 0 && (t.index < 0 || t.index >= in.input_counts[t.input]
                || t.other < -1 || t.other >= in.input_counts[t.input]))) return cudaErrorInvalidValue;
    }
    for (int i = 0; i < in.symmetry_pairs; ++i)
        if (in.symmetry[i].first < 0 || in.symmetry[i].first >= in.input_counts[0]
            || in.symmetry[i].second < 0 || in.symmetry[i].second >= in.input_counts[0]) return cudaErrorInvalidValue;
    const int bounds = in.input_counts[4] + in.input_counts[7];
    for (int i = 0; i < bounds; ++i)
        if (in.bound_types[i] < 0 || in.bound_types[i] > 4) return cudaErrorInvalidValue;
    try {
        auto w = std::make_unique<QocoGpuConversion>();
        w->plan = in;
        w->offsets.allocate(static_cast<std::size_t>(in.outputs) + 1, w->memory);
        w->entries.allocate(in.terms, w->memory); w->symmetry.allocate(in.symmetry_pairs, w->memory);
        w->bound_types.allocate(bounds, w->memory); w->invalid.allocate(1, w->memory);
        w->values.allocate(in.outputs, w->memory);
        upload(w->offsets.data, in.offsets, static_cast<std::size_t>(in.outputs) + 1, stream, w->transfers);
        upload(w->entries.data, in.entries, in.terms, stream, w->transfers);
        upload(w->symmetry.data, in.symmetry, in.symmetry_pairs, stream, w->transfers);
        upload(w->bound_types.data, in.bound_types, bounds, stream, w->transfers);
        w->plan.offsets = w->offsets.data; w->plan.entries = w->entries.data;
        w->plan.symmetry = w->symmetry.data; w->plan.bound_types = w->bound_types.data;
        for (int count : in.input_counts) w->maximum_input = std::max(w->maximum_input, count);
        check(cudaStreamSynchronize(stream));
        *output = w.release();
        return cudaSuccess;
    } catch (const Failure& e) { return e.status; }
      catch (const std::bad_alloc&) { return cudaErrorMemoryAllocation; }
}

cudaError_t qoco_gpu_conversion_run(QocoGpuConversion* w, const QocoConversionInputs& in,
                                    double* host_output, int* invalid, cudaStream_t stream) {
    if (!w || !invalid) return cudaErrorInvalidValue;
    for (int i = 0; i < 9; ++i) if (w->plan.input_counts[i] && !in.arrays[i]) return cudaErrorInvalidValue;
    try {
        check(cudaMemsetAsync(w->invalid.data, 0, sizeof(int), stream));
        if (w->maximum_input)
            conversion_validate<<<dim3(std::min(256, (w->maximum_input - 1) / 256 + 1), 9), 256, 0, stream>>>(w->plan, in, w->invalid.data);
        if (w->plan.symmetry_pairs)
            conversion_symmetry<<<std::min(256, (w->plan.symmetry_pairs - 1) / 256 + 1), 256, 0, stream>>>(w->plan, in.arrays[0], w->invalid.data);
        if (w->plan.outputs)
            conversion_gather<<<std::min(256, (w->plan.outputs - 1) / 256 + 1), 256, 0, stream>>>(w->plan, in, w->values.data, w->invalid.data);
        check(cudaGetLastError());
        check(cudaMemcpyAsync(invalid, w->invalid.data, sizeof(int), cudaMemcpyDeviceToHost, stream));
        ++w->transfers.d2h_count; w->transfers.d2h_bytes += sizeof(int);
        if (w->plan.outputs && host_output) {
            check(cudaMemcpyAsync(host_output, w->values.data, w->plan.outputs * sizeof(double), cudaMemcpyDeviceToHost, stream));
            ++w->transfers.d2h_count; w->transfers.d2h_bytes += w->plan.outputs * sizeof(double);
        }
        check(cudaStreamSynchronize(stream));
        return cudaSuccess;
    } catch (const Failure& e) { cudaStreamSynchronize(stream); return e.status; }
}
const double* qoco_gpu_conversion_values(const QocoGpuConversion* w) { return w ? w->values.data : nullptr; }
QocoAuditTransfers qoco_gpu_conversion_transfers(const QocoGpuConversion* w) { return w ? w->transfers : QocoAuditTransfers{}; }
QocoAuditMemory qoco_gpu_conversion_memory(const QocoGpuConversion* w) { return w ? w->memory : QocoAuditMemory{}; }
void qoco_gpu_conversion_destroy(QocoGpuConversion* w) { delete w; }
