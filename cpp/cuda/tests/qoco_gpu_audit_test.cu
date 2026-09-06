#include "../internal/native_qoco_gpu.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <random>
#include <vector>

namespace {
void require(bool ok, const char* what) { if (!ok) { std::fprintf(stderr, "FAIL: %s\n", what); std::exit(1); } }
void check(cudaError_t status) { require(status == cudaSuccess, cudaGetErrorString(status)); }
struct Dense {
    int rows, cols;
    std::vector<double> values;
    Dense(int r, int c) : rows(r), cols(c), values(static_cast<std::size_t>(r) * c) {}
    double& at(int r, int c) { return values[static_cast<std::size_t>(r) * cols + c]; }
    double at(int r, int c) const { return values[static_cast<std::size_t>(r) * cols + c]; }
    std::vector<long double> product(const std::vector<double>& x) const {
        std::vector<long double> result(rows);
        for (int r = 0; r < rows; ++r) for (int c = 0; c < cols; ++c)
            result[r] += static_cast<long double>(at(r, c)) * x[c];
        return result;
    }
};
struct Csc {
    int rows, cols;
    std::vector<int> offsets, indices;
    std::vector<double> values;
    Csc(const Dense& a, bool upper = false) : rows(a.rows), cols(a.cols), offsets(a.cols + 1) {
        for (int c = 0; c < cols; ++c) {
            offsets[c] = static_cast<int>(values.size());
            for (int r = 0; r < rows; ++r) if ((!upper || r <= c) && a.at(r, c) != 0) {
                // Duplicate entries exercise ordered summation, while preserving
                // exactly the dense matrix used by the independent oracle.
                if ((r + c) % 13 == 0) { indices.push_back(r); values.push_back(a.at(r, c) / 2); }
                indices.push_back(r); values.push_back(a.at(r, c) * ((r + c) % 13 == 0 ? 0.5 : 1.0));
            }
        }
        offsets[cols] = static_cast<int>(values.size());
    }
    QocoAuditCsc view() const { return {rows, cols, static_cast<int>(values.size()), offsets.data(), indices.data(), values.data()}; }
};

QocoAuditResult reference(const Dense& p, const Dense& a, const Dense& g,
    const std::vector<double>& c, const std::vector<double>& b, const std::vector<double>& h,
    int nonnegative, const std::vector<int>& cones, const std::vector<double>& x,
    const std::vector<double>& y, const std::vector<double>& z) {
    auto px = p.product(x), ax = a.product(x), gx = g.product(x);
    long double primal = 0, dual = 0, dual_cone = 0, comp = 0;
    long double max_rhs = 0, max_ax = 0, max_c = 0, max_px = 0, max_aty = 0, pobj = 0, dobj = 0;
    std::vector<long double> slack(g.rows);
    for (int r = 0; r < a.rows; ++r) {
        primal = std::max(primal, std::abs(ax[r] - b[r]));
        max_rhs = std::max(max_rhs, std::abs(static_cast<long double>(b[r])));
        max_ax = std::max(max_ax, std::abs(ax[r])); dobj += static_cast<long double>(b[r]) * y[r];
    }
    for (int r = 0; r < g.rows; ++r) {
        slack[r] = h[r] - gx[r]; max_ax = std::max(max_ax, std::abs(gx[r]));
        max_rhs = std::max(max_rhs, std::abs(static_cast<long double>(h[r])));
        dobj += static_cast<long double>(h[r]) * z[r];
        if (r < nonnegative) { primal = std::max(primal, -slack[r]);
            dual_cone = std::max(dual_cone, -static_cast<long double>(z[r]));
            comp = std::max(comp, std::abs(slack[r] * z[r])); }
    }
    int start = nonnegative;
    for (int size : cones) {
        long double ns = 0, nz = 0, inner = 0;
        for (int k = 0; k < size; ++k) { inner += slack[start + k] * z[start + k];
            if (k) { ns += slack[start + k] * slack[start + k]; nz += static_cast<long double>(z[start + k]) * z[start + k]; } }
        primal = std::max(primal, std::sqrt(ns) - slack[start]);
        dual_cone = std::max(dual_cone, std::sqrt(nz) - z[start]); comp = std::max(comp, std::abs(inner)); start += size;
    }
    for (int col = 0; col < p.cols; ++col) {
        long double aty = 0;
        for (int row = 0; row < a.rows; ++row) aty += static_cast<long double>(a.at(row, col)) * y[row];
        for (int row = 0; row < g.rows; ++row) aty += static_cast<long double>(g.at(row, col)) * z[row];
        dual = std::max(dual, std::abs(c[col] + px[col] + aty));
        max_c = std::max(max_c, std::abs(static_cast<long double>(c[col])));
        max_px = std::max(max_px, std::abs(px[col])); max_aty = std::max(max_aty, std::abs(aty));
        pobj += (c[col] + 0.5L * px[col]) * x[col];
    }
    long double ps = 1 + max_rhs + max_ax, ds = 1 + max_c + max_px + max_aty, gs = 1 + std::abs(pobj) + std::abs(dobj);
    return {static_cast<double>(std::max(primal, dual_cone) / ps), static_cast<double>(std::max(dual / ds, comp / gs)),
        static_cast<double>(primal), static_cast<double>(dual), static_cast<double>(dual_cone / ps), static_cast<double>(comp / gs)};
}
void equal(const QocoAuditResult& a, const QocoAuditResult& b) {
    const double x[]{a.primal, a.dual, a.absolute_primal, a.absolute_dual, a.dual_cone, a.complementarity};
    const double y[]{b.primal, b.dual, b.absolute_primal, b.absolute_dual, b.dual_cone, b.complementarity};
    for (int i = 0; i < 6; ++i) require(std::isfinite(x[i]) && std::abs(x[i] - y[i]) < 2e-12 * (1 + std::abs(y[i])), "GPU audit differs from independent dense arithmetic");
}

void run_case(int n, int equalities, int nonnegative, const std::vector<int>& cones) {
    int m = nonnegative; for (int size : cones) m += size;
    Dense p(n, n), a(equalities, n), g(m, n), mapping(equalities + m, equalities + m);
    std::mt19937 random(42 + n);
    std::uniform_real_distribution<double> sample(-1, 1);
    for (int i = 0; i < n; ++i) {
        p.at(i, i) = 2.0;
        if (i + 1 < n) p.at(i, i + 1) = p.at(i + 1, i) = sample(random) * 0.1;
    }
    for (int row = 0; row < equalities; ++row) for (int col = 0; col < n; ++col)
        if ((row * 17 + col) % 11 == 0) a.at(row, col) = sample(random);
    for (int row = 0; row < m; ++row) for (int col = 0; col < n; ++col)
        if ((row * 23 + col) % 13 == 0) g.at(row, col) = sample(random);
    for (int row = 0; row < mapping.rows; ++row) {
        mapping.at(row, row) = row % 2 ? -1.0 : 1.0;
        if (row + 1 < mapping.cols) mapping.at(row, row + 1) = 1.0 / std::sqrt(2.0);
    }
    std::vector<double> c(n), b(equalities), h(m), x(n), y(equalities), z(m);
    for (auto* vector : {&c, &b, &h, &x, &y, &z}) for (double& value : *vector) value = sample(random);
    Csc cp(p, true), ca(a), cg(g), map(mapping);
    QocoAuditInput input{cp.view(), ca.view(), cg.view(), map.view(), c.data(), b.data(), h.data(), nonnegative, static_cast<int>(cones.size()), cones.data()};
    cudaStream_t stream{}; check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    QocoGpuAudit* audit{}; check(qoco_gpu_audit_create(input, true, stream, &audit));
    const auto memory = qoco_gpu_audit_memory(audit);
    require(memory.allocations > 0 && memory.bytes > 0 && memory.peak_bytes >= memory.bytes, "audit memory accounted");
    double* mapped{}; if (mapping.rows) check(cudaMalloc(&mapped, mapping.rows * sizeof(double)));
    const double *dx{}, *dy{}, *dz{};
    for (int update = 0; update < 3; ++update) {
        if (update) {
            for (double& value : cp.values) value *= 0.7;
            for (double& value : p.values) value *= 0.7;
            for (double& value : ca.values) value *= -0.5;
            for (double& value : a.values) value *= -0.5;
            for (double& value : cg.values) value *= 1.1;
            for (double& value : g.values) value *= 1.1;
            for (double& value : c) value += 0.2;
            check(qoco_gpu_audit_update(audit, input, stream));
        }
        check(qoco_gpu_audit_upload_solution(audit, x.data(), y.data(), z.data(), stream, &dx, &dy, &dz));
        QocoAuditResult result{}; check(qoco_gpu_audit_run(audit, dx, dy, dz, mapped, stream, &result));
        equal(result, reference(p, a, g, c, b, h, nonnegative, cones, x, y, z));
        const auto first = result;
        const auto transfers=qoco_gpu_audit_transfers(audit);
        const QocoAuditResult* device_result{};
        cudaGraph_t graph{}; cudaGraphExec_t executable{};
        check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
        check(qoco_gpu_audit_run_device(audit,dx,dy,dz,mapped,stream,&device_result));
        check(cudaStreamEndCapture(stream,&graph));
        check(cudaGraphInstantiate(&executable,graph,nullptr,nullptr,0));
        check(cudaGraphLaunch(executable,stream));
        check(cudaMemcpyAsync(&result,device_result,sizeof(result),cudaMemcpyDeviceToHost,stream));
        check(cudaStreamSynchronize(stream));
        equal(result,first);
        const auto after_device=qoco_gpu_audit_transfers(audit);
        require(after_device.d2h_count==transfers.d2h_count && after_device.d2h_bytes==transfers.d2h_bytes,
            "device audit queues no internal download");
        check(cudaGraphExecDestroy(executable)); check(cudaGraphDestroy(graph));
        for (int repeat = 0; repeat < 4; ++repeat) {
            check(qoco_gpu_audit_run(audit, dx, dy, dz, mapped, stream, &result));
            require(result.primal == first.primal && result.dual == first.dual, "GPU audit reproducibility");
        }
        std::vector<double> packed(y); packed.insert(packed.end(), z.begin(), z.end());
        auto expected = mapping.product(packed);
        std::vector<double> actual(mapping.rows);
        if (mapping.rows) check(cudaMemcpy(actual.data(), mapped, actual.size() * sizeof(double), cudaMemcpyDeviceToHost));
        for (int i = 0; i < mapping.rows; ++i) require(std::abs(actual[i] - expected[i]) < 1e-12L, "dual mapping accuracy");
    }
    // A non-finite unused vector component or coefficient must never disappear
    // through fmax and produce an apparently qualified finite certificate.
    x[0] = std::numeric_limits<double>::quiet_NaN();
    check(qoco_gpu_audit_upload_solution(audit, x.data(), y.data(), z.data(), stream, &dx, &dy, &dz));
    QocoAuditResult invalid{}; check(qoco_gpu_audit_run(audit, dx, dy, dz, mapped, stream, &invalid));
    require(std::isinf(invalid.primal) && std::isinf(invalid.dual), "nonfinite primal rejected");
    x[0] = 0.0; cp.values[0] = std::numeric_limits<double>::infinity();
    check(qoco_gpu_audit_update(audit, input, stream));
    check(qoco_gpu_audit_upload_solution(audit, x.data(), y.data(), z.data(), stream, &dx, &dy, &dz));
    check(qoco_gpu_audit_run(audit, dx, dy, dz, mapped, stream, &invalid));
    require(std::isinf(invalid.primal) && std::isinf(invalid.dual), "nonfinite coefficient rejected");
    const auto after = qoco_gpu_audit_memory(audit);
    require(after.allocations == memory.allocations && after.bytes == memory.bytes
        && after.peak_bytes == memory.peak_bytes, "updates and audit must reuse allocations");
    qoco_gpu_audit_destroy(audit); if (mapped) check(cudaFree(mapped)); check(cudaStreamDestroy(stream));
}
void topology_case() {
    cudaStream_t stream{}; check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    QocoTopologyInput host{}, device{};
    const int counts[]{0, 1, 257, 513, 4097, 131073};
    std::vector<int> expected[6];
    int* allocated[6]{};
    for (int i = 0; i < 6; ++i) {
        expected[i].resize(counts[i]);
        for (int j = 0; j < counts[i]; ++j) expected[i][j] = j % 123 + i;
        host.counts[i] = device.counts[i] = counts[i];
        host.arrays[i] = expected[i].data();
        if (counts[i]) {
            check(cudaMalloc(&allocated[i], counts[i] * sizeof(int)));
            check(cudaMemcpyAsync(allocated[i], expected[i].data(), counts[i] * sizeof(int), cudaMemcpyHostToDevice, stream));
        }
        device.arrays[i] = allocated[i];
    }
    QocoGpuTopology* cache{}; check(qoco_gpu_topology_create(host, stream, &cache));
    const auto before = qoco_gpu_topology_memory(cache);
    bool matches{}; check(qoco_gpu_topology_validate(cache, device, stream, &matches));
    require(matches, "unchanged topology matches");
    for (int i = 1; i < 6; ++i) {
        const int changed = -1;
        check(cudaMemcpyAsync(allocated[i] + counts[i] - 1, &changed, sizeof(int), cudaMemcpyHostToDevice, stream));
        check(qoco_gpu_topology_validate(cache, device, stream, &matches));
        require(!matches, "in-place topology mutation must be rejected, including grid-stride tail");
        check(cudaMemcpyAsync(allocated[i] + counts[i] - 1, &expected[i].back(), sizeof(int), cudaMemcpyHostToDevice, stream));
        check(qoco_gpu_topology_validate(cache, device, stream, &matches));
        require(matches, "restored topology matches");
    }
    int* replacement{}; check(cudaMalloc(&replacement, counts[2] * sizeof(int)));
    check(cudaMemcpyAsync(replacement, expected[2].data(), counts[2] * sizeof(int), cudaMemcpyHostToDevice, stream));
    device.arrays[2] = replacement;
    check(qoco_gpu_topology_validate(cache, device, stream, &matches));
    require(matches, "equal topology at a different device address matches");
    ++device.counts[2];
    require(qoco_gpu_topology_validate(cache, device, stream, &matches) == cudaErrorInvalidValue && !matches,
        "changed topology dimensions rejected");
    const auto after = qoco_gpu_topology_memory(cache);
    const auto transfers = qoco_gpu_topology_transfers(cache);
    require(before.allocations == after.allocations && before.peak_bytes == after.peak_bytes,
        "topology validation must not allocate");
    require(transfers.d2h_count == 12 && transfers.d2h_bytes == 12 * sizeof(int),
        "topology validation downloads only one flag per successful invocation");
    qoco_gpu_topology_destroy(cache);
    check(cudaFree(replacement));
    for (auto* pointer : allocated) if (pointer) check(cudaFree(pointer));
    check(cudaStreamDestroy(stream));
    std::puts("GPU topology cache: exact mutation detection, stream ordering, retained buffers PASS");
}
} // namespace

int main() {
    topology_case();
    run_case(1, 0, 0, {});
    run_case(7, 0, 3, {4, 8});
    run_case(17, 9, 2, {3, 5});
    run_case(513, 201, 7, {4, 32, 257});
    std::puts("GPU QOCO audit: independent dense reference, updates, mapping, stream, nonfinite rejection PASS");
}
