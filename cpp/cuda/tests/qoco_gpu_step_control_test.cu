// Standalone QOCO CUDA test. Cone feasibility is checked by independent bisection;
// no factorization is used, so racecheck covers only our operators and reductions.
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "../algebra/cuda/cuda_types.h"
extern "C" {
#include "cone.h"
void qoco_gpu_compute_centering(QOCOSolver*);
void qoco_gpu_take_step(QOCOSolver*);
int qoco_gpu_begin_reduction_scope();
void qoco_gpu_end_reduction_scope();
}
bool load_cuda_libraries();
static void require(bool ok, const char* message) {
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", message); std::exit(1); }
}
static void check(cudaError_t code) { require(code == cudaSuccess, cudaGetErrorString(code)); }
static void near(double got, long double expected, const char* message) {
    require(std::isfinite(got) && std::abs(got - expected) <= 3e-12L * (1 + std::abs(expected)), message);
}
// Offset device pointers and canaries catch writes outside each logical vector.
struct Vector {
    QOCOVectorf v{};
    double* allocation{};
    std::vector<double> initial;
    explicit Vector(std::vector<double> values) : initial(std::move(values)) {
        std::vector<double> padded(initial.size() + 2, 12345.0);
        std::copy(initial.begin(), initial.end(), padded.begin() + 1);
        check(cudaMalloc(&allocation, padded.size() * sizeof(double)));
        check(cudaMemcpy(allocation, padded.data(), padded.size() * sizeof(double), cudaMemcpyHostToDevice));
        v.d_data = allocation + 1; v.len = initial.size(); v.data = initial.data();
    }
    std::vector<double> read() {
        std::vector<double> padded(initial.size() + 2);
        check(cudaMemcpy(padded.data(), allocation, padded.size() * sizeof(double), cudaMemcpyDeviceToHost));
        require(padded.front() == 12345.0 && padded.back() == 12345.0, "vector guard overwritten");
        return {padded.begin() + 1, padded.end() - 1};
    }
    ~Vector() { check(cudaFree(allocation)); }
};
static void run(int lp, const std::vector<int>& sizes, int edge = 0) {
    int m = lp;
    std::vector<int> starts;
    for (int size : sizes) { starts.push_back(m); m += size; }
    std::vector<double> s(m), z(m), ds(m), dz(m);
    for (int i = 0; i < m; ++i) {
        s[i] = 1.2 + .1 * std::sin(i); z[i] = 1.4 + .1 * std::cos(i);
        ds[i] = 3 * std::cos(.7 * i); dz[i] = 2 * std::sin(.3 * i);
    }
    for (int j = 0; j < static_cast<int>(sizes.size()); ++j) {
        long double sn = 0, zn = 0;
        for (int k = 1; k < sizes[j]; ++k) {
            sn += s[starts[j] + k] * s[starts[j] + k];
            zn += z[starts[j] + k] * z[starts[j] + k];
        }
        s[starts[j]] = std::sqrt(sn) + .5; z[starts[j]] = std::sqrt(zn) + .8;
    }
    if (edge > 0 && edge <= 6) {
        s = (edge == 1 || edge == 5) ? std::vector<double>{1, 0, 0} : std::vector<double>{1, 1, 0};
        if (edge == 1) ds = {-1, 1, 0};
        if (edge == 2) ds = {-2, -1, 0};
        if (edge == 3) ds = {0, 1, 0};
        if (edge == 4) ds = {1, -1, 0};
        if (edge == 5) ds = {-1, 1 + 2e-15, 0};
        if (edge == 6) ds = {0, -3, 0};
        z = {2, 0, 0}; dz = {0, 0, 0};
    }
    if (edge == 7) { s.back() = 1e-15; ds.back() = -1; }
    auto violation = [&](const auto& u, const auto& du, long double a) {
        long double result = -1e7;
        for (int i = 0; i < lp; ++i) result = std::max(result, -(u[i] + a * du[i]));
        for (int j = 0; j < static_cast<int>(sizes.size()); ++j) {
            long double norm = 0;
            for (int k = 1; k < sizes[j]; ++k) {
                long double value = u[starts[j] + k] + a * du[starts[j] + k]; norm += value * value;
            }
            result = std::max(result, std::sqrt(norm) - (u[starts[j]] + a * du[starts[j]]));
        }
        return result;
    };
    auto bound = [&](const auto& u, const auto& du) {
        long double lo = 0, hi = 1;
        if (violation(u, du, hi) <= 0) lo = hi;
        else for (int j = 0; j < 90; ++j) {
            const auto mid = (lo + hi) / 2;
            if (violation(u, du, mid) <= 0) lo = mid; else hi = mid;
        }
        return lo < 1e-12L ? 0.L : lo;
    };
    const long double alpha = std::min(bound(s, ds), bound(z, dz));
    const int n = m ? 517 : 17, p = m ? 263 : 0;
    std::vector<double> x(n), y(p), xyz(n + p + m);
    for (int i = 0; i < n; ++i) { x[i] = std::sin(i); xyz[i] = std::cos(i); }
    for (int i = 0; i < p; ++i) { y[i] = std::cos(i); xyz[n + i] = std::sin(i); }
    std::copy(dz.begin(), dz.end(), xyz.begin() + n + p);
    Vector vx(x), vy(y), vs(s), vz(z), vds(ds), vxyz(xyz), u1(std::vector<double>(m, 0)), u2(std::vector<double>(m, 0));
    QOCOProblemData data{}; data.n = n; data.p = p; data.m = m; data.l = lp; data.nsoc = sizes.size();
    data.q = new_qoco_vectori(sizes.data(), data.nsoc);
    QOCOWorkspace work{}; work.data = &data;
    work.soc_idx = new_qoco_vectori(starts.data(), data.nsoc);
    work.x = &vx.v; work.y = &vy.v; work.s = &vs.v; work.z = &vz.v;
    work.Ds = &vds.v; work.xyz = &vxyz.v; work.ubuff1 = &u1.v; work.ubuff2 = &u2.v;
    QOCOSolver solver{}; solver.work = &work;
    qoco_gpu_compute_centering(&solver);
    long double numerator = 0, denominator = 0;
    for (int i = 0; i < m; ++i) {
        numerator += (z[i] + alpha * dz[i]) * (s[i] + alpha * ds[i]);
        denominator += static_cast<long double>(z[i]) * s[i];
    }
    if (m) {
        const auto rho = std::max(0.L, std::min(1.L, numerator / denominator));
        near(work.sigma, rho * rho * rho, "centering versus independent arithmetic");
    } else require(std::isnan(work.sigma), "empty centering preserves reference NaN");
    const auto a = alpha * .99L;
    qoco_gpu_take_step(&solver);
    near(work.a, a, "step versus independent feasibility");
    const auto rx = vx.read(), ry = vy.read(), rs = vs.read(), rz = vz.read();
    for (int i = 0; i < n; ++i) near(rx[i], x[i] + a * xyz[i], "x update");
    for (int i = 0; i < p; ++i) near(ry[i], y[i] + a * xyz[n + i], "y update");
    for (int i = 0; i < m; ++i) {
        near(rs[i], s[i] + a * ds[i], "s update"); near(rz[i], z[i] + a * dz[i], "z update");
    }
    require(violation(rs, ds, 0) < 3e-12 && violation(rz, dz, 0) < 3e-12, "updated cone feasibility");
    require(vxyz.read() == xyz && vds.read() == ds, "input directions changed");
    u1.read(); u2.read();
    double host[]{2, 3}; near(qoco_dot(host, host, 2), 13, "host dot after pointer mode restoration");
    free_qoco_vectori(work.soc_idx); free_qoco_vectori(data.q);
}
int main() {
    require(load_cuda_libraries(), "load CUDA libraries");
    setenv("SPACEPDHCG_TEST_QOCO_DEVICE_STEPS_COMPARE", "1", 1);
    for (int scoped = 0; scoped < 2; ++scoped) {
        if (scoped) { require(qoco_gpu_begin_reduction_scope() == 0, "begin scope"); require(qoco_gpu_begin_reduction_scope() == 0, "nested scope"); }
        run(0, {}); run(13, {}); run(0, {3, 5, 33});
        run(8193, std::vector<int>(1027, 4)); run(262145, {});
        for (int edge = 1; edge <= 6; ++edge) run(0, {3}, edge);
        run(13, {}, 7);
        if (scoped) { qoco_gpu_end_reduction_scope(); qoco_gpu_end_reduction_scope(); }
    }
    check(cudaDeviceSynchronize());
    std::puts("QOCO GPU step control: centering, independent feasibility, fused updates, guards, scopes PASS");
}
