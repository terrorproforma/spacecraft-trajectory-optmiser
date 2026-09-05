// Independent CPU arithmetic plus the previous stopping-metric implementation.
#include "qoco.h"
#include "../algebra/cuda/cuda_types.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <dlfcn.h>
#include <vector>

namespace {
void require(bool yes, const char* message) { if (!yes) { std::fprintf(stderr, "FAIL: %s\n", message); std::exit(1); } }
void check(cudaError_t status) { require(status == cudaSuccess, cudaGetErrorString(status)); }
void upload(double* out, const std::vector<double>& in) { if (!in.empty()) check(cudaMemcpy(out, in.data(), in.size() * sizeof(double), cudaMemcpyHostToDevice)); }
void compare(double actual, double expected, const char* label) {
    if (!std::isfinite(actual) || !std::isfinite(expected) || std::abs(actual - expected) > 3e-11 * std::max(1.0, std::abs(expected))) {
        std::fprintf(stderr, "%s: %.17g != %.17g\n", label, actual, expected); std::exit(1);
    }
}
struct Sparse {
    int rows, cols;
    std::vector<int> offsets{0}, indices;
    std::vector<double> values;
    QOCOCscMatrix view() { return {rows, cols, static_cast<int>(values.size()), indices.data(), offsets.data(), values.data()}; }
    std::vector<long double> product(const std::vector<double>& x, bool transpose = false, bool symmetric = false) const {
        std::vector<long double> out(transpose ? cols : rows, 0.0L);
        for (int col = 0; col < cols; ++col) for (int k = offsets[col]; k < offsets[col + 1]; ++k) {
            const int row = indices[k]; const long double value = values[k];
            if (transpose) out[col] += value * x[row];
            else { out[row] += value * x[col]; if (symmetric && row != col) out[col] += value * x[row]; }
        }
        return out;
    }
};
template<class T> long double norm(const std::vector<T>& x, const std::vector<double>& scale) {
    long double out = 0; for (size_t i = 0; i < x.size(); ++i) out = std::max(out, std::abs(static_cast<long double>(x[i]) * scale[i])); return out;
}
long double dot(const std::vector<double>& a, const std::vector<double>& b) {
    long double out = 0; for (size_t i = 0; i < a.size(); ++i) out += static_cast<long double>(a[i]) * b[i]; return out;
}
void run(int n, bool absent, bool zero_p) {
    const int p = absent ? 0 : 5, m = absent ? 0 : 9;
    Sparse P{n, n}, A{p, n}, G{m, n};
    for (int col = 0; col < n; ++col) {
        if (!zero_p) {
            if (col) { P.indices.push_back(col - 1); P.values.push_back(0.03 * std::cos(col)); }
            P.indices.push_back(col); P.values.push_back(1.0 + 0.1 * (col % 7));
        }
        P.offsets.push_back(P.values.size());
        for (int row = 0; row < p; ++row) { A.indices.push_back(row); A.values.push_back(0.1 * std::cos(row + col * .3)); }
        A.offsets.push_back(A.values.size());
        for (int row = 0; row < m; ++row) { G.indices.push_back(row); G.values.push_back(0.1 * std::sin(row + col * .4)); }
        G.offsets.push_back(G.values.size());
    }
    std::vector<double> c(n), b(p), h(m);
    for (int i = 0; i < n; ++i) c[i] = .2 + std::cos(i * .7);
    for (int i = 0; i < p; ++i) b[i] = .3 + .07 * i;
    for (int i = 0; i < m; ++i) h[i] = .5 + .02 * i;
    auto pc = P.view(), ac = A.view(), gc = G.view();
    int cones[]{3, 4}; QOCOSettings settings{}; set_default_settings(&settings); settings.ruiz_iters = 0; settings.verbose = 0;
    auto* solver = static_cast<QOCOSolver*>(std::calloc(1, sizeof(QOCOSolver)));
    require(qoco_setup(solver, n, m, p, zero_p ? nullptr : &pc, c.data(), p ? &ac : nullptr, b.data(), m ? &gc : nullptr,
                       h.data(), absent ? 0 : 2, absent ? 0 : 2, cones, &settings) == 0, "setup metrics fixture");
    auto* w = solver->work;
    using Metrics = void (*)(QOCOSolver*, double*);
    auto gpu = reinterpret_cast<Metrics>(dlsym(RTLD_DEFAULT, "qoco_gpu_stopping_metrics"));
    auto reference = reinterpret_cast<Metrics>(dlsym(RTLD_DEFAULT, "qoco_reference_stopping_metrics"));
    auto begin = reinterpret_cast<int (*)()>(dlsym(RTLD_DEFAULT, "qoco_gpu_begin_reduction_scope"));
    auto end = reinterpret_cast<void (*)()>(dlsym(RTLD_DEFAULT, "qoco_gpu_end_reduction_scope"));
    require(gpu && reference && begin && end, "complete stopping interface");
    for (int iteration = 0; iteration < 3; ++iteration) {
        std::vector<double> x(n), y(p), z(m), s(m), di(n), ei(p), fi(m), f(m), residual(n + p + m);
        for (int i = 0; i < n; ++i) { x[i] = .4 * std::sin(i * .13 + iteration); di[i] = .7 + .1 * (i % 9); }
        for (int i = 0; i < p; ++i) { y[i] = .3 * std::cos(i + iteration); ei[i] = .6 + .05 * i; }
        for (int i = 0; i < m; ++i) { s[i] = 1.2 + .3 * std::cos(i + iteration); z[i] = .1 + .4 * std::sin(i + iteration); fi[i] = .5 + .04 * i; f[i] = 1.0 / fi[i]; }
        for (int i = 0; i < n + p + m; ++i) residual[i] = .07 * std::sin(i + iteration * .9);
        upload(w->x->d_data, x); upload(w->y->d_data, y); upload(w->z->d_data, z); upload(w->s->d_data, s);
        upload(w->scaling->Dinvruiz->d_data, di); upload(w->scaling->Einvruiz->d_data, ei);
        upload(w->scaling->Finvruiz->d_data, fi); upload(w->scaling->Fruiz->d_data, f); upload(w->kktres->d_data, residual);
        const double kinv = w->scaling->kinv = .37 + iteration * .11;
        auto px = P.product(x, false, true), aty = A.product(y, true), gtz = G.product(z, true), ax = A.product(x), gx = G.product(x);
        long double xpx = 0, gap = 0;
        for (int i = 0; i < n; ++i) xpx += x[i] * px[i] * di[i];
        for (int i = 0; i < m; ++i) gap += static_cast<long double>(s[i]) * f[i] * z[i] * f[i];
        const std::vector<double> eq(residual.begin() + n, residual.begin() + n + p), conic(residual.begin() + n + p, residual.end()), dual(residual.begin(), residual.begin() + n);
        const long double expected[]{std::max(norm(eq, ei), norm(conic, fi)), norm(dual, di) * kinv, gap * kinv,
            std::max({norm(ax, ei), norm(b, ei), norm(gx, fi), norm(h, fi), norm(s, f)}),
            std::max({norm(px, di), norm(aty, di), norm(gtz, di), norm(c, di)}) * kinv,
            std::max({1.0L, std::abs(.5L * xpx + dot(c, x)), std::abs(-.5L * xpx - dot(b, y) - dot(h, z))})};
        if (iteration == 1) { require(begin() == 0 && begin() == 0, "nested metrics scope"); }
        double got[6], old[6]; gpu(solver, got); reference(solver, old);
        for (int i = 0; i < 6; ++i) { compare(got[i], old[i], "legacy metric parity"); compare(got[i], expected[i], "independent long-double metrics"); }
        compare(qoco_dot(w->x->d_data, w->x->d_data, n), dot(x, x), "host dot after pointer mode restoration");
        if (iteration == 1) { end(); end(); }
    }
    qoco_cleanup(solver);
    std::printf("GPU stopping metrics n=%d absent=%d zero_P=%d PASS\n", n, absent, zero_p);
}
} // namespace
int main() { run(17, false, false); run(4103, false, false); run(17, true, false); run(17, false, true); }
