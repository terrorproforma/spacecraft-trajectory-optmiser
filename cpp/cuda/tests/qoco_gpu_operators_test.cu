// Standalone optional QOCO-GPU extension test; build against the prepared backend.
// Independent dense arithmetic covers rectangular/empty/symmetric matrices,
// duplicate entries, topology refresh, value updates and repeatable products.
#include <cuda_runtime.h>
#include <dlfcn.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
extern "C" {
#include "qoco_linalg.h"
#include "qoco_api.h"
#include "qoco_utils.h"
#include "cone.h"
}
QOCOFloat cone_residual(const QOCOFloat*, QOCOInt, QOCOInt, const QOCOInt*, QOCOInt*);
// Pinned CUDA backend's loader (normally invoked by qoco_setup).
bool load_cuda_libraries();

static void require(bool value, const char* message) {
    if (!value) { std::fprintf(stderr, "FAIL: %s\n", message); std::exit(1); }
}
static void check(cudaError_t status) { require(status == cudaSuccess, cudaGetErrorString(status)); }

static void cone_reduction_case(int lp, const std::vector<int>& sizes, int edge = 0) {
    std::vector<int> starts;
    int n = lp;
    for (int size : sizes) { starts.push_back(n); n += size; }
    std::vector<double> x(n), direction(n);
    for (int i = 0; i < n; ++i) { x[i] = 1.1 + 0.1 * std::sin(i); direction[i] = 3.0 * std::cos(0.7 * i); }
    for (int j = 0; j < static_cast<int>(sizes.size()); ++j) {
        long double norm = 0;
        for (int k = 1; k < sizes[j]; ++k) { double v = x[starts[j] + k]; norm += v * v; }
        x[starts[j]] = static_cast<double>(std::sqrt(norm)) + 0.5;
    }
    if (edge) {
        require(lp == 0 && n == 3, "SOC edge fixture shape");
        x = (edge == 1 || edge == 5) ? std::vector<double>{1, 0, 0} : std::vector<double>{1, 1, 0};
        if (edge == 1) direction = {-1, 1, 0};
        if (edge == 2) direction = {-2, -1, 0};
        if (edge == 3) direction = {0, 1, 0};
        if (edge == 4) direction = {1, -1, 0};
        if (edge == 5) direction = {-1, 1 + 2e-15, 0};
        if (edge == 6) direction = {0, -3, 0};
    }
    auto violation = [&](long double alpha) {
        long double result = -1e7L;
        for (int i = 0; i < lp; ++i) result = std::max(result, -(x[i] + alpha * direction[i]));
        for (int j = 0; j < static_cast<int>(sizes.size()); ++j) {
            long double norm = 0;
            for (int k = 1; k < sizes[j]; ++k) {
                long double v = x[starts[j] + k] + alpha * direction[starts[j] + k]; norm += v * v;
            }
            result = std::max(result, std::sqrt(norm) - (x[starts[j]] + alpha * direction[starts[j]]));
        }
        return result;
    };
    // Independent feasibility bisection, not the implementation's quadratic formula.
    long double lower = 0, upper = 1;
    if (violation(upper) <= 0) lower = upper;
    else for (int i = 0; i < 90; ++i) {
        long double middle = (lower + upper) / 2;
        if (violation(middle) <= 0) lower = middle; else upper = middle;
    }
    QOCOProblemData data{}; data.l = lp; data.nsoc = static_cast<int>(sizes.size()); data.m = n;
    data.q = new_qoco_vectori(sizes.data(), data.nsoc);
    QOCOWorkspace work{}; work.data = &data; work.soc_idx = new_qoco_vectori(starts.data(), data.nsoc);
    QOCOSolver solver{}; solver.work = &work;
    double *dx{}, *dd{};
    check(cudaMalloc(&dx, std::max(1, n) * sizeof(double)));
    check(cudaMalloc(&dd, std::max(1, n) * sizeof(double)));
    if (n) {
        check(cudaMemcpy(dx, x.data(), n * sizeof(double), cudaMemcpyHostToDevice));
        check(cudaMemcpy(dd, direction.data(), n * sizeof(double), cudaMemcpyHostToDevice));
    }
    for (int repeat = 0; repeat < 3; ++repeat) {
        double step = linesearch(dx, dd, 0.99, &solver);
        require(std::isfinite(step) && std::abs(step - 0.99L * lower) < 2e-12L, "GPU line search versus independent feasibility bisection");
        require(violation(step) <= 1e-12L, "line search preserves cone feasibility");
        double residual = cone_residual(dx, lp, data.nsoc, get_data_vectori(data.q), get_data_vectori(work.soc_idx));
        require(std::abs(residual - violation(0)) < 2e-12L, "GPU cone residual independent reference");
    }
    if (lp > 262144) {
        // The old final reduction inspected only the first 1024 partial blocks.
        const double violated = -7.0;
        check(cudaMemcpy(dx + lp - 1, &violated, sizeof(double), cudaMemcpyHostToDevice));
        require(cone_residual(dx, lp, data.nsoc, get_data_vectori(data.q), get_data_vectori(work.soc_idx)) == 7.0,
                "cone residual must include partial blocks beyond 1024");
    }
    check(cudaFree(dx)); check(cudaFree(dd));
    free_qoco_vectori(work.soc_idx); free_qoco_vectori(data.q);
}

static void cone_reduction_cases() {
    auto begin = reinterpret_cast<int (*)()>(dlsym(RTLD_DEFAULT, "qoco_gpu_begin_reduction_scope"));
    auto end = reinterpret_cast<void (*)()>(dlsym(RTLD_DEFAULT, "qoco_gpu_end_reduction_scope"));
    for (int scoped = 0; scoped < 2; ++scoped) {
        if (scoped && begin) require(begin() == 0, "cone reduction scope");
        cone_reduction_case(0, {});
        cone_reduction_case(13, {});
        cone_reduction_case(0, {3, 5, 33});
        cone_reduction_case(8193, std::vector<int>(1027, 4));
        cone_reduction_case(262145, {});
        for (int edge = 1; edge <= 6; ++edge) cone_reduction_case(0, {3}, edge);
        if (scoped && end) end();
    }
    std::puts("QOCO device cone reductions: independent feasibility, mixed cones, >1024 blocks PASS");
}

static void queued_chain_case(int (*begin)(), void (*end)()) {
    // A dependent sequence with in-place updates and both sparse directions.
    // No downloads or scalar reductions are allowed to hide missing ordering.
    constexpr int n = 513;
    std::vector<int> offsets(n + 1), indices;
    std::vector<double> values, x(n), y(n), output(n);
    std::vector<long double> expected(n), temporary(n);
    for (int col = 0; col < n; ++col) {
        offsets[col] = static_cast<int>(values.size());
        for (int row = std::max(0, col - 1); row <= std::min(n - 1, col + 1); ++row) {
            indices.push_back(row); values.push_back(row == col ? 0.75 : -0.125);
        }
        x[col] = std::sin(col * 0.13); y[col] = std::cos(col * 0.07);
    }
    offsets[n] = static_cast<int>(values.size());
    QOCOCscMatrix csc{n, n, static_cast<int>(values.size()), indices.data(), offsets.data(), values.data()};
    auto* matrix = new_qoco_matrix(&csc);
    double *dx{}, *dy{}, *dz{}, *dt{};
    for (auto** ptr : {&dx, &dy, &dz, &dt}) check(cudaMalloc(ptr, n * sizeof(double)));
    check(cudaMemcpy(dx, x.data(), n * sizeof(double), cudaMemcpyHostToDevice));
    check(cudaMemcpy(dy, y.data(), n * sizeof(double), cudaMemcpyHostToDevice));
    cudaEvent_t done{}; check(cudaEventCreateWithFlags(&done, cudaEventDisableTiming));
    for (int scoped = 0; scoped < 2; ++scoped) {
        if (scoped && begin) { require(begin() == 0, "queued scope"); require(begin() == 0, "queued nested scope"); }
        copy_arrayf(dx, dz, n);
        expected.assign(x.begin(), x.end());
        for (int repeat = 0; repeat < 12; ++repeat) {
            ew_product(dz, dy, dz, n);
            qoco_axpy(dz, dx, dz, 0.125, n);
            scale_arrayf(dz, dz, 0.5, n);
            copy_and_negate_arrayf(dz, dt, n);
            SpMv(matrix, dt, dz);
            SpMtv(matrix, dz, dt);
            copy_arrayf(dt, dz, n);
            for (int i = 0; i < n; ++i) expected[i] = -0.5L * (0.125L * expected[i] * y[i] + x[i]);
            std::fill(temporary.begin(), temporary.end(), 0);
            for (int col = 0; col < n; ++col)
                for (int k = offsets[col]; k < offsets[col + 1]; ++k)
                    temporary[indices[k]] += values[k] * expected[col];
            std::fill(expected.begin(), expected.end(), 0);
            for (int col = 0; col < n; ++col)
                for (int k = offsets[col]; k < offsets[col + 1]; ++k)
                    expected[col] += values[k] * temporary[indices[k]];
        }
        // The event precedes scope teardown; query it before any synchronizing copy.
        check(cudaEventRecord(done, nullptr));
        if (scoped && end) { end(); end(); check(cudaEventQuery(done)); }
        else check(cudaEventSynchronize(done));
        check(cudaMemcpy(output.data(), dz, n * sizeof(double), cudaMemcpyDeviceToHost));
        for (int i = 0; i < n; ++i) require(std::isfinite(output[i])
            && std::abs(output[i] - expected[i]) < 2e-12L * (1 + std::abs(expected[i])),
            "queued chain differs from independent arithmetic");
    }
    check(cudaEventDestroy(done));
    for (auto* ptr : {dx, dy, dz, dt}) check(cudaFree(ptr));
    free_qoco_matrix(matrix);
}

static void stopping_scale_case() {
    // min x subject to x = 1e9. This feasible point with y=-0.5 has
    // stationarity residual 0.5, regardless of the magnitude of x.
    int offsets[]{0, 1}, indices[]{0};
    double values[]{1.0}, c[]{1.0}, b[]{1e9};
    QOCOCscMatrix a{1, 1, 1, indices, offsets, values};
    QOCOSettings settings{};
    set_default_settings(&settings);
    settings.ruiz_iters = 0;
    settings.verbose = 0;
    settings.abstol = settings.reltol = 1e-8;
    // qoco_cleanup owns and frees the solver allocation itself.
    auto* allocation = static_cast<QOCOSolver*>(std::calloc(1, sizeof(QOCOSolver)));
    require(allocation != nullptr, "solver allocation");
    auto& solver = *allocation;
    require(qoco_setup(&solver, 1, 0, 1, nullptr, c, &a, b, nullptr, nullptr, 0, 0, nullptr, &settings) == 0,
            "stopping regression setup");
    const double x = 1e9, y = -0.5, residual[]{0.5, 0.0};
    check(cudaMemcpy(get_data_vectorf(solver.work->x), &x, sizeof(double), cudaMemcpyHostToDevice));
    check(cudaMemcpy(get_data_vectorf(solver.work->y), &y, sizeof(double), cudaMemcpyHostToDevice));
    check(cudaMemcpy(get_data_vectorf(solver.work->kktres), residual, sizeof(residual), cudaMemcpyHostToDevice));
    solver.work->a = 1.0;
    const auto stopped = check_stopping(&solver);
    const auto dual_residual = solver.sol->dres;
    qoco_cleanup(&solver);
    require(std::abs(dual_residual - 0.5) < 1e-12, "constructed stationarity residual");
    require(stopped == 0, "large primal value must not loosen relative stationarity tolerance");
    std::puts("QOCO stopping scale regression PASS");
}

static void matrix_case(int rows, int cols, bool symmetric, bool empty, bool repeatability) {
    std::vector<int> offsets(cols + 1), indices;
    std::vector<double> values;
    for (int col = 0; col < cols; ++col) {
        offsets[col] = static_cast<int>(values.size());
        for (int row = 0; row < rows; ++row) {
            if (empty || (symmetric && row > col) || (rows > 1 && (row + col * 7) % 4 == 0)) continue;
            // Include duplicates: both coefficients must contribute.
            for (int duplicate = 0; duplicate < 1 + ((row + col) % 11 == 0); ++duplicate) {
                indices.push_back(row);
                values.push_back(std::sin(1.3 * row + col + duplicate));
            }
        }
    }
    offsets[cols] = static_cast<int>(values.size());
    QOCOCscMatrix csc{rows, cols, static_cast<int>(values.size()), indices.data(), offsets.data(), values.data()};
    auto* matrix = new_qoco_matrix(&csc);
    const int dimension = std::max(rows, cols);
    std::vector<double> vector(dimension), result(dimension), first(dimension);
    for (int j = 0; j < dimension; ++j) vector[j] = std::cos(0.37 * j);
    double *device_vector{}, *device_result{};
    check(cudaMalloc(&device_vector, dimension * sizeof(double)));
    check(cudaMalloc(&device_result, dimension * sizeof(double)));
    check(cudaMemcpy(device_vector, vector.data(), dimension * sizeof(double), cudaMemcpyHostToDevice));
    for (int update = 0; update < 3; ++update) {
        if (update != 0) {
            set_cpu_mode(1);
            auto* host = get_csc_matrix(matrix);
            for (int j = 0; j < csc.nnz; ++j) {
                values[j] *= -0.7;
                host->x[j] = values[j];
                if (!symmetric && update == 2) host->i[j] = indices[j] = (indices[j] + 1) % rows;
            }
            sync_matrix_to_device(matrix);
            set_cpu_mode(0);
        }
        for (int transpose = 0; transpose < (symmetric ? 1 : 2); ++transpose) {
            std::vector<long double> reference(dimension, 0.0L);
            for (int col = 0; col < cols; ++col) {
                for (int k = offsets[col]; k < offsets[col + 1]; ++k) {
                    const int row = indices[k];
                    reference[transpose ? col : row] += static_cast<long double>(values[k]) * vector[transpose ? row : col];
                    if (symmetric && row != col) reference[col] += static_cast<long double>(values[k]) * vector[row];
                }
            }
            for (int repeat = 0; repeat < (repeatability ? 32 : 1); ++repeat) {
                if (symmetric) USpMv(matrix, device_vector, device_result);
                else if (transpose) SpMtv(matrix, device_vector, device_result);
                else SpMv(matrix, device_vector, device_result);
                const int count = transpose ? cols : rows;
                check(cudaMemcpy(result.data(), device_result, count * sizeof(double), cudaMemcpyDeviceToHost));
                for (int j = 0; j < count; ++j) require(
                    std::isfinite(result[j]) && std::abs(result[j] - reference[j]) <= 1e-12L * (1 + std::abs(reference[j])),
                    "GPU sparse product differs from independent dense arithmetic");
                if (repeat == 0) first = result;
                else require(std::memcmp(first.data(), result.data(), count * sizeof(double)) == 0,
                             "GPU sparse product is not bitwise repeatable");
            }
        }
    }
    check(cudaFree(device_vector));
    check(cudaFree(device_result));
    free_qoco_matrix(matrix);
}

int main(int argc, char** argv) {
    require(load_cuda_libraries(), "load CUDA algebra libraries");
    if (argc == 2 && std::strcmp(argv[1], "--cone-reductions-only") == 0) {
        cone_reduction_cases();
        return 0;
    }
    if (argc == 2 && std::strcmp(argv[1], "--cone-boundaries-only") == 0) {
        for (int edge = 1; edge <= 6; ++edge) cone_reduction_case(0, {3}, edge);
        std::puts("QOCO SOC linear and boundary directions PASS");
        return 0;
    }
    if (argc == 2 && std::strcmp(argv[1], "--stopping-only") == 0) {
        stopping_scale_case();
        return 0;
    }
    const bool repeatability = argc == 1 || std::strcmp(argv[1], "--accuracy-only") != 0;
    matrix_case(37, 23, false, false, repeatability);
    matrix_case(23, 37, false, false, repeatability);
    matrix_case(23, 23, true, false, repeatability);
    matrix_case(7, 11, false, true, repeatability);
    matrix_case(7, 7, true, true, repeatability);
    matrix_case(1, 1, false, false, repeatability);
    auto* empty = new_qoco_matrix(nullptr);
    free_qoco_matrix(empty);
    using Begin = int (*)();
    using End = void (*)();
    auto begin = reinterpret_cast<Begin>(dlsym(RTLD_DEFAULT, "qoco_gpu_begin_reduction_scope"));
    auto end = reinterpret_cast<End>(dlsym(RTLD_DEFAULT, "qoco_gpu_end_reduction_scope"));
    require((begin == nullptr) == (end == nullptr), "partial reduction-scope extension");
    queued_chain_case(begin, end);
    std::vector<double> x(1027), y(1027);
    long double reference = 0;
    for (int i = 0; i < 1027; ++i) { x[i] = std::sin(i); y[i] = std::cos(0.37 * i); reference += static_cast<long double>(x[i]) * y[i]; }
    double *dx{}, *dy{};
    check(cudaMalloc(&dx, x.size() * sizeof(double)));
    check(cudaMalloc(&dy, y.size() * sizeof(double)));
    check(cudaMemcpy(dx, x.data(), x.size() * sizeof(double), cudaMemcpyHostToDevice));
    check(cudaMemcpy(dy, y.data(), y.size() * sizeof(double), cudaMemcpyHostToDevice));
    for (int scope = 0; scope < 64; ++scope) {
        if (begin) { require(begin() == 0, "begin scope"); require(begin() == 0, "nested scope"); }
        const auto dot = qoco_dot(dx, dy, static_cast<int>(x.size()));
        require(std::isfinite(dot) && std::abs(dot - reference) < 1e-12L, "scoped reduction accuracy");
        if (end) { end(); end(); end(); }
        require(std::abs(qoco_dot(dx, dy, static_cast<int>(x.size())) - reference) < 1e-12L,
                "unscoped reduction after scope cleanup");
    }
    check(cudaFree(dx)); check(cudaFree(dy));
    check(cudaDeviceSynchronize());
    std::printf("QOCO GPU operators: independent arithmetic, updates, scope lifecycle%s PASS\n",
                repeatability ? ", bitwise repeatability" : "");
    return 0;
}
