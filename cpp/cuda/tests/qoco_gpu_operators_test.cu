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
}
// Pinned CUDA backend's loader (normally invoked by qoco_setup).
bool load_cuda_libraries();

static void require(bool value, const char* message) {
    if (!value) { std::fprintf(stderr, "FAIL: %s\n", message); std::exit(1); }
}
static void check(cudaError_t status) { require(status == cudaSuccess, cudaGetErrorString(status)); }

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
