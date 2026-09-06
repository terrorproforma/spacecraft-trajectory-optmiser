#include "qoco.h"
#include "../algebra/cuda/cuda_types.h"
#include <cuda_runtime.h>
#include <dlfcn.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {
void require(bool ok, const char* message) { if (!ok) { std::fprintf(stderr, "FAIL: %s\n", message); std::exit(1); } }
void check(cudaError_t code) { require(code == cudaSuccess, cudaGetErrorString(code)); }
std::vector<double> download(const double* values, int count) {
    std::vector<double> result(count);
    if (count) check(cudaMemcpy(result.data(), values, count * sizeof(double), cudaMemcpyDeviceToHost));
    return result;
}
void compare(const std::vector<double>& actual, const std::vector<double>& expected, const char* label) {
    require(actual.size() == expected.size(), "comparison dimensions");
    for (size_t i = 0; i < actual.size(); ++i)
        if (!std::isfinite(actual[i]) || std::abs(actual[i] - expected[i]) > 2e-11 * std::max(1.0, std::abs(expected[i]))) {
            std::fprintf(stderr, "%s[%zu] %.17g != %.17g\n", label, i, actual[i], expected[i]); std::exit(1);
        }
}
struct Sparse {
    std::vector<int> offsets{0}, indices;
    std::vector<double> values;
    int rows, columns;
    QOCOCscMatrix view() { return {rows, columns, static_cast<int>(values.size()), indices.data(), offsets.data(), values.data()}; }
};
void run_case(int n, int iterations, bool missing_diagonal, bool unconstrained, bool zero_quadratic = false,
              bool off_diagonal = false) {
    const int p = unconstrained ? 0 : 5, m = unconstrained ? 0 : 9, l = unconstrained ? 0 : 2;
    Sparse P{{0}, {}, {}, n, n}, A{{0}, {}, {}, p, n}, G{{0}, {}, {}, m, n};
    for (int j = 0; j < n; ++j) {
        if (off_diagonal && j > 0) { P.indices.push_back(j - 1); P.values.push_back(1e-5); }
        if (!zero_quadratic && (!missing_diagonal || j % 2 == 0)) { P.indices.push_back(j); P.values.push_back(std::pow(10.0, j % 7 - 3)); }
        P.offsets.push_back(P.values.size());
        for (int row = 0; row < p; ++row) if ((row + j) % 3 != 1) { A.indices.push_back(row); A.values.push_back(0.1 + std::cos(row + j * 0.2)); }
        A.offsets.push_back(A.values.size());
        for (int row = 0; row < m; ++row) if ((row + j) % 3 != 1) { G.indices.push_back(row); G.values.push_back(0.5 + std::sin(row + j * 0.2)); }
        G.offsets.push_back(G.values.size());
    }
    auto pc = P.view(), ac = A.view(), gc = G.view();
    std::vector<double> c(n), b(p, 0.3), h(m, 1.1);
    for (int j = 0; j < n; ++j) c[j] = 0.1 + std::sin(j * 0.3);
    int cones[]{3, 4};
    QOCOSettings settings{}; set_default_settings(&settings); settings.ruiz_iters = iterations; settings.verbose = 0;
    QOCOSolver *cpu = static_cast<QOCOSolver*>(std::calloc(1, sizeof(QOCOSolver))),
               *gpu = static_cast<QOCOSolver*>(std::calloc(1, sizeof(QOCOSolver)));
    for (auto* solver : {cpu, gpu}) require(qoco_setup(solver, n, m, p, zero_quadratic ? nullptr : &pc, c.data(), p ? &ac : nullptr,
        b.data(), m ? &gc : nullptr, h.data(), l, unconstrained ? 0 : 2, cones, &settings) == 0, "setup reference solver");
    using Create = int (*)(QOCOSolver*, int, int, int, void**);
    using Update = int (*)(void*, const double*, cudaStream_t);
    using Destroy = void (*)(void*);
    auto create = reinterpret_cast<Create>(dlsym(RTLD_DEFAULT, "qoco_gpu_create_numeric_update"));
    auto update = reinterpret_cast<Update>(dlsym(RTLD_DEFAULT, "qoco_gpu_update_numeric"));
    auto destroy = reinterpret_cast<Destroy>(dlsym(RTLD_DEFAULT, "qoco_gpu_destroy_numeric_update"));
    require(create && update && destroy, "complete device update interface");
    void* workspace{}; require(create(gpu, pc.nnz, ac.nnz, gc.nnz, &workspace) == 0, "create retained update context");
    for (auto* vector : {gpu->work->data->c, gpu->work->data->b, gpu->work->data->h})
        compare(download(vector->d_data, vector->len), std::vector<double>(vector->data, vector->data + vector->len),
            "initial scaled vectors must reach device");
    cudaStream_t stream{}; check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    std::vector<double> packed(pc.nnz + ac.nnz + gc.nnz + n + p + m);
    double* device{}; check(cudaMalloc(&device, (packed.size() + 1) * sizeof(double)));
    for (int repeat = 0; repeat < 3; ++repeat) {
        for (double& value : P.values) value *= 1.1;
        for (double& value : A.values) value *= -0.9;
        for (double& value : G.values) value *= 0.95;
        for (double& value : c) value *= -0.8;
        for (double& value : b) value += 0.1;
        for (double& value : h) value += 0.2;
        size_t cursor = 0;
        for (const auto* values : {&P.values, &A.values, &G.values, &c, &b, &h}) {
            std::copy(values->begin(), values->end(), packed.begin() + cursor); cursor += values->size();
        }
        {
            qoco_update_matrix_data(cpu, P.values.data(), A.values.data(), G.values.data());
            qoco_update_vector_data(cpu, c.data(), b.data(), h.data());
        }
        check(cudaMemcpyAsync(device + 1, packed.data(), packed.size() * sizeof(double), cudaMemcpyHostToDevice, stream));
        require(update(workspace, device + 1, stream) == 0, "device numerical update");
        auto* data = gpu->work->data; auto* scales = gpu->work->scaling;
        {
            int matrix_index = 0;
            for (const auto pair : std::initializer_list<std::pair<QOCOMatrix*, QOCOMatrix*>>{{data->P, cpu->work->data->P}, {data->A, cpu->work->data->A},
                                    {data->G, cpu->work->data->G}, {data->At, cpu->work->data->At}, {data->Gt, cpu->work->data->Gt}}) {
                const char* labels[]{"P", "A", "G", "At", "Gt"};
                compare(download(pair.first->d_csc_host->x, pair.first->csc->nnz),
                        download(pair.second->d_csc_host->x, pair.second->csc->nnz), labels[matrix_index++]);
            }
            for (const auto pair : std::initializer_list<std::pair<QOCOVectorf*, QOCOVectorf*>>{{data->c, cpu->work->data->c}, {data->b, cpu->work->data->b}, {data->h, cpu->work->data->h},
                {scales->Druiz, cpu->work->scaling->Druiz}, {scales->Eruiz, cpu->work->scaling->Eruiz}, {scales->Fruiz, cpu->work->scaling->Fruiz},
                {scales->Dinvruiz, cpu->work->scaling->Dinvruiz}, {scales->Einvruiz, cpu->work->scaling->Einvruiz}, {scales->Finvruiz, cpu->work->scaling->Finvruiz}})
                compare(download(pair.first->d_data, pair.first->len), download(pair.second->d_data, pair.second->len), "vector against CPU updater");
            compare({scales->k, scales->kinv}, {cpu->work->scaling->k, cpu->work->scaling->kinv}, "cost scale");
        }
#ifdef SPACEPDHCG_QOCO_LAZY_HOST_MIRRORS
        for (auto* matrix : {data->At, data->Gt}) {
            require(matrix->lazy_host_mirror && matrix->host_values_pending,
                    "GPU update invalidates transpose host cache");
            auto* topology_before = matrix->csc->p;
            set_cpu_mode(1);
            auto* host = get_csc_matrix(matrix);
            set_cpu_mode(0);
            require(!matrix->host_values_pending, "explicit inspection refreshes values");
            require(!topology_before || host->p == topology_before, "fixed topology cache reused");
            if (host->nnz) {
                compare(download(matrix->d_csc_host->x, host->nnz),
                        std::vector<double>(host->x, host->x + host->nnz), "fresh transpose cache");
                // The next real GPU update must replace this stale host cache.
                std::fill_n(host->x, host->nnz, -987654.0);
            }
        }
#endif
        const auto D = download(scales->Druiz->d_data, n), E = download(scales->Eruiz->d_data, p), F = download(scales->Fruiz->d_data, m);
        for (const auto item : std::initializer_list<std::pair<Sparse*, QOCOMatrix*>>{{&P, data->P}, {&A, data->A}, {&G, data->G}}) {
            const bool quadratic = item.first == &P;
            const auto* matrix = item.second->csc;
            std::vector<double> expected(matrix->nnz, 0.0);
            const auto& row_scale = quadratic ? D : item.first == &A ? E : F;
            for (int col = 0; col < matrix->n; ++col) {
                for (int k = matrix->p[col]; k < matrix->p[col + 1]; ++k) {
                    for (int raw = item.first->offsets[col]; raw < item.first->offsets[col + 1]; ++raw)
                        if (item.first->indices[raw] == matrix->i[k])
                            expected[k] += static_cast<long double>(item.first->values[raw]) * D[col] * row_scale[matrix->i[k]] * (quadratic ? scales->k : 1.0);
                    if (quadratic && matrix->i[k] == col) expected[k] += settings.kkt_static_reg_P;
                }
            }
            compare(download(item.second->d_csc_host->x, matrix->nnz), expected, "independent scaling equation");
        }
    }
    destroy(workspace); check(cudaFree(device)); check(cudaStreamDestroy(stream)); qoco_cleanup(cpu); qoco_cleanup(gpu);
    std::printf("GPU numeric updates n=%d Ruiz=%d missing_diagonal=%d unconstrained=%d zero_quadratic=%d off_diagonal=%d PASS\n",
        n, iterations, missing_diagonal, unconstrained, zero_quadratic, off_diagonal);
}
}
int main() {
    require(setenv("SPACEPDHCG_TEST_QOCO_UPDATE_MAPS_COMPARE", "1", 1) == 0, "enable exact setup map oracle");
    double host_values[]{2, -4, 0};
    scale_arrayf(host_values, host_values, 0.25, 3);
    compare({host_values[0], host_values[1], host_values[2]}, {0.5, -1.0, 0.0}, "host setup scaling regression");
    run_case(17, 0, false, false); run_case(17, 1, false, false); run_case(17, 4, false, false);
    run_case(1031, 4, false, false); run_case(17, 4, true, false); run_case(17, 0, false, true);
    run_case(17, 4, true, false, true);
    run_case(1031, 4, true, false);
    run_case(1031, 4, true, false, false, true);
}
