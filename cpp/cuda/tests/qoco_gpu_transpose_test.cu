// Exact transpose/map and independent ownership tests. Poisoning the source's
// host mirrors proves that construction reads the resident GPU representation.
#include "qoco.h"
#include "../algebra/cuda/cuda_types.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <vector>

extern "C" QOCOMatrix* qoco_gpu_transpose(const QOCOMatrix*, QOCOInt*);
#ifdef SPACEPDHCG_QOCO_LAZY_HOST_MIRRORS
extern "C" QOCOMatrix* qoco_gpu_transpose_lazy(const QOCOMatrix*, QOCOInt*);
#endif
namespace {
void require(bool ok, const char* message) {
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", message); std::exit(1); }
}
template<class T> void compare_device(const T* device, const std::vector<T>& expected) {
    std::vector<T> actual(expected.size());
    if (expected.empty()) return;
    require(cudaMemcpy(actual.data(), device, actual.size()*sizeof(T), cudaMemcpyDeviceToHost)
                == cudaSuccess, "download");
    require(std::memcmp(actual.data(), expected.data(), actual.size()*sizeof(T)) == 0,
            "device transpose differs bitwise from independent reference");
}
void run(int rows, int columns, bool empty, bool missing = false, bool lazy = false) {
    std::vector<int> offsets{0}, indices, original_columns;
    std::vector<double> values;
    for (int col = 0; col < columns; ++col) {
        if (!empty && rows && col % 7 != 0) {
            // Deliberately unsorted rows, repeated coordinates, empty columns,
            // both signs and explicit signed zeros.
            for (int k = 0; k < 9; ++k) {
                indices.push_back(k < 2 ? (col*3)%rows : (rows-1-(col+k*13)%rows));
                original_columns.push_back(col);
                values.push_back(k == 0 ? -0.0 : (col*9+k-17)*0.125);
            }
        }
        offsets.push_back(static_cast<int>(values.size()));
    }
    QOCOCscMatrix input{rows, columns, static_cast<int>(values.size()),
                        indices.data(), offsets.data(), values.data()};
    auto* source = new_qoco_matrix(missing ? nullptr : &input);
    std::vector<int> order(values.size()); std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return indices[a] < indices[b]; });
    std::vector<int> expected_offsets(rows+1, 0), expected_rows, expected_inverse(order.size());
    std::vector<double> expected_values;
    for (size_t k = 0; k < order.size(); ++k) {
        const int entry = order[k];
        ++expected_offsets[indices[entry]+1];
        expected_rows.push_back(original_columns[entry]);
        expected_values.push_back(values[entry]);
        expected_inverse[entry] = static_cast<int>(k);
    }
    std::partial_sum(expected_offsets.begin(), expected_offsets.end(), expected_offsets.begin());
    if (!missing) {
        std::fill_n(source->csc->p, columns+1, -123);
        if (!values.empty()) {
            std::fill_n(source->csc->i, values.size(), -77);
            std::fill_n(source->csc->x, values.size(), 987654.0);
        }
    }
    std::vector<int> inverse(order.size(), -1);
    auto construct = qoco_gpu_transpose;
#ifdef SPACEPDHCG_QOCO_LAZY_HOST_MIRRORS
    if (lazy) construct = qoco_gpu_transpose_lazy;
#endif
    auto* result = construct(source, inverse.data());
    auto* sibling = construct(source, nullptr);
    // The result must outlive its input and another independently built result.
    free_qoco_matrix(source);
    free_qoco_matrix(sibling);
    require(result->csc->m == columns && result->csc->n == rows, "transpose dimensions");
    require(inverse == expected_inverse, "inverse update map");
    compare_device(result->d_csc_host->p, expected_offsets);
    compare_device(result->d_csc_host->i, expected_rows);
    compare_device(result->d_csc_host->x, expected_values);
#ifdef SPACEPDHCG_QOCO_LAZY_HOST_MIRRORS
    if (lazy) {
        require(result->lazy_host_mirror && result->host_values_pending, "device owns lazy values");
        require(!result->csc->p && !result->csc->i && !result->csc->x, "no eager host arrays");
        sync_matrix_to_device(result);
        require(!result->csc->p && result->host_values_pending, "unchanged upload must remain deferred");
        set_cpu_mode(1);
        require(get_csc_matrix(result) == result->csc, "legacy accessor");
        require(!result->host_values_pending, "host cache materialized");
        auto* offsets_before = result->csc->p;
        get_csc_matrix(result);
        require(result->csc->p == offsets_before, "repeated access reuses cache");
        set_cpu_mode(0);
    }
#endif
    require(std::memcmp(result->csc->p, expected_offsets.data(), expected_offsets.size()*sizeof(int)) == 0,
            "legacy host offsets");
    if (!values.empty()) {
        require(std::memcmp(result->csc->i, expected_rows.data(), expected_rows.size()*sizeof(int)) == 0,
                "legacy host rows");
        require(std::memcmp(result->csc->x, expected_values.data(), expected_values.size()*sizeof(double)) == 0,
                "legacy host values");
    }
    // Re-transposition uses the result's independently owned gather topology.
    auto* twice = qoco_gpu_transpose(result, nullptr);
    free_qoco_matrix(result);
    require(twice->csc->m == rows && twice->csc->n == columns, "double transpose dimensions");
    compare_device(twice->d_csc_host->p, offsets);
    std::vector<double> sorted_values;
    std::vector<int> sorted_rows;
    for (int col = 0; col < columns; ++col) {
        std::vector<int> entries(offsets[col+1]-offsets[col]);
        std::iota(entries.begin(), entries.end(), offsets[col]);
        std::stable_sort(entries.begin(), entries.end(), [&](int a, int b) {return indices[a]<indices[b];});
        for (int entry : entries) {sorted_rows.push_back(indices[entry]);sorted_values.push_back(values[entry]);}
    }
    compare_device(twice->d_csc_host->i, sorted_rows);
    compare_device(twice->d_csc_host->x, sorted_values);
    free_qoco_matrix(twice);
    std::printf("{\"case\":\"gpu_transpose\",\"rows\":%d,\"columns\":%d,\"nnz\":%zu,\"lazy\":%s,\"passed\":true}\n",
                 rows, columns, values.size(), lazy ? "true" : "false");
}
}
int main() {
    run(0,0,true,true); run(0,0,true); run(0,37,true); run(29,0,true);
    run(41,53,true); run(1,19,false); run(17,29,false); run(1031,1537,false);
#ifdef SPACEPDHCG_QOCO_LAZY_HOST_MIRRORS
    run(0,0,true,true,true); run(0,0,true,false,true); run(0,37,true,false,true);
    run(29,0,true,false,true); run(41,53,true,false,true); run(1,19,false,false,true);
    run(17,29,false,false,true); run(1031,1537,false,false,true);
#endif
    require(cudaDeviceSynchronize() == cudaSuccess, "completion");
}
