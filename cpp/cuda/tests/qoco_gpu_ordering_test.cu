// Standalone test of static-degree permutation, without vendor factorization.
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <vector>
extern "C" void qoco_gpu_degree_ordering(int, const int*, const int*, int*);
static void require(bool ok, const char* message) {
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", message); std::exit(1); }
}
static void check(cudaError_t code) { require(code == cudaSuccess, cudaGetErrorString(code)); }
static void run(int n, bool disconnected) {
    std::vector<int> offsets(n + 1), columns, degrees(n, 0), expected(n);
    for (int row = 0; row < n; ++row) {
        offsets[row] = columns.size();
        columns.push_back(row);
        if (!disconnected) for (int distance : {1, 5, 5, 29}) {
            const int col = row + distance;
            if (col >= n || row % 7 == 0) continue;
            columns.push_back(col); ++degrees[row]; ++degrees[col];
        }
        if (!disconnected && row % 3 == 1 && row < n - 1) {
            columns.push_back(n - 1); ++degrees[row]; ++degrees[n - 1];
        }
    }
    offsets[n] = columns.size();
    std::iota(expected.begin(), expected.end(), 0);
    std::stable_sort(expected.begin(), expected.end(), [&](int a, int b) { return degrees[a] < degrees[b]; });
    int *d_offsets{}, *d_columns{}, *guarded{};
    check(cudaMalloc(&d_offsets, offsets.size() * sizeof(int)));
    check(cudaMalloc(&d_columns, std::max(size_t(1), columns.size()) * sizeof(int)));
    check(cudaMalloc(&guarded, (n + 2) * sizeof(int)));
    check(cudaMemcpy(d_offsets, offsets.data(), offsets.size() * sizeof(int), cudaMemcpyHostToDevice));
    if (!columns.empty()) check(cudaMemcpy(d_columns, columns.data(), columns.size() * sizeof(int), cudaMemcpyHostToDevice));
    for (int repeat = 0; repeat < 3; ++repeat) {
        std::vector<int> result(n + 2, -77);
        check(cudaMemcpy(guarded, result.data(), result.size() * sizeof(int), cudaMemcpyHostToDevice));
        qoco_gpu_degree_ordering(n, d_offsets, d_columns, guarded + 1);
        check(cudaMemcpy(result.data(), guarded, result.size() * sizeof(int), cudaMemcpyDeviceToHost));
        require(result.front() == -77 && result.back() == -77, "permutation bounds");
        require(std::equal(expected.begin(), expected.end(), result.begin() + 1), "independent stable degree order");
        auto sorted = std::vector<int>(result.begin() + 1, result.end() - 1);
        std::sort(sorted.begin(), sorted.end());
        for (int i = 0; i < n; ++i) require(sorted[i] == i, "permutation bijection");
    }
    check(cudaFree(guarded)); check(cudaFree(d_columns)); check(cudaFree(d_offsets));
}
int main() {
    for (int n : {0, 1, 13, 4099, 65539}) { run(n, false); run(n, true); }
    check(cudaDeviceSynchronize());
    std::puts("GPU static-degree ordering: independent reference, stable ties, bijection, duplicates, guards PASS");
}
