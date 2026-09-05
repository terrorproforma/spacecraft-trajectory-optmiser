// Exact GPU KKT CSR/map comparison is enabled in each assembly. Cases exercise
// empty A/G blocks, duplicate sparse entries, empty rows and irregular SOCs.
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
extern "C" {
#include "qoco_api.h"
int qoco_test_gpu_kkt(QOCOCscMatrix*, QOCOCscMatrix*, QOCOCscMatrix*,
                      int, int, int, int, int, int*);
}
static void require(bool ok, const char* message) {
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", message); std::exit(1); }
}
struct Csc {
    QOCOCscMatrix view{};
    std::vector<int> offsets, rows;
    std::vector<double> values;
    Csc(int m, int n, bool quadratic) : offsets(n + 1) {
        for (int col = 0; col < n; ++col) {
            offsets[col] = static_cast<int>(values.size());
            if (quadratic) {
                if (col > 0) { rows.push_back(col - 1); values.push_back(0.125); }
                rows.push_back(col); values.push_back(3 + 0.01 * col);
            } else if (m) {
                // Deliberately leave rows empty and retain duplicate entries.
                rows.push_back(col % m); values.push_back(0.2 + 0.001 * col);
                rows.push_back(col % m); values.push_back(-0.05);
            }
        }
        offsets[n] = static_cast<int>(values.size());
        qoco_set_csc(&view, m, n, static_cast<int>(values.size()), values.data(), offsets.data(), rows.data());
    }
};
static void run(int n, int p, int l, std::vector<int> cones) {
    int m = l; for (int q : cones) m += q;
    Csc P(n, n, true), A(p, n, false), G(m, n, false);
    int wn = l; for (int q : cones) wn += q * (q + 1) / 2;
    require(qoco_test_gpu_kkt(&P.view, p ? &A.view : nullptr, m ? &G.view : nullptr,
        n, p, m, l, static_cast<int>(cones.size()), cones.data())
        == P.view.nnz + A.view.nnz + G.view.nnz + p + wn,
        "GPU assembly and complete independent CSR/map comparison");
    require(cudaDeviceSynchronize() == cudaSuccess, "cleanup completion");
    std::printf("GPU KKT exact oracle n=%d p=%d m=%d cones=%zu PASS\n", n, p, m, cones.size());
}
int main() {
    run(1, 0, 0, {});
    run(17, 5, 0, {});
    run(5, 17, 0, {});
    run(17, 0, 7, {});
    run(7, 0, 0, {3, 9});
    run(19, 3, 5, {1, 2, 3, 17, 257});
    run(9, 2, 0, std::vector<int>(513, 1));
}
