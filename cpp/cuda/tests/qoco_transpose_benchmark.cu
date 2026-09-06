// Isolated construction benchmark: both strategies share one frozen library,
// source matrix and GPU ordering. CPU transpose + normal constructor is the
// old setup path; GPU transpose reads the precomputed source gather ordering.
#include "qoco.h"
#include "../algebra/cuda/cuda_types.h"
#include <cuda_runtime.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>
extern "C" QOCOMatrix* qoco_gpu_transpose(const QOCOMatrix*, QOCOInt*);
int main() {
    for (int columns : {1000, 25000, 100000}) {
        const int rows = columns*3/5;
        std::vector<int> offsets{0}, indices;
        std::vector<double> values;
        for (int col=0; col<columns; ++col) {
            for (int k=0; k<9; ++k) {
                indices.push_back((col*13+k*97)%rows);
                values.push_back((col+k+1)*0.125);
            }
            offsets.push_back(static_cast<int>(values.size()));
        }
        QOCOCscMatrix input{rows,columns,static_cast<int>(values.size()),
                           indices.data(),offsets.data(),values.data()};
        auto* source = new_qoco_matrix(&input);
        std::vector<int> mapping(values.size());
        for (int repeat=0; repeat<23; ++repeat) {
            for (int order=0; order<2; ++order) {
                const bool gpu = (order + repeat)%2;
                if (cudaDeviceSynchronize()!=cudaSuccess) return 1;
                const auto start = std::chrono::steady_clock::now();
                QOCOMatrix* result;
                if (gpu) {
                    result = qoco_gpu_transpose(source,mapping.data());
                } else {
                    set_cpu_mode(1);
                    auto* host = create_transposed_matrix(get_csc_matrix(source),mapping.data());
                    result = new_qoco_matrix(host);
                    free_qoco_csc_matrix(host);
                    set_cpu_mode(0);
                }
                if (cudaDeviceSynchronize()!=cudaSuccess) return 1;
                const double seconds = std::chrono::duration<double>(
                    std::chrono::steady_clock::now()-start).count();
                std::printf("{\"columns\":%d,\"rows\":%d,\"nnz\":%zu,\"repeat\":%d,"
                             "\"warmup\":%s,\"gpu\":%s,\"construction_seconds\":%.17g}\n",
                             columns,rows,values.size(),repeat,repeat<2?"true":"false",
                             gpu?"true":"false",seconds);
                free_qoco_matrix(result);
            }
        }
        free_qoco_matrix(source);
    }
}
