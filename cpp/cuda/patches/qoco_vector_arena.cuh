// SPDX-License-Identifier: Apache-2.0
// One solver-owned allocation for the 26 post-analysis scratch vectors.
#include "qoco.h"
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>

namespace qoco_vector_arena {
struct Arena { unsigned char* base{}; size_t bytes{}, used{}; int vectors{}; };
inline size_t aligned(int count) {
    if (count < 0 || static_cast<size_t>(count) > (std::numeric_limits<size_t>::max() - 255) / sizeof(double)) {
        std::fprintf(stderr, "Invalid QOCO vector arena size\n"); std::exit(1);
    }
    return (static_cast<size_t>(count) * sizeof(double) + 255) & ~size_t(255);
}
}
extern "C" void* qoco_gpu_vector_arena_create(int n, int m, int p, int wn, int nt) {
    using namespace qoco_vector_arena;
    auto* arena = new Arena{};
    const long long total = static_cast<long long>(n) + m + p;
    if (total > std::numeric_limits<int>::max()) {
        std::fprintf(stderr, "Unsupported QOCO vector arena dimensions\n"); std::exit(1);
    }
    const int xyz = static_cast<int>(total);
    const int lengths[]{n,n,m,p,m,n,m,p,m,wn,nt,wn,wn,m,m,m,n,p,m,m,m,m,xyz,xyz,xyz,xyz};
    for (int length : lengths) {
        const size_t bytes = aligned(length);
        if (arena->bytes > std::numeric_limits<size_t>::max() - bytes) {
            std::fprintf(stderr, "QOCO vector arena overflow\n"); std::exit(1);
        }
        arena->bytes += bytes;
    }
    CUDA_CHECK(cudaMalloc(&arena->base, arena->bytes));
    CUDA_CHECK(cudaMemsetAsync(arena->base, 0, arena->bytes));
    return arena;
}
extern "C" QOCOVectorf* qoco_gpu_vector_arena_vector(void* opaque, int length) {
    using namespace qoco_vector_arena;
    auto* arena = static_cast<Arena*>(opaque);
    const size_t bytes = aligned(length);
    if (!arena || bytes > arena->bytes - arena->used) {
        std::fprintf(stderr, "QOCO vector arena capacity mismatch\n"); std::exit(1);
    }
    auto* vector = static_cast<QOCOVectorf*>(qoco_malloc(sizeof(QOCOVectorf)));
    vector->len = length;
    vector->data = static_cast<double*>(qoco_calloc(length, sizeof(double)));
    vector->d_data = length ? reinterpret_cast<double*>(arena->base + arena->used) : nullptr;
    vector->arena_owned = 1;
    arena->used += bytes;
    ++arena->vectors;
    return vector;
}
extern "C" void qoco_gpu_vector_arena_finish(void* opaque) {
    auto* arena = static_cast<qoco_vector_arena::Arena*>(opaque);
    if (!arena || arena->used != arena->bytes || arena->vectors != 26) {
        std::fprintf(stderr, "QOCO vector arena layout mismatch\n"); std::exit(1);
    }
    // Preserve qoco_setup's completed-initialization boundary for callers.
    CUDA_CHECK(cudaStreamSynchronize(nullptr));
}
extern "C" void qoco_gpu_vector_arena_destroy(void* opaque) {
    auto* arena = static_cast<qoco_vector_arena::Arena*>(opaque);
    if (!arena) return;
    CUDA_CHECK(cudaFree(arena->base));
    delete arena;
}
// Explicit capability for the independent ownership/aliasing fixture.
extern "C" int qoco_test_vector_arena_info(QOCOSolver* solver, void** base, size_t* bytes) {
    if (!solver || !solver->work || !base || !bytes) return 1;
    auto* arena = static_cast<qoco_vector_arena::Arena*>(solver->work->gpu_vector_arena);
    if (!arena || arena->vectors != 26 || arena->used != arena->bytes) return 1;
    *base = arena->base; *bytes = arena->bytes;
    return 0;
}
