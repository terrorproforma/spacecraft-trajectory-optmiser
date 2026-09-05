// SPDX-License-Identifier: Apache-2.0
// Experimental interval separators. All graph labeling and permutation work is
// on CUDA; host input is dimensions and borrowed device index maps only.
#include <cub/device/device_radix_sort.cuh>
#include <climits>
#include <cstdlib>
struct QocoTrajectoryLayout {
    int intervals{}, nx{}, nu{};
    const int *states{}, *controls{}, *virtuals{};
};
static thread_local QocoTrajectoryLayout qoco_next_trajectory;
extern "C" int qoco_gpu_set_trajectory(int intervals, int nx, int nu,
    const int* states, const int* controls, const int* virtuals, cudaStream_t stream) {
    qoco_next_trajectory = {};
    if (intervals == 0) return 0;
    if (intervals < 0 || intervals == INT_MAX || nx <= 0 || nu <= 0 ||
        !states || !controls || !virtuals) return -1;
    if (cudaStreamSynchronize(stream) != cudaSuccess) return -2;
    qoco_next_trajectory = {intervals, nx, nu, states, controls, virtuals};
    return 0;
}
__global__ void qoco_trajectory_map(int n, long long count, int width,
    const int* indices, int* stage, int* state_vertex, bool is_state, int* invalid) {
    for (long long k = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
         k < count; k += static_cast<long long>(blockDim.x) * gridDim.x) {
        const int v = indices[k];
        if (v < 0 || v >= n) { atomicExch(invalid, 1); continue; }
        if (atomicCAS(stage + v, -1, static_cast<int>(k / width)) != -1)
            atomicExch(invalid, 1);
        else if (is_state) state_vertex[v] = 1;
    }
}
__global__ void qoco_trajectory_seed(int total, const int* stage, int* lo, int* hi) {
    for (int v = blockIdx.x * blockDim.x + threadIdx.x; v < total; v += blockDim.x * gridDim.x) {
        lo[v] = stage[v] < 0 ? INT_MAX : stage[v]; hi[v] = stage[v];
    }
}
__device__ void qoco_trajectory_union(int v, int low, int high, int* lo, int* hi) {
    atomicMin(lo + v, low); atomicMax(hi + v, high);
}
__global__ void qoco_trajectory_edges(int total, const int* offsets, const int* columns,
    const int* stage, int* lo, int* hi, int* degree) {
    for (int r = blockIdx.x * blockDim.x + threadIdx.x; r < total; r += blockDim.x * gridDim.x) {
        int count = 0;
        for (int k = offsets[r]; k < offsets[r + 1]; ++k) {
            const int c = columns[k]; if (r == c) continue;
            ++count; atomicAdd(degree + c, 1);
            if (stage[r] >= 0) qoco_trajectory_union(c, stage[r], stage[r], lo, hi);
            if (stage[c] >= 0) qoco_trajectory_union(r, stage[c], stage[c], lo, hi);
        }
        atomicAdd(degree + r, count);
    }
}
__global__ void qoco_trajectory_unmapped(int n, const int* offsets, const int* columns,
    const int* stage, const int* lo, const int* hi, int* outlo, int* outhi) {
    for (int r = blockIdx.x * blockDim.x + threadIdx.x; r < n; r += blockDim.x * gridDim.x) {
        if (stage[r] >= 0) continue;
        int low = lo[r], high = hi[r];
        for (int k = offsets[r]; k < offsets[r + 1]; ++k) {
            const int c = columns[k]; low = min(low, lo[c]); high = max(high, hi[c]);
        }
        outlo[r] = low; outhi[r] = high;
    }
}
__global__ void qoco_trajectory_duals(int n, const int* offsets, const int* columns,
    const int* primal_lo, const int* primal_hi, int* lo, int* hi) {
    for (int r = blockIdx.x * blockDim.x + threadIdx.x; r < n; r += blockDim.x * gridDim.x)
        for (int k = offsets[r]; k < offsets[r + 1]; ++k) {
            const int c = columns[k];
            if (c >= n) qoco_trajectory_union(c, primal_lo[r], primal_hi[r], lo, hi);
        }
}
__global__ void qoco_trajectory_cones(int total, int n, const int* offsets, const int* columns,
    const int* lo, const int* hi, int* outlo, int* outhi) {
    for (int r = n + blockIdx.x * blockDim.x + threadIdx.x; r < total; r += blockDim.x * gridDim.x)
        for (int k = offsets[r]; k < offsets[r + 1]; ++k) {
            const int c = columns[k]; if (r == c) continue;
            qoco_trajectory_union(r, lo[c], hi[c], outlo, outhi);
            qoco_trajectory_union(c, lo[r], hi[r], outlo, outhi);
        }
}
__global__ void qoco_trajectory_tree(int total, int intervals, const int* lo, const int* hi,
    unsigned int* node, int* depth, int* midpoint) {
    for (int v = blockIdx.x * blockDim.x + threadIdx.x; v < total; v += blockDim.x * gridDim.x) {
        int left = 0, right = intervals, d = 0, middle = intervals / 2; unsigned int path = 1;
        if (lo[v] <= hi[v]) while (left < right) {
            middle = left + (right - left) / 2;
            if (hi[v] < middle) { right = middle - 1; path *= 2; ++d; }
            else if (lo[v] > middle) { left = middle + 1; path = path * 2 + 1; ++d; }
            else break;
        }
        if (left == right) middle = left;
        node[v] = path; depth[v] = d; midpoint[v] = middle;
    }
}
// Every edge must join ancestor/descendant separator nodes. A conservative
// root promotion repairs unexpected nonlocal couplings without CPU inspection.
__global__ void qoco_trajectory_guard(int total, const int* offsets, const int* columns,
    const unsigned int* node, const int* depth, int* root) {
    for (int r = blockIdx.x * blockDim.x + threadIdx.x; r < total; r += blockDim.x * gridDim.x)
        for (int k = offsets[r]; k < offsets[r + 1]; ++k) {
            const int c = columns[k];
            const bool related = depth[r] < depth[c]
                ? node[r] == (node[c] >> (depth[c] - depth[r]))
                : node[c] == (node[r] >> (depth[r] - depth[c]));
            if (!related) { atomicExch(root + r, 1); atomicExch(root + c, 1); }
        }
}
__global__ void qoco_trajectory_partition(int total, int levels, const int* state_vertex,
    const int* lo, const int* hi, unsigned int* node, int* depth) {
    for (int v = blockIdx.x * blockDim.x + threadIdx.x; v < total; v += blockDim.x * gridDim.x) {
        int d = depth[v]; unsigned int h = node[v];
        if (d >= levels) { h >>= d - levels + 1; d = levels - 1; }
        if (!state_vertex[v] && lo[v] == hi[v]) { h <<= levels - 1 - d; d = levels - 1; }
        node[v] = h; depth[v] = d;
    }
}
__global__ void qoco_trajectory_keys(int total, int intervals, const int* depth,
    const int* midpoint, const int* degree, const int* root,
    const int* state_vertex, const int* lo, const int* hi,
    unsigned long long* keys, int* identity, int* promoted) {
    for (int v = blockIdx.x * blockDim.x + threadIdx.x; v < total; v += blockDim.x * gridDim.x) {
        const int d = root[v] ? 0 : depth[v], mid = root[v] ? intervals / 2 : midpoint[v];
        // Eliminate interval-local auxiliary variables/cones before the state
        // separator system. Their neighbours lie on one separator ancestor path.
        const bool local = !state_vertex[v] && !root[v] && lo[v] == hi[v];
        keys[v] = local
            ? (static_cast<unsigned long long>(lo[v]) << 27) | min(degree[v], 0x7ffffff)
            : (1ULL << 63) | (static_cast<unsigned long long>(31 - d) << 58)
                | (static_cast<unsigned long long>(mid) << 27) | min(degree[v], 0x7ffffff);
        identity[v] = v; if (root[v]) atomicAdd(promoted, 1);
    }
}
__global__ void qoco_trajectory_tree_keys(int total, int levels, const unsigned int* node,
    const int* depth, const int* degree, const int* root, const int* state_vertex,
    const int* lo, const int* hi, unsigned long long* keys, int* identity,
    int* sizes, int* promoted) {
    for (int v = blockIdx.x * blockDim.x + threadIdx.x; v < total; v += blockDim.x * gridDim.x) {
        const int d = root[v] ? 0 : depth[v];
        const unsigned int h = root[v] ? 1 : node[v];
        const int group = (1 << levels) - (1 << (d + 1)) + h - (1 << d);
        const bool local = !state_vertex[v] && !root[v] && lo[v] == hi[v];
        keys[v] = (static_cast<unsigned long long>(group) << 32)
            | (local ? 0 : (1ULL << 31)) | static_cast<unsigned int>(degree[v]);
        identity[v] = v; atomicAdd(sizes + group, 1);
        if (root[v]) atomicAdd(promoted, 1);
    }
}
extern "C" int qoco_gpu_trajectory_ordering_with_tree(int total, int n, const int* offsets,
    const int* columns, int* permutation, int** tree_sizes, int* tree_levels) {
    if (tree_sizes) *tree_sizes = nullptr;
    if (tree_levels) *tree_levels = 0;
    const auto layout = qoco_next_trajectory; qoco_next_trajectory = {};
    if (layout.intervals == 0) return 0;
    if (n <= 0 || total < n) return -1;
    int levels = 0; int* sizes{};
    if (tree_sizes && tree_levels) {
        for (int remaining = layout.intervals + 1; remaining > 1 && levels < 8; remaining /= 2) ++levels;
        // cuDSS 0.7.1 rejects ND_NLEVELS=1 even for a one-node problem.
        levels = std::max(2, levels);
    }
    int* buffer{}; unsigned long long *keys{}, *sorted{};
    const size_t bytes = static_cast<size_t>(total) * sizeof(int);
    CUDA_CHECK(cudaMalloc(&buffer, bytes * 12 + 2 * sizeof(int)));
    int *stage = buffer, *lo = stage + total, *hi = lo + total, *blo = hi + total,
        *bhi = blo + total, *degree = bhi + total, *depth = degree + total,
        *midpoint = depth + total, *node = midpoint + total, *root = node + total,
        *identity = root + total, *state_vertex = identity + total, *status = state_vertex + total;
    CUDA_CHECK(cudaMemset(buffer, 0, bytes * 12 + 2 * sizeof(int)));
    CUDA_CHECK(cudaMemset(stage, 0xff, bytes));
    const int blocks = std::min(256, (total - 1) / 256 + 1);
    qoco_trajectory_map<<<blocks,256>>>(n, (static_cast<long long>(layout.intervals) + 1) * layout.nx, layout.nx, layout.states, stage, state_vertex, true, status);
    qoco_trajectory_map<<<blocks,256>>>(n, static_cast<long long>(layout.intervals) * layout.nu, layout.nu, layout.controls, stage, state_vertex, false, status);
    qoco_trajectory_map<<<blocks,256>>>(n, static_cast<long long>(layout.intervals) * layout.nx, layout.nx, layout.virtuals, stage, state_vertex, false, status);
    int invalid{}; CUDA_CHECK(cudaMemcpy(&invalid, status, sizeof(int), cudaMemcpyDeviceToHost));
    if (invalid) { CUDA_CHECK(cudaFree(buffer)); return -2; }
    qoco_trajectory_seed<<<blocks,256>>>(total, stage, lo, hi);
    qoco_trajectory_edges<<<blocks,256>>>(total, offsets, columns, stage, lo, hi, degree);
    CUDA_CHECK(cudaMemcpyAsync(blo, lo, bytes, cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpyAsync(bhi, hi, bytes, cudaMemcpyDeviceToDevice));
    // Cone rows without a direct primal coefficient still belong to the same
    // interval. Propagate through the NT block before labeling epigraphs.
    qoco_trajectory_cones<<<blocks,256>>>(total, n, offsets, columns, lo, hi, blo, bhi);
    qoco_trajectory_unmapped<<<blocks,256>>>(n, offsets, columns, stage, blo, bhi, lo, hi);
    qoco_trajectory_duals<<<blocks,256>>>(n, offsets, columns, lo, hi, blo, bhi);
    CUDA_CHECK(cudaMemcpyAsync(blo, lo, n * sizeof(int), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpyAsync(bhi, hi, n * sizeof(int), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpyAsync(lo, blo, bytes, cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpyAsync(hi, bhi, bytes, cudaMemcpyDeviceToDevice));
    qoco_trajectory_cones<<<blocks,256>>>(total, n, offsets, columns, blo, bhi, lo, hi);
    qoco_trajectory_tree<<<blocks,256>>>(total, layout.intervals, lo, hi, reinterpret_cast<unsigned int*>(node), depth, midpoint);
    if (levels) qoco_trajectory_partition<<<blocks,256>>>(total, levels, state_vertex, lo, hi,
        reinterpret_cast<unsigned int*>(node), depth);
    qoco_trajectory_guard<<<blocks,256>>>(total, offsets, columns, reinterpret_cast<unsigned int*>(node), depth, root);
    CUDA_CHECK(cudaMalloc(&keys, static_cast<size_t>(total) * sizeof(*keys)));
    CUDA_CHECK(cudaMalloc(&sorted, static_cast<size_t>(total) * sizeof(*sorted)));
    if (levels) {
        CUDA_CHECK(cudaMalloc(&sizes, ((1 << levels) - 1) * sizeof(int)));
        CUDA_CHECK(cudaMemsetAsync(sizes, 0, ((1 << levels) - 1) * sizeof(int)));
        qoco_trajectory_tree_keys<<<blocks,256>>>(total, levels, reinterpret_cast<unsigned int*>(node),
            depth, degree, root, state_vertex, lo, hi, keys, identity, sizes, status + 1);
    } else qoco_trajectory_keys<<<blocks,256>>>(total, layout.intervals, depth, midpoint, degree, root, state_vertex, lo, hi, keys, identity, status + 1);
    size_t scratch_bytes{}; void* scratch{};
    CUDA_CHECK(cub::DeviceRadixSort::SortPairs(nullptr, scratch_bytes, keys, sorted, identity, permutation, total));
    CUDA_CHECK(cudaMalloc(&scratch, scratch_bytes));
    CUDA_CHECK(cub::DeviceRadixSort::SortPairs(scratch, scratch_bytes, keys, sorted, identity, permutation, total));
    CUDA_CHECK(cudaGetLastError());
    if (const char* audit = std::getenv("SPACEPDHCG_TEST_QOCO_TRAJECTORY_ORDERING"); audit && audit[0] == '1') {
        int promoted{}; CUDA_CHECK(cudaMemcpy(&promoted, status + 1, sizeof(int), cudaMemcpyDeviceToHost));
        fprintf(stderr, "{\"case\":\"qoco_trajectory_ordering\",\"vertices\":%d,\"intervals\":%d,\"root_promotions\":%d}\n", total, layout.intervals, promoted);
    }
    CUDA_CHECK(cudaFree(scratch)); CUDA_CHECK(cudaFree(sorted)); CUDA_CHECK(cudaFree(keys));
    CUDA_CHECK(cudaFree(buffer));
    if (tree_sizes && tree_levels) { *tree_sizes = sizes; *tree_levels = levels; }
    return 1;
}
extern "C" int qoco_gpu_trajectory_ordering(int total, int n, const int* offsets,
    const int* columns, int* permutation) {
    return qoco_gpu_trajectory_ordering_with_tree(total, n, offsets, columns, permutation, nullptr, nullptr);
}
