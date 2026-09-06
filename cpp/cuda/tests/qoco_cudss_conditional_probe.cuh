// Diagnostic only: injected into an isolated prepared cuDSS backend.
// Tests actual, already factorized trajectory systems without changing the
// production refinement algorithm. Never included by a normal QOCO build.
#include <vector>
#include <cstring>
#include <cmath>
#include <algorithm>

namespace qoco_conditional_probe {
__global__ void start(cudaGraphConditionalHandle handle, int* count, int limit) {
    *count = 0;
    cudaGraphSetConditional(handle, limit > 0);
}
__global__ void advance(cudaGraphConditionalHandle handle, int* count, int limit) {
    cudaGraphSetConditional(handle, ++*count < limit);
}

static bool inventory(cudaGraph_t graph, int depth = 0) {
    size_t size = 0;
    CUDA_CHECK(cudaGraphGetNodes(graph, nullptr, &size));
    std::vector<cudaGraphNode_t> nodes(size);
    CUDA_CHECK(cudaGraphGetNodes(graph, nodes.data(), &size));
    bool allowed = true;
    int types[20] = {};
    for (auto node : nodes) {
        cudaGraphNodeType type;
        CUDA_CHECK(cudaGraphNodeGetType(node, &type));
        if (int(type) < 20) ++types[int(type)];
        if (type == cudaGraphNodeTypeMemcpy) {
            cudaMemcpy3DParms copy = {};
            CUDA_CHECK(cudaGraphMemcpyNodeGetParams(node, &copy));
            cudaPointerAttributes src = {}, dst = {};
            const auto src_status = cudaPointerGetAttributes(&src, copy.srcPtr.ptr);
            const auto dst_status = cudaPointerGetAttributes(&dst, copy.dstPtr.ptr);
            (void)cudaGetLastError();
            fprintf(stderr, "CONDITIONAL_PROBE copy kind=%d bytes=%zu src_status=%d src_type=%d dst_status=%d dst_type=%d\n",
                    int(copy.kind), copy.extent.width, int(src_status), int(src.type), int(dst_status), int(dst.type));
        }
        if (type == cudaGraphNodeTypeGraph) {
            cudaGraph_t child;
            CUDA_CHECK(cudaGraphChildGraphNodeGetGraph(node, &child));
            allowed = inventory(child, depth + 1) && allowed;
        } else if (type != cudaGraphNodeTypeKernel &&
                   type != cudaGraphNodeTypeEmpty &&
                   type != cudaGraphNodeTypeMemcpy &&
                   type != cudaGraphNodeTypeMemset) {
            allowed = false;
        }
    }
    fprintf(stderr, "CONDITIONAL_PROBE nodes depth=%d total=%zu allowed=%d types=",
            depth, size, allowed);
    for (int i = 0; i < 20; ++i) if (types[i]) fprintf(stderr, "%d:%d,", i, types[i]);
    fprintf(stderr, "\n");
    return allowed;
}

static bool compare(const std::vector<double>& expected, const std::vector<double>& actual) {
    double error = 0, scale = 1;
    for (size_t i = 0; i < expected.size(); ++i) {
        if (!std::isfinite(expected[i]) || !std::isfinite(actual[i])) return false;
        error = std::max(error, std::abs(expected[i] - actual[i]));
        scale = std::max(scale, std::abs(expected[i]));
    }
    const bool bitwise = std::memcmp(expected.data(), actual.data(), expected.size() * sizeof(double)) == 0;
    fprintf(stderr, "CONDITIONAL_PROBE parity bitwise=%d abs_error=%.17g scale=%.17g scaled_error=%.17g\n",
            bitwise, error, scale, error / scale);
    // cuDSS's normal solve uses atomics: bitwise equality is diagnostic only.
    // This is a solve-replay comparison, not a replacement physics certificate.
    return error <= 1e-12 * scale;
}

static std::vector<void*> snapshot_host_inputs(cudaGraph_t graph) {
    std::vector<void*> owned;
    size_t size = 0;
    CUDA_CHECK(cudaGraphGetNodes(graph, nullptr, &size));
    std::vector<cudaGraphNode_t> nodes(size);
    CUDA_CHECK(cudaGraphGetNodes(graph, nodes.data(), &size));
    for (auto node : nodes) {
        cudaGraphNodeType type;
        CUDA_CHECK(cudaGraphNodeGetType(node, &type));
        if (type != cudaGraphNodeTypeMemcpy) continue;
        cudaMemcpy3DParms copy = {};
        CUDA_CHECK(cudaGraphMemcpyNodeGetParams(node, &copy));
        if (copy.kind != cudaMemcpyHostToDevice) continue;
        // Deliberately narrow diagnostic: two scalar inputs seen in cuDSS0.8.
        // Never turn arbitrary host-copy nodes into stale numerical constants.
        if (copy.srcArray || copy.dstArray || copy.extent.width != 8 ||
            copy.extent.height != 1 || copy.extent.depth != 1 ||
            copy.srcPos.x || copy.srcPos.y || copy.srcPos.z ||
            copy.dstPos.x || copy.dstPos.y || copy.dstPos.z) exit(100);
        unsigned long long bits = 0;
        std::memcpy(&bits, copy.srcPtr.ptr, 8);
        fprintf(stderr, "CONDITIONAL_PROBE snapshot bytes=8 bits=%016llx\n", bits);
        void* device;
        CUDA_CHECK(cudaMalloc(&device, 8));
        CUDA_CHECK(cudaMemcpy(device, copy.srcPtr.ptr, 8, cudaMemcpyHostToDevice));
        owned.push_back(device);
        copy.srcPtr.ptr = device;
        copy.kind = cudaMemcpyDeviceToDevice;
        CUDA_CHECK(cudaGraphMemcpyNodeSetParams(node, &copy));
    }
    return owned;
}

static void run(LinSysData* s, QOCOWorkspace* work, const double* b, const double* initial_x) {
    const char* setting = getenv("SPACEPDHCG_TEST_QOCO_CONDITIONAL_PROBE");
    if (!setting) return;
    static thread_local int calls = 0;
    if (++calls > atoi(setting)) return;
    CUDA_CHECK(cudaDeviceSynchronize());
    fprintf(stderr, "CONDITIONAL_PROBE begin call=%d Kn=%d\n", calls, int(s->Kn));
    const size_t bytes = s->Kn * sizeof(double);
    std::vector<double> expected(s->Kn), actual(s->Kn);
    CUDA_CHECK(cudaMemcpy(expected.data(), initial_x, bytes, cudaMemcpyDeviceToHost));
    const double initial_residual = compute_linsys_residual(s, work, b, initial_x, s->d_xyz_matrix_data);
    fprintf(stderr, "CONDITIONAL_PROBE initial true_kkt_inf=%.17g\n", initial_residual);
    double* direct;
    CUDA_CHECK(cudaMalloc(&direct, bytes));
    for (int repeat = 0; repeat < 2; ++repeat) {
        cudss_solve_system(s, b, direct);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(actual.data(), direct, bytes, cudaMemcpyDeviceToHost));
        const bool parity = compare(expected, actual);
        const double norm = compute_linsys_residual(s, work, b, direct, s->d_xyz_matrix_data);
        fprintf(stderr, "CONDITIONAL_PROBE direct repeat=%d parity=%d true_kkt_inf=%.17g\n", repeat, parity, norm);
    }
    CUDA_CHECK(cudaFree(direct));
    auto set_stream = reinterpret_cast<decltype(&::cudssSetStream)>(
        dlsym(g_cudss_handle, "cudssSetStream"));
    if (!set_stream) { fprintf(stderr, "CONDITIONAL_PROBE missing stream API\n"); exit(95); }
    cudaStream_t stream;
#ifdef QOCO_PROBE_EARLY_STREAM
    stream = qoco_probe_streams.at(s->handle);
#else
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUDSS_CHECK(set_stream(s->handle, stream));
#endif
    int* count;
    CUDA_CHECK(cudaMalloc(&count, sizeof(int)));
    cudaGraph_t solve_graph;
    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    CUDA_CHECK(cudaMemcpyAsync(s->d_rhs_matrix_data, b, bytes, cudaMemcpyDeviceToDevice, stream));
    CUDA_CHECK(cudaMemsetAsync(s->d_xyz_matrix_data, 0, bytes, stream));
    const auto vendor_status = g_cuda_funcs.cudssExecute(
        s->handle, CUDSS_PHASE_SOLVE, s->config, s->data, s->K_csr,
        s->d_xyz_matrix, s->d_rhs_matrix);
    const auto capture_status = cudaStreamEndCapture(stream, &solve_graph);
    fprintf(stderr, "CONDITIONAL_PROBE capture cudss=%d cuda=%d\n",
            int(vendor_status), int(capture_status));
    if (vendor_status != CUDSS_STATUS_SUCCESS || capture_status != cudaSuccess) exit(96);
    if (!inventory(solve_graph)) exit(97);
    std::vector<void*> snapshots;
    if (getenv("SPACEPDHCG_TEST_QOCO_CONDITIONAL_SNAPSHOT")) snapshots = snapshot_host_inputs(solve_graph);

    cudaGraphExec_t ordinary;
    CUDA_CHECK(cudaGraphInstantiate(&ordinary, solve_graph, 0));
    CUDA_CHECK(cudaGraphLaunch(ordinary, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaMemcpy(actual.data(), s->d_xyz_matrix_data, bytes, cudaMemcpyDeviceToHost));
    const bool ordinary_equal = compare(expected, actual);
    fprintf(stderr, "CONDITIONAL_PROBE ordinary parity=%d\n", ordinary_equal);
    CUDA_CHECK(cudaGraphExecDestroy(ordinary));
    int parity_failures = !ordinary_equal;
    const double ordinary_residual = compute_linsys_residual(s, work, b, s->d_xyz_matrix_data, s->d_xyz_matrix_data);
    fprintf(stderr, "CONDITIONAL_PROBE ordinary true_kkt_inf=%.17g\n", ordinary_residual);

    // Reuse the captured solve for zero, one, and multiple device-controlled
    // executions. Repeated launches also test conditional reset semantics.
    for (int limit : {0, 1, 3, 3}) {
        cudaGraph_t graph;
        CUDA_CHECK(cudaGraphCreate(&graph, 0));
        cudaGraphConditionalHandle handle;
        CUDA_CHECK(cudaGraphConditionalHandleCreate(&handle, graph, 0, cudaGraphCondAssignDefault));
        CUDA_CHECK(cudaStreamBeginCaptureToGraph(stream, graph, nullptr, nullptr, 0,
                                                cudaStreamCaptureModeThreadLocal));
        start<<<1, 1, 0, stream>>>(handle, count, limit);
        CUDA_CHECK(cudaGetLastError());
        cudaGraph_t ended;
        CUDA_CHECK(cudaStreamEndCapture(stream, &ended));
        size_t root_count = 0;
        CUDA_CHECK(cudaGraphGetNodes(graph, nullptr, &root_count));
        std::vector<cudaGraphNode_t> roots(root_count);
        CUDA_CHECK(cudaGraphGetNodes(graph, roots.data(), &root_count));
        cudaGraphNodeParams params = {};
        params.type = cudaGraphNodeTypeConditional;
        params.conditional.handle = handle;
        params.conditional.type = cudaGraphCondTypeWhile;
        params.conditional.size = 1;
        cudaGraphNode_t conditional, solve;
        CUDA_CHECK(cudaGraphAddNode(&conditional, graph, roots.data(), root_count, &params));
        const cudaGraph_t body = params.conditional.phGraph_out[0];
        CUDA_CHECK(cudaGraphAddChildGraphNode(&solve, body, nullptr, 0, solve_graph));
        CUDA_CHECK(cudaStreamBeginCaptureToGraph(stream, body, &solve, nullptr, 1,
                                                cudaStreamCaptureModeThreadLocal));
        advance<<<1, 1, 0, stream>>>(handle, count, limit);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaStreamEndCapture(stream, &ended));
        cudaGraphExec_t executable;
        char log[4096] = {};
        cudaGraphNode_t error_node = nullptr;
        const auto instantiate = cudaGraphInstantiate(&executable, graph, &error_node, log, sizeof(log));
        fprintf(stderr, "CONDITIONAL_PROBE instantiate cuda=%d log=%s\n", int(instantiate), log);
        CUDA_CHECK(instantiate);
        for (int replay = 0; replay < 2; ++replay) {
            std::vector<double> unchanged(s->Kn);
            if (!limit) CUDA_CHECK(cudaMemcpy(unchanged.data(), s->d_xyz_matrix_data, bytes, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaGraphLaunch(executable, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
            int actual_count = -1;
            CUDA_CHECK(cudaMemcpy(&actual_count, count, sizeof(int), cudaMemcpyDeviceToHost));
            bool equal = true;
            if (limit) {
                CUDA_CHECK(cudaMemcpy(actual.data(), s->d_xyz_matrix_data, bytes, cudaMemcpyDeviceToHost));
                equal = compare(expected, actual);
                parity_failures += !equal;
                const double norm = compute_linsys_residual(s, work, b, s->d_xyz_matrix_data, s->d_xyz_matrix_data);
                fprintf(stderr, "CONDITIONAL_PROBE conditional true_kkt_inf=%.17g\n", norm);
            } else {
                CUDA_CHECK(cudaMemcpy(actual.data(), s->d_xyz_matrix_data, bytes, cudaMemcpyDeviceToHost));
                equal = std::memcmp(unchanged.data(), actual.data(), bytes) == 0;
                if (!equal) exit(98);
            }
            fprintf(stderr, "CONDITIONAL_PROBE result call=%d limit=%d replay=%d count=%d parity=%d\n",
                    calls, limit, replay, actual_count, equal);
            if (actual_count != limit) exit(98);
        }
        CUDA_CHECK(cudaGraphExecDestroy(executable));
        CUDA_CHECK(cudaGraphDestroy(graph));
    }
    CUDA_CHECK(cudaGraphDestroy(solve_graph));
    for (auto buffer : snapshots) CUDA_CHECK(cudaFree(buffer));
    CUDA_CHECK(cudaFree(count));
#ifndef QOCO_PROBE_EARLY_STREAM
    CUDSS_CHECK(set_stream(s->handle, nullptr));
    CUDA_CHECK(cudaStreamDestroy(stream));
#endif
    // x is untouched. Production residual assembly overwrites vendor scratch.
    fprintf(stderr, "CONDITIONAL_PROBE complete call=%d controls_pass=1 parity_failures=%d\n", calls, parity_failures);
}
} // namespace qoco_conditional_probe
