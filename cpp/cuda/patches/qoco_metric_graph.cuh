// SPDX-License-Identifier: Apache-2.0
// Capture only the existing device calculation. Host result copies/decisions
// stay outside the graph. Cache lifetime is bounded by the reduction scope;
// scratch growth destroys graphs before freeing their captured storage.
#include "qoco.h"
#include <array>
#include <cstdlib>

namespace qoco_metric_graph {
struct Key {
    std::array<const void*, 48> pointers{};
    std::array<int, 3> dimensions{};
    std::array<double, 3> scalars{};
    bool operator==(const Key& b) const {
        return pointers == b.pointers && dimensions == b.dimensions && scalars == b.scalars;
    }
};
struct Cache {
    Key key{};
    bool valid{}, primed{};
    cudaStream_t stream{};
    cudaGraph_t graph{};
    cudaGraphExec_t executable{};
};
static thread_local Cache caches[2];
static thread_local unsigned long long captures{}, launches{}, uncaptured{}, invalidations{};

static void clear(Cache& cache) {
    if (cache.valid) ++invalidations;
    if (cache.executable) CUDA_CHECK(cudaGraphExecDestroy(cache.executable));
    if (cache.graph) CUDA_CHECK(cudaGraphDestroy(cache.graph));
    if (cache.stream) CUDA_CHECK(cudaStreamDestroy(cache.stream));
    cache = {};
}
static Key make_key(QOCOSolver* solver, double* scratch, cublasHandle_t handle) {
    auto* w = solver->work; auto* d = w->data; auto* scale = w->scaling;
    Key key{};
    key.dimensions = {d->n, d->p, d->m};
    key.scalars = {scale->kinv, scale->k, solver->settings->kkt_static_reg_P};
    key.pointers = {solver, w, d, scratch, handle,
        w->xbuff->d_data, w->ybuff->d_data, w->ubuff1->d_data, w->ubuff2->d_data,
        w->ubuff3->d_data, w->kktres->d_data, w->x->d_data, w->y->d_data,
        w->z->d_data, w->s->d_data, d->c->d_data, d->b->d_data, d->h->d_data,
        scale->Dinvruiz->d_data, scale->Einvruiz->d_data, scale->Finvruiz->d_data,
        scale->Fruiz->d_data, d->P->d_csc, d->A->d_csc, d->G->d_csc,
        d->P->gather->offsets, d->P->gather->entries, d->P->gather->columns,
        d->A->gather->offsets, d->A->gather->entries, d->A->gather->columns,
        d->G->gather->offsets, d->G->gather->entries, d->G->gather->columns};
    return key;
}
template<bool Iteration, class Enqueue>
static void run(QOCOSolver* solver, double* scratch, cublasHandle_t handle, Enqueue enqueue) {
    const char* disable = std::getenv("SPACEPDHCG_TEST_QOCO_METRIC_GRAPH_DISABLE");
    if (qoco_reduction_scope_depth < 2 || (disable && disable[0] == '1')) {
        ++uncaptured; enqueue(); return;
    }
    auto& cache = caches[Iteration ? 1 : 0];
    const auto key = make_key(solver, scratch, handle);
    if (!cache.valid || !(cache.key == key)) {
        clear(cache); cache.key = key; cache.valid = true;
    }
    // Run once normally to initialize library internals before capture.
    if (!cache.primed) { ++uncaptured; enqueue(); cache.primed = true; return; }
    if (!cache.executable) {
        auto* funcs = get_cuda_funcs();
        cudaStream_t previous{};
        qoco_batched_stopping::blas(funcs->cublasGetStream(handle, &previous));
        CUDA_CHECK(cudaStreamCreateWithFlags(&cache.stream, cudaStreamNonBlocking));
        qoco_batched_stopping::blas(funcs->cublasSetStream(handle, cache.stream));
        CUDA_CHECK(cudaStreamBeginCapture(cache.stream, cudaStreamCaptureModeThreadLocal));
        qoco_metric_stream = cache.stream;
        enqueue();
        qoco_metric_stream = nullptr;
        CUDA_CHECK(cudaStreamEndCapture(cache.stream, &cache.graph));
        qoco_batched_stopping::blas(funcs->cublasSetStream(handle, previous));
        CUDA_CHECK(cudaGraphInstantiate(&cache.executable, cache.graph, nullptr, nullptr, 0));
        ++captures;
    }
    // Default-stream replay preserves ordering with residuals and the scalar
    // download. The dedicated stream is used for capture, never for execution.
    CUDA_CHECK(cudaGraphLaunch(cache.executable, nullptr));
    ++launches;
}
} // namespace qoco_metric_graph

static void qoco_release_metric_graphs() {
    for (auto& cache : qoco_metric_graph::caches) qoco_metric_graph::clear(cache);
}
extern "C" void qoco_gpu_metric_graph_stats(unsigned long long* out) {
    using namespace qoco_metric_graph;
    out[0] = captures; out[1] = launches; out[2] = uncaptured; out[3] = invalidations;
    out[4] = (caches[0].executable != nullptr) + (caches[1].executable != nullptr);
    out[5] = (caches[0].stream != nullptr) + (caches[1].stream != nullptr);
}
