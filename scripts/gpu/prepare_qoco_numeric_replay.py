#!/usr/bin/env python3
"""Queue numerical updates and consume device scaling in guarded IPM replay."""

# ruff: noqa: E501 -- CUDA replacement literals must match the pinned preparation.
import argparse
import hashlib
import json
import re
from pathlib import Path

from prepare_qoco_device_control import once


def prepare(root):
    path = root / "algebra/cuda"
    update = (path / "qoco_device_update.cuh").read_text()
    update = once(update, "#include <vector>", "#include <vector>\n#include <thread>")
    update = once(update, "    cudaEvent_t ready{};", """    cudaEvent_t ready{};
    bool scale_valid{}, pending{}, host_stale{};
    cudaStream_t pending_stream{};
    int device{};
    std::thread::id owner = std::this_thread::get_id();""")
    update = once(update, "        auto w = std::make_unique<Context>(); w->solver = solver;",
                  "        auto w = std::make_unique<Context>(); w->solver = solver;\n        check(cudaGetDevice(&w->device));")
    update = once(update, "double kinv, double* factors, int* invalid) {",
                  "double kinv, double* factors, int* invalid, const double* previous_kinv = nullptr) {\n    if (previous_kinv) kinv = *previous_kinv;")
    update = once(update, "s->kinv, w->factors.data, w->invalid.data);",
                  "s->kinv, w->factors.data, w->invalid.data, w->scale_valid ? w->result.data + 1 : nullptr);")
    update = once(update, "        s->k = result[0]; s->kinv = result[1];",
                  "        w->scale_valid = true;\n        s->k = result[0]; s->kinv = result[1];")

    # Reuse the exact scaling arithmetic and ordering from the validated update.
    start = update.index("        load_p<<<", update.index('extern "C" int qoco_gpu_update_numeric('))
    end = update.index("        check(cudaGetLastError());", start)
    kernels = update[start:end]
    # Member access (w->...) occurs inside launch dimensions. Matching a single
    # '>' as the delimiter silently leaves those launches on the default stream.
    launches = re.findall(r"<<<(.*?)>>>", kernels, flags=re.DOTALL)
    assert len(launches) == 16, "Pinned numerical-update launch sequence changed"
    kernels = re.sub(r"<<<(.*?)>>>", lambda m: "<<<" + m[1] + ", 0, producer>>>", kernels, flags=re.DOTALL)
    assert all(v.endswith(", 0, producer") for v in re.findall(r"<<<(.*?)>>>", kernels, flags=re.DOTALL))
    kernels = kernels.replace("cudaMemcpyDeviceToDevice));", "cudaMemcpyDeviceToDevice, producer));")
    declarations = update[update.index("    const int n = data->n", update.index('extern "C" int qoco_gpu_update_numeric(')):start]
    declarations = declarations[:declarations.index("    double result[9]{};")]
    queued = """
extern "C" int qoco_gpu_update_kkt_values(QOCOSolver*, cudaStream_t);
extern "C" int qoco_gpu_update_numeric_device(void* opaque, const double* packed,
    cudaStream_t producer, const double** output) {
    using namespace qoco_device_update;
    if (output) *output = nullptr;
    auto* w = static_cast<Context*>(opaque);
    int device = -1;
    if (!w || !packed || !output || cudaGetDevice(&device) != cudaSuccess ||
        device != w->device || w->owner != std::this_thread::get_id()) return 1;
    if (!w->scale_valid) return 2; // Prime numerical scaling synchronously once.
    if (w->pending && producer != w->pending_stream) return 2;
    cudaStreamCaptureStatus capture;
    if (cudaStreamIsCapturing(producer, &capture) != cudaSuccess) return 4;
    if (capture != cudaStreamCaptureStatusNone) return 2;
    auto* solver = w->solver; auto* data = solver->work->data; auto* s = solver->work->scaling;
""" + declarations + """
    w->pending = true; w->host_stale = true; w->pending_stream = producer;
    solver->sol->status = QOCO_UNSOLVED;
#ifdef SPACEPDHCG_QOCO_DEFERRED_TRANSPOSES
    for (auto* matrix : {data->At, data->Gt})
        if (matrix && matrix->transpose_source) matrix->transpose_values_pending = 1;
#endif
    try {
""" + kernels + """
        check(cudaGetLastError());
        if (qoco_gpu_update_kkt_values(solver, producer)) return 4;
        *output = w->result.data;
        return 0;
    } catch (const Failure&) { return 4; }
}
extern "C" int qoco_gpu_finish_numeric_update(void* opaque, int materialize) {
    using namespace qoco_device_update;
    auto* w = static_cast<Context*>(opaque);
    int device = -1;
    if (!w || cudaGetDevice(&device) != cudaSuccess || device != w->device ||
        w->owner != std::this_thread::get_id() || (materialize != 0 && materialize != 1)) return 1;
    if (!w->pending && !(materialize && w->host_stale)) return 0;
    double result[9]{};
    if (materialize && cudaMemcpyAsync(result,w->result.data,sizeof(result),
        cudaMemcpyDeviceToHost,w->pending_stream)!=cudaSuccess) {
        cudaStreamSynchronize(w->pending_stream); return 4;
    }
    if (cudaStreamSynchronize(w->pending_stream)!=cudaSuccess) return 4;
    w->pending = false;
    if (!materialize) return 0;
    if (result[8] != 0) return 3;
    w->host_stale = false;
    auto* s = w->solver->work->scaling; auto* data = w->solver->work->data;
    s->k=result[0]; s->kinv=result[1];
    data->obj_range_min=result[2]; data->obj_range_max=result[3];
    data->constraint_range_min=result[4]; data->constraint_range_max=result[5];
    data->rhs_range_min=result[6]; data->rhs_range_max=result[7];
    return 0;
}
"""
    update += queued
    update = once(update, 'extern "C" int qoco_gpu_update_numeric(void* opaque',
                  'extern "C" int qoco_gpu_finish_numeric_update(void*, int);\nextern "C" int qoco_gpu_update_numeric(void* opaque')
    update = once(update, "    if (!w || !packed) return 1;", """    if (!w || !packed) return 1;
    const int finished = qoco_gpu_finish_numeric_update(w,1);
    if (finished) return finished;""")
    update = once(update, "    delete static_cast<qoco_device_update::Context*>(opaque);",
                  "    qoco_gpu_finish_numeric_update(opaque,0);\n    delete static_cast<qoco_device_update::Context*>(opaque);")

    backend = (path / "cudss_backend.cu").read_text()
    export = """
extern "C" int qoco_gpu_update_kkt_values(QOCOSolver* solver, cudaStream_t stream) {
    auto* s = solver->linsys_data; auto* data = solver->work->data;
    QOCOMatrix* matrices[] = {data->P, data->A, data->G};
    const QOCOInt* maps[] = {s->d_PregtoKKTcsr, s->d_AttoKKTcsr, s->d_GttoKKTcsr};
    for (int i=0;i<3;++i) if (matrices[i] && maps[i]) {
        const auto* matrix = matrices[i]->d_csc_host;
        if (matrix && matrix->nnz) update_csr_matrix_data_kernel<<<(matrix->nnz+255)/256,256,0,stream>>>(
            matrix->x,s->d_csr_val,maps[i],matrix->nnz);
    }
    return cudaGetLastError()==cudaSuccess ? 0 : 4;
}
"""
    backend = once(backend, '#include "qoco_ipm_graph.cuh"', export + '\n#include "qoco_ipm_graph.cuh"')
    parameters = (path / "qoco_ipm_parameters.cuh").read_text()
    parameters = once(parameters, "    int warm_start;", "    int warm_start;\n    int update_invalid;")
    parameters += """
static __global__ void qoco_ipm_parameters_from_update(QocoIpmParameters* target,
    QocoIpmParameters value, const double* update) {
    value.k=update[0]; value.kinv=update[1];
    value.update_invalid=update[8]!=0 || !isfinite(value.k) || !isfinite(value.kinv);
    *target=value;
}
"""
    graph = (path / "qoco_ipm_graph.cuh").read_text()
    guard = """
__global__ void qoco_ipm_update_guard(const QocoIpmParameters* p,
    cudaGraphConditionalHandle handle,QocoGpuCompletion* result) {
    cudaGraphSetConditional(handle,!p->update_invalid);
    if (p->update_invalid) *result={1,QOCO_NUMERICAL_ERROR,0,0,0,0,
        INFINITY,INFINITY,INFINITY,NAN,p->dynamic_reg};
}
"""
    graph = guard + graph
    graph = once(graph, "    const auto root = capture.graph;\n    qoco_ipm_current = &capture;", """    const auto root = capture.graph;
    cudaGraphConditionalHandle valid_update;
    CUDA_CHECK(cudaGraphConditionalHandleCreate(&valid_update,root,1,cudaGraphCondAssignDefault));
    qoco_ipm_current = &capture;
    qoco_ipm_resume();
    qoco_ipm_update_guard<<<1,1>>>(cache.parameters,valid_update,cache.completion);
    qoco_ipm_pause();
    cudaGraphNodeParams guard_params{};
    guard_params.type=cudaGraphNodeTypeConditional;
    guard_params.conditional.handle=valid_update;
    guard_params.conditional.type=cudaGraphCondTypeIf;
    guard_params.conditional.size=1;
    cudaGraphNode_t guard_node;
    CUDA_CHECK(cudaGraphAddNode(&guard_node,root,capture.dependencies.data(),capture.dependencies.size(),&guard_params));
    const auto solve_graph=guard_params.conditional.phGraph_out[0];
    capture.graph=solve_graph; capture.dependencies.clear();""")
    graph = once(graph, "cudaGraphConditionalHandleCreate(&loop, root,", "cudaGraphConditionalHandleCreate(&loop, solve_graph,")
    graph = once(graph, "cudaGraphAddNode(&node, root,", "cudaGraphAddNode(&node, solve_graph,")
    graph = once(graph, "        capture.graph = root;", "        capture.graph = solve_graph;")
    replay = (path / "qoco_ipm_replay.cuh").read_text()
    replay = once(replay, 'extern "C" int qoco_gpu_ipm_replay_device(QOCOSolver* solver, void* stream_pointer, QocoGpuOutput* output) {',
                  'static int qoco_gpu_ipm_replay_impl(QOCOSolver* solver, void* stream_pointer, QocoGpuOutput* output, const double* numeric_update) {')
    replay = once(replay, "    qoco_ipm_parameters_set<<<1, 1, 0, stream>>>(cache.parameters, parameters);", """    if (numeric_update) qoco_ipm_parameters_from_update<<<1,1,0,stream>>>(cache.parameters,parameters,numeric_update);
    else qoco_ipm_parameters_set<<<1, 1, 0, stream>>>(cache.parameters, parameters);""")
    replay += """
extern "C" int qoco_gpu_ipm_replay_device(QOCOSolver* solver,void* stream,QocoGpuOutput* output) {
    return qoco_gpu_ipm_replay_impl(solver,stream,output,nullptr);
}
extern "C" int qoco_gpu_ipm_replay_updated_device(QOCOSolver* solver,void* stream,
    const double* numeric_update,QocoGpuOutput* output) {
    if (!numeric_update) { if (output) *output={}; return 1; }
    return qoco_gpu_ipm_replay_impl(solver,stream,output,numeric_update);
}
"""
    header = (path / "qoco_gpu_replay.h").read_text()
    header = once(header, "#ifdef __cplusplus\n}", """// Queued numeric updates require one successful synchronous numeric update first.
// The borrowed nine-double result contains k,kinv,three min/max pairs,invalid.
// Queue updated replay and consumers on the SAME stream, then finish both APIs.
// No other solver APIs may run while update/replay work remains pending.
// materialize=1 refreshes legacy host scale/range metadata before host API use;
// materialize=0 waits only and permits subsequent device-update/replay chains.
// An invalid update causes updated replay to skip all IPM factor/solve work and
// publish numerical-error completion. Rebuild after such an update failure.
int qoco_gpu_update_numeric_device(void*,const double*,cudaStream_t,const double**);
int qoco_gpu_finish_numeric_update(void*,int materialize);
int qoco_gpu_ipm_replay_updated_device(QOCOSolver*,void*,const double*,QocoGpuOutput*);
#ifdef __cplusplus
}""")
    files = {"qoco_device_update.cuh": update, "cudss_backend.cu": backend,
             "qoco_ipm_parameters.cuh": parameters, "qoco_ipm_graph.cuh": graph,
             "qoco_ipm_replay.cuh": replay, "qoco_gpu_replay.h": header}
    for name, text in files.items():
        (path / name).write_text(text)
    return {name: hashlib.sha256((path / name).read_bytes()).hexdigest() for name in files}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.destination), indent=2))
