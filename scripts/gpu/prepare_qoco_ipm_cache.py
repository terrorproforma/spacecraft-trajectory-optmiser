#!/usr/bin/env python3
"""Retain v121 IPM graphs/resources and consume changing settings on device."""
# ruff: noqa: E501 -- CUDA replacement literals preserve the validated generated source.

import argparse
import hashlib
import json
from pathlib import Path

from prepare_qoco_device_control import once


def prepare(root):
    runtime = (root / "algebra/cuda/qoco_ir_runtime.cuh").read_text()
    runtime = once(
        runtime,
        "struct QocoIrRuntime {",
        '#include "qoco_ipm_cache.cuh"\n\nstruct QocoIrRuntime {\n    QocoIpmCache ipm;',
    )
    algebra = (root / "algebra/cuda/cuda_linalg.cu").read_text()
    algebra = once(
        algebra,
        '#include "qoco_batched_stopping.cuh"',
        '#include "qoco_ipm_parameters.cuh"\n#include "qoco_batched_stopping.cuh"',
    )
    algebra = once(
        algebra,
        '#include "qoco_ipm_control.cuh"',
        '#include "qoco_ipm_resources.cuh"\n#include "qoco_ipm_control.cuh"',
    )
    backend = (root / "algebra/cuda/cudss_backend.cu").read_text()
    backend = once(
        backend,
        "static void cudss_cleanup(LinSysData* linsys_data)\n{",
        "static void cudss_cleanup(LinSysData* linsys_data)\n{\n"
        "  qoco_ipm_cache_destroy(linsys_data->ir->ipm);",
    )

    metrics = (root / "algebra/cuda/qoco_batched_stopping.cuh").read_text()
    metrics = once(metrics, "__global__ void finish(", "__device__ void finish_values(")
    metrics = once(
        metrics,
        "} // namespace qoco_batched_stopping",
        """template<bool Iteration>
__global__ void finish(const double* v, double kinv, double k, double reg, int m, double* out) {
    finish_values<Iteration>(v, kinv, k, reg, m, out);
}
template<bool Iteration>
__global__ void finish_ipm(const double* v, const QocoIpmParameters* p, double reg, int m, double* out) {
    finish_values<Iteration>(v, p->kinv, p->k, reg, m, out);
}
} // namespace qoco_batched_stopping""",
    )
    metrics = once(
        metrics,
        "bool device = false) {",
        "bool device = false, const QocoIpmParameters* ipm = nullptr) {",
    )
    metrics = once(
        metrics,
        "    scale_arrayf(xb, xb, scale->kinv, d->n);",
        "    if (ipm) qoco_ipm_scale<<<std::min(256, (d->n + 255) / 256), 256>>>(xb, d->n, ipm);\n"
        "    else scale_arrayf(xb, xb, scale->kinv, d->n);",
    )
    metrics = once(
        metrics,
        "    finish<Iteration><<<1, 1, 0, qoco_metric_stream>>>(scratch, scale->kinv, scale->k,",
        "    if (ipm) finish_ipm<Iteration><<<1, 1, 0, qoco_metric_stream>>>(scratch, ipm,\n"
        "        solver->settings->kkt_static_reg_P, d->m, result);\n"
        "    else finish<Iteration><<<1, 1, 0, qoco_metric_stream>>>(scratch, scale->kinv, scale->k,",
    )
    control = (root / "algebra/cuda/qoco_ipm_control.cuh").read_text()
    control = once(
        control,
        "    double absolute, double relative, double inaccurate_absolute, double inaccurate_relative) {\n"
        "    decide_impl(state, absolute, relative, inaccurate_absolute, inaccurate_relative, *iteration);",
        "    const QocoIpmParameters* p) {\n"
        "    decide_impl(state, p->absolute, p->relative, p->inaccurate_absolute, p->inaccurate_relative, *iteration);",
    )
    control = once(
        control,
        "State* state, int* iteration, int maximum,",
        "State* state, int* iteration, const QocoIpmParameters* p,",
    )
    control = once(control, "if (*iteration >= maximum)", "if (*iteration >= p->maximum)")
    control = once(
        control,
        "    cudaGraphConditionalHandle loop, cudaGraphConditionalHandle step) {",
        "    cudaGraphConditionalHandle loop, cudaGraphConditionalHandle step, const QocoIpmParameters* p) {",
    )
    control = once(
        control,
        "    qoco_gpu_metrics<true>(solver, state->metrics, true);\n    qoco_device_control::ipm_check",
        "    qoco_gpu_metrics<true>(solver, state->metrics, true, p);\n    qoco_device_control::ipm_check",
    )
    control = once(control, "    auto* settings = solver->settings;\n", "")
    control = once(
        control,
        "        settings->abstol, settings->reltol, settings->abstol_inacc, settings->reltol_inacc);",
        "        p);",
    )
    control = once(
        control,
        "                                     cudaGraphConditionalHandle loop) {",
        "                                     cudaGraphConditionalHandle loop, const QocoIpmParameters* p) {",
    )
    control = once(
        control,
        "        iteration, solver->settings->max_iters, loop);",
        "        iteration, p, loop);",
    )

    graph = (root / "algebra/cuda/qoco_ipm_graph.cuh").read_text()
    graph = once(
        graph,
        "    cudaGraph_t linear{};",
        "    cudaGraph_t linear{};\n    QocoIpmParameters* parameters{};",
    )
    graph = once(
        graph,
        "                                    cudaGraphConditionalHandle);",
        "                                    cudaGraphConditionalHandle, const QocoIpmParameters*);",
    )
    graph = once(
        graph,
        'extern "C" void qoco_gpu_ipm_advance(QOCOSolver*, int*, cudaGraphConditionalHandle);',
        'extern "C" void qoco_gpu_ipm_advance(QOCOSolver*, int*, cudaGraphConditionalHandle, const QocoIpmParameters*);',
    )
    graph = once(
        graph,
        "    qoco_ir_set_parameters<<<1, 1>>>(runtime->cache.parameters, tolerance, maximum);",
        "    (void)tolerance; (void)maximum;\n"
        "    qoco_ipm_ir_parameters<<<1, 1>>>(runtime->cache.parameters, c.parameters);",
    )
    graph = once(
        graph,
        "    qoco_ir_prepare_sparse(data);",
        """    auto& cache = s->ir->ipm;
    cache.resources = qoco_gpu_ipm_resources_enter(cache.resources);
    if (!cache.parameters) {
        CUDA_CHECK(cudaMalloc(&cache.parameters, sizeof(QocoIpmParameters)));
        CUDA_CHECK(cudaMalloc(&cache.iterations, sizeof(int)));
    }
    auto* settings = solver->settings;
    const QocoIpmParameters parameters{work->scaling->k, work->scaling->kinv,
        settings->abstol, settings->reltol, settings->abstol_inacc, settings->reltol_inacc,
        settings->ir_tol, settings->max_iters, settings->max_ir_iters};
    qoco_ipm_parameters_set<<<1, 1>>>(cache.parameters, parameters);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemsetAsync(cache.iterations, 0, sizeof(int)));
    // QOCO's fixed-topology workspace owns every captured operand. Static
    // regularisation changes are rare and invalidate rather than freeze a value.
    if (cache.executable && (cache.work != work || cache.control != work->gpu_control ||
        cache.static_p != settings->kkt_static_reg_P || cache.static_a != settings->kkt_static_reg_A ||
        cache.static_g != settings->kkt_static_reg_G || cache.linsys_static_p != s->kkt_static_reg_P)) {
        CUDA_CHECK(cudaDeviceSynchronize());
        qoco_ipm_cache_clear(cache);
    }
    if (!cache.executable) {
    qoco_ir_prepare_sparse(data);""",
    )
    graph = once(
        graph,
        "    QocoIpmCapture capture;",
        "    QocoIpmCapture capture;\n    capture.parameters = cache.parameters;",
    )
    graph = once(
        graph,
        "    int* iterations = nullptr;\n"
        "    CUDA_CHECK(cudaMalloc(&iterations, sizeof(int)));\n"
        "    CUDA_CHECK(cudaMemset(iterations, 0, sizeof(int)));",
        "    int* iterations = cache.iterations;",
    )
    graph = once(
        graph,
        "qoco_gpu_ipm_check(solver, iterations, loop, step);",
        "qoco_gpu_ipm_check(solver, iterations, loop, step, cache.parameters);",
    )
    graph = once(
        graph,
        "qoco_gpu_ipm_advance(solver, iterations, loop);",
        "qoco_gpu_ipm_advance(solver, iterations, loop, cache.parameters);",
    )
    graph = once(
        graph,
        "    CUDA_CHECK(cudaGraphLaunch(executable, cudaStreamPerThread));",
        """    cache.executable = executable;
    cache.graph = root; cache.linear = capture.linear;
    cache.work = work; cache.control = work->gpu_control;
    cache.static_p = settings->kkt_static_reg_P; cache.static_a = settings->kkt_static_reg_A;
    cache.static_g = settings->kkt_static_reg_G; cache.linsys_static_p = s->kkt_static_reg_P;
    ++cache.builds;
    }
    CUDA_CHECK(cudaGraphLaunch(cache.executable, cudaStreamPerThread));
    ++cache.launches;""",
    )
    graph = once(
        graph,
        "cudaMemcpy(&solver->sol->iters, iterations, sizeof(int)",
        "cudaMemcpy(&solver->sol->iters, cache.iterations, sizeof(int)",
    )
    graph = once(
        graph,
        "    CUDA_CHECK(cudaGraphExecDestroy(executable));\n"
        "    CUDA_CHECK(cudaGraphDestroy(root));\n"
        "    CUDA_CHECK(cudaGraphDestroy(capture.linear));\n"
        "    CUDA_CHECK(cudaFree(iterations));",
        "    qoco_gpu_ipm_resources_leave(cache.resources);\n"
        '    if (getenv("SPACEPDHCG_TEST_QOCO_IPM_CACHE_DISABLE")) qoco_ipm_cache_clear(cache);',
    )
    patches = Path(__file__).resolve().parents[2] / "cpp/cuda/patches"
    sources = {
        "algebra/cuda/qoco_ir_runtime.cuh": runtime,
        "algebra/cuda/cuda_linalg.cu": algebra,
        "algebra/cuda/cudss_backend.cu": backend,
        "algebra/cuda/qoco_batched_stopping.cuh": metrics,
        "algebra/cuda/qoco_ipm_control.cuh": control,
        "algebra/cuda/qoco_ipm_graph.cuh": graph,
    }
    for name in ["qoco_ipm_parameters.cuh", "qoco_ipm_resources.cuh", "qoco_ipm_cache.cuh"]:
        sources["algebra/cuda/" + name] = (patches / name).read_text()
    for name, contents in sources.items():
        (root / name).write_text(contents)
    (root / "spacepdhcg-ipm-cache.json").write_text(
        json.dumps(
            {name: hashlib.sha256(value.encode()).hexdigest() for name, value in sources.items()},
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", required=True, type=Path)
    prepare(parser.parse_args().destination)
