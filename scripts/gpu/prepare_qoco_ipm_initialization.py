#!/usr/bin/env python3
"""Capture GPU initialisation before the retained IPM graph (after v123)."""

# ruff: noqa: E501 -- Preserve the validated CUDA replacement literals.
import argparse
import hashlib
import json
from pathlib import Path

from prepare_qoco_device_control import once


def prepare(root):
    cone = (root / "src/cone.cu").read_text()
    cone = once(
        cone, "void bring2cone(QOCOFloat* u,", "void qoco_reference_bring2cone(QOCOFloat* u,"
    )
    start = cone.index("__global__ void cone_residual_stage1(")
    end = cone.index("__global__ void cone_residual_stage2(", start)
    segment = cone[start:end]
    segment = once(
        segment, "QOCOFloat* block_out)", "QOCOFloat* block_out, int* special = nullptr)"
    )
    segment = once(
        segment,
        "  sdata[tid] = val;",
        "  if (special && !isfinite(val)) atomicOr(special, 1);\n  sdata[tid] = val;",
    )
    cone = cone[:start] + segment + cone[end:]
    # Compact SOC identities must clear their tail even when there are no LP rows.
    cone = once(
        cone,
        "  // kernel 1: zero + linear cone\n  if (data->l > 0) {",
        "  // kernel 1: zero + linear cone, including pure-SOC workspaces\n  if (nt_scaling_nnz > 0) {",
    )
    cone += '\n#include "qoco_initial_cones.cuh"\n'
    parameters = (root / "algebra/cuda/qoco_ipm_parameters.cuh").read_text()
    parameters = once(
        parameters,
        "    int maximum, ir_maximum;",
        "    int maximum, ir_maximum;\n    double dynamic_reg;\n    int warm_start;",
    )
    control = (root / "algebra/cuda/qoco_ipm_control.cuh").read_text()
    control = once(
        control,
        "size_t(std::max(d->l, d->nsoc) + 255LL) / 256 + 7",
        "size_t(d->l + static_cast<long long>(d->nsoc) + 255) / 256 + 7",
    )
    algebra = (root / "algebra/cuda/cuda_linalg.cu").read_text()
    algebra += '\n#include "qoco_ipm_initial_control.cuh"\n'
    cache = (root / "algebra/cuda/qoco_ipm_cache.cuh").read_text()
    cache = once(
        cache,
        "    size_t builds{}, launches{};",
        "    bool initialization = false;\n    size_t builds{}, launches{};",
    )
    backend = (root / "algebra/cuda/cudss_backend.cu").read_text()
    start = backend.index("void cudss_set_nt_identity(")
    end = backend.index("static void cudss_update_nt(", start)
    segment = backend[start:end]
    assert segment.count("CUDA_CHECK(cudaDeviceSynchronize());") == 2
    segment = segment.replace(
        "CUDA_CHECK(cudaDeviceSynchronize());",
        "if (!qoco_gpu_ipm_capturing()) CUDA_CHECK(cudaDeviceSynchronize());",
    )
    segment = once(
        segment,
        "    CUDSS_CHECK(g_cuda_funcs.cudssMatrixSetValues(",
        "    if (!qoco_gpu_ipm_capturing()) CUDSS_CHECK(g_cuda_funcs.cudssMatrixSetValues(",
    )
    backend = backend[:start] + segment + backend[end:]
    kkt = (root / "src/kkt.c").read_text()
    kkt = once(
        kkt,
        "void qoco_gpu_note_alpha(",
        "int qoco_gpu_ipm_warmstart(QOCOSolver*);\nvoid qoco_gpu_note_alpha(",
    )
    kkt = once(
        kkt, "  if (work->use_x0) {", "  if (!qoco_gpu_ipm_warmstart(solver) && work->use_x0) {"
    )
    api = (root / "src/qoco_api.c").read_text()
    api = once(
        api,
        "int qoco_gpu_ipm_loop(QOCOSolver*);",
        "int qoco_gpu_ipm_loop(QOCOSolver*);\n"
        "int qoco_gpu_ipm_initialization_enabled(void);\n"
        "void qoco_gpu_ipm_allocate_control(QOCOSolver*);",
    )
    api = once(
        api,
        "  qoco_gpu_reset_control(solver);\n  qoco_gpu_ir_reset_counts(solver, 1);\n  initialize_ipm(solver);",
        "  if (qoco_gpu_ipm_initialization_enabled()) qoco_gpu_ipm_allocate_control(solver);\n"
        "  else {\n"
        "    qoco_gpu_reset_control(solver);\n"
        "    qoco_gpu_ir_reset_counts(solver, 1);\n"
        "    initialize_ipm(solver);\n"
        "  }",
    )
    graph = (root / "algebra/cuda/qoco_ipm_graph.cuh").read_text()
    graph = once(
        graph,
        'extern "C" int qoco_gpu_ipm_loop(QOCOSolver* solver) {',
        """#include "qoco_ipm_warmstart.cuh"
extern "C" void qoco_gpu_ipm_initial_control(QOCOSolver*, const QocoIpmParameters*);
extern "C" int qoco_gpu_ipm_initialization_enabled() {
    const char* enabled = getenv("SPACEPDHCG_TEST_QOCO_IPM_GRAPH");
    return enabled && enabled[0] == '1' && !getenv("SPACEPDHCG_TEST_QOCO_IPM_INIT_DISABLE");
}
extern "C" int qoco_gpu_ipm_loop(QOCOSolver* solver) {""",
    )
    graph = once(
        graph,
        "    auto* s = solver->linsys_data;",
        """    const bool initialization = qoco_gpu_ipm_initialization_enabled();
    if (initialization && getenv("SPACEPDHCG_TEST_QOCO_HOST_INITIAL_CONE")) {
        fprintf(stderr, "GPU initialisation requires device cone shifts\\n"); exit(1);
    }
    auto* s = solver->linsys_data;""",
    )
    graph = once(
        graph,
        "        settings->ir_tol, settings->max_iters, settings->max_ir_iters};",
        "        settings->ir_tol, settings->max_iters, settings->max_ir_iters,\n"
        "        settings->kkt_dynamic_reg, int(work->use_x0)};",
    )
    graph = once(
        graph,
        "    if (cache.executable && (cache.work != work ||",
        "    if (cache.executable && (cache.initialization != initialization || cache.work != work ||",
    )
    graph = once(
        graph,
        "    if (!cache.executable) {\n    qoco_ir_prepare_sparse(data);",
        "    if (!cache.executable) {\n"
        "    // Prime vendor analysis/factor/SOLVE kernels once before graph capture.\n"
        "    if (initialization) initialize_ipm(solver);\n"
        "    qoco_ir_prepare_sparse(data);",
    )
    graph = once(
        graph,
        "    const auto root = capture.graph;\n    int* iterations",
        """    const auto root = capture.graph;
    qoco_ipm_current = &capture;
    if (initialization) {
        qoco_ipm_resume();
        qoco_gpu_ipm_initial_control(solver, cache.parameters);
        qoco_gpu_ir_reset_counts(solver, 1);
        initialize_ipm(solver);
        qoco_ipm_pause();
    }
    int* iterations""",
    )
    graph = once(
        graph,
        "    CUDA_CHECK(cudaGraphAddNode(&node, root, nullptr, 0, &params));",
        "    CUDA_CHECK(cudaGraphAddNode(&node, root, capture.dependencies.data(),\n"
        "        capture.dependencies.size(), &params));",
    )
    graph = once(
        graph,
        "    capture.graph = body;\n    qoco_ipm_current = &capture;",
        "    capture.graph = body;\n    capture.dependencies.clear();\n    qoco_ipm_current = &capture;",
    )
    graph = once(
        graph,
        "    cache.executable = executable;",
        "    cache.initialization = initialization;\n    cache.executable = executable;",
    )
    patch = Path(__file__).resolve().parents[2] / "cpp/cuda/patches"
    sources = {
        "src/cone.cu": cone,
        "src/kkt.c": kkt,
        "src/qoco_api.c": api,
        "algebra/cuda/cudss_backend.cu": backend,
        "algebra/cuda/qoco_ipm_parameters.cuh": parameters,
        "algebra/cuda/qoco_ipm_control.cuh": control,
        "algebra/cuda/qoco_ipm_cache.cuh": cache,
        "algebra/cuda/qoco_ipm_graph.cuh": graph,
        "algebra/cuda/cuda_linalg.cu": algebra,
    }
    for name, directory in [
        ("qoco_initial_cones.cuh", "src"),
        ("qoco_ipm_initial_control.cuh", "algebra/cuda"),
        ("qoco_ipm_warmstart.cuh", "algebra/cuda"),
    ]:
        sources[directory + "/" + name] = (patch / name).read_text()
    for name, contents in sources.items():
        (root / name).write_text(contents)
    (root / "spacepdhcg-ipm-initialization.json").write_text(
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
