#!/usr/bin/env python3
"""Experimental whole-IPM conditional graph after v117; compile with PTDS enabled."""

import argparse
import hashlib
import json
from pathlib import Path

from prepare_qoco_device_control import once


def prepare(root):
    backend = (root / "algebra/cuda/cudss_backend.cu").read_text()
    backend = once(
        backend,
        "  g_cuda_funcs.cublasSetStream =",
        "  g_cuda_funcs.cublasSetWorkspace = reinterpret_cast<decltype(&::cublasSetWorkspace)>(\n"
        '      dlsym(g_cublas_handle, "cublasSetWorkspace_v2"));\n'
        "  if (!g_cuda_funcs.cublasSetWorkspace) return 0;\n"
        "  g_cuda_funcs.cublasSetStream =",
    )
    table = (root / "algebra/cuda/cudss_backend.h").read_text()
    table = once(
        table,
        "  decltype(&::cublasSetStream) cublasSetStream;",
        "  decltype(&::cublasSetStream) cublasSetStream;\n"
        "  decltype(&::cublasSetWorkspace) cublasSetWorkspace;",
    )
    scope = (root / "algebra/cuda/qoco_reduction_scope.cuh").read_text()
    scope = once(
        scope,
        "static thread_local cublasHandle_t qoco_scoped_reduction_handle = nullptr;",
        "static thread_local void* qoco_ipm_blas_workspace = nullptr;\n"
        "static thread_local cublasHandle_t qoco_scoped_reduction_handle = nullptr;",
    )
    scope = once(
        scope,
        "    get_cuda_funcs()->cublasDestroy(qoco_scoped_reduction_handle);",
        "    get_cuda_funcs()->cublasDestroy(qoco_scoped_reduction_handle);\n"
        "    if (qoco_ipm_blas_workspace) {\n"
        "        const auto released_blas = cudaFree(qoco_ipm_blas_workspace);\n"
        "        qoco_ipm_blas_workspace = nullptr;\n"
        "        if (released_blas != cudaSuccess) {\n"
        '            fprintf(stderr, "QOCO cuBLAS workspace release failed: %s\\n", '
        "cudaGetErrorString(released_blas));\n"
        "            exit(1);\n"
        "        }\n"
        "    }",
    )
    backend = once(
        backend,
        "static void cudss_factor(",
        'extern "C" int qoco_gpu_ipm_capturing();\n'
        "static void qoco_ipm_factor(LinSysData*);\n"
        "static void qoco_ipm_linear(LinSysData*, QOCOWorkspace*, "
        "const double*, double*, double, int);\n"
        "static void cudss_factor(",
    )
    backend = once(
        backend,
        "  (void)n;\n  (void)kkt_dynamic_reg;",
        "  (void)n;\n  (void)kkt_dynamic_reg;\n"
        "  if (qoco_gpu_ipm_capturing()) { qoco_ipm_factor(linsys_data); return; }",
    )
    backend = once(
        backend,
        "  // Initial solve. Store the current solution in x;",
        "  if (qoco_gpu_ipm_capturing()) {\n"
        "    qoco_ipm_linear(linsys_data, work, b, x, ir_tol, max_ir_iters); return;\n"
        "  }\n  // Initial solve. Store the current solution in x;",
    )
    start = backend.index("static void cudss_update_nt(")
    end = backend.index("static void cudss_update_data(", start)
    segment = backend[start:end]
    segment = once(
        segment,
        "CUDA_CHECK(cudaDeviceSynchronize());",
        "if (!qoco_gpu_ipm_capturing()) CUDA_CHECK(cudaDeviceSynchronize());",
    )
    segment = once(segment, "cudaMemcpy(linsys_data->d_WtW", "cudaMemcpyAsync(linsys_data->d_WtW")
    backend = backend[:start] + segment + backend[end:]
    backend = once(
        backend,
        "static void cudss_cleanup(",
        '#include "qoco_ipm_graph.cuh"\n\nstatic void cudss_cleanup(',
    )
    control = (root / "algebra/cuda/qoco_device_control.cuh").read_text()
    control = once(
        control, "__global__ void decide(State* s,", "__device__ void decide_impl(State* s,"
    )
    control = once(
        control,
        "__global__ void restore_decision(",
        "__global__ void decide(State* s,double absolute,double relative,"
        "double inaccurate_absolute,\n"
        "    double inaccurate_relative,int iteration) {\n"
        "    decide_impl(s,absolute,relative,inaccurate_absolute,inaccurate_relative,iteration);\n"
        "}\n__global__ void restore_decision(",
    )
    metric = (root / "algebra/cuda/qoco_metric_graph.cuh").read_text()
    metric = once(
        metric,
        "namespace qoco_metric_graph {",
        'extern "C" int qoco_gpu_ipm_capturing();\nnamespace qoco_metric_graph {',
    )
    metric = once(
        metric,
        '    const char* disable = std::getenv("SPACEPDHCG_TEST_QOCO_METRIC_GRAPH_DISABLE");',
        '    const char* ipm = std::getenv("SPACEPDHCG_TEST_QOCO_IPM_GRAPH");\n'
        "    if (qoco_gpu_ipm_capturing() || (ipm && ipm[0] == '1')) { enqueue(); return; }\n"
        '    const char* disable = std::getenv("SPACEPDHCG_TEST_QOCO_METRIC_GRAPH_DISABLE");',
    )
    algebra = (root / "algebra/cuda/cuda_linalg.cu").read_text()
    algebra += '\n#include "qoco_ipm_control.cuh"\n'
    api = (root / "src/qoco_api.c").read_text()
    api = once(
        api,
        "void qoco_gpu_ir_finish_step(QOCOSolver*);",
        "void qoco_gpu_ir_finish_step(QOCOSolver*);\nint qoco_gpu_ipm_loop(QOCOSolver*);",
    )
    api = once(
        api,
        "  initialize_ipm(solver);",
        """  initialize_ipm(solver);
  if (qoco_gpu_ipm_loop(solver)) {
    stop_timer(&(work->solve_timer));
    if (solver->sol->status == QOCO_NUMERICAL_ERROR || solver->sol->status == QOCO_MAX_ITER)
      restore_best_iterate(solver);
    unscale_variables(work);
    copy_solution(solver);
    qoco_gpu_ir_report_counts(solver);
    return solver->sol->status;
  }""",
    )
    patch = Path(__file__).resolve().parents[2] / "cpp/cuda/patches"
    sources = {
        "algebra/cuda/cudss_backend.cu": backend,
        "algebra/cuda/cudss_backend.h": table,
        "algebra/cuda/qoco_reduction_scope.cuh": scope,
        "algebra/cuda/qoco_device_control.cuh": control,
        "algebra/cuda/qoco_metric_graph.cuh": metric,
        "algebra/cuda/cuda_linalg.cu": algebra,
        "src/qoco_api.c": api,
        "algebra/cuda/qoco_ipm_graph.cuh": (patch / "qoco_ipm_graph.cuh").read_text(),
        "algebra/cuda/qoco_ipm_control.cuh": (patch / "qoco_ipm_control.cuh").read_text(),
    }
    for name, contents in sources.items():
        (root / name).write_text(contents)
    (root / "spacepdhcg-ipm-graph.json").write_text(
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
