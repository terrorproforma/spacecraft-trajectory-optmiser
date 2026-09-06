#!/usr/bin/env python3
"""Capture terminal recovery and unscaling after v124 initialisation/IPM."""

# ruff: noqa: E501 -- Preserve the validated CUDA replacement literals.
import argparse
import hashlib
import json
from pathlib import Path

from prepare_qoco_device_control import once


def prepare(root):
    control = (root / "algebra/cuda/qoco_device_control.cuh").read_text()
    control = once(
        control,
        "__global__ void restore_decision(State* s,int status) {",
        "__device__ void restore_decision_impl(State* s,int status) {",
    )
    control = once(
        control,
        "template<bool Restore>\n__global__ void best_vectors",
        "__global__ void restore_decision(State* s,int status) { restore_decision_impl(s,status); }\n\ntemplate<bool Restore>\n__global__ void best_vectors",
    )
    algebra = (root / "algebra/cuda/cuda_linalg.cu").read_text()
    algebra += '\n#include "qoco_ipm_terminal.cuh"\n'
    cache = (root / "algebra/cuda/qoco_ipm_cache.cuh").read_text()
    cache = once(
        cache,
        "    bool initialization = false;",
        "    bool initialization = false;\n    bool terminal = false;",
    )
    graph = (root / "algebra/cuda/qoco_ipm_graph.cuh").read_text()
    graph = once(
        graph,
        'extern "C" void qoco_gpu_sync_control(QOCOSolver*);',
        'extern "C" void qoco_gpu_sync_control(QOCOSolver*);\nextern "C" void qoco_gpu_ipm_terminal(QOCOSolver*, const QocoIpmParameters*);',
    )
    graph = once(
        graph,
        "    const bool initialization = qoco_gpu_ipm_initialization_enabled();",
        '    const bool initialization = qoco_gpu_ipm_initialization_enabled();\n    const bool terminal = !getenv("SPACEPDHCG_TEST_QOCO_IPM_TERMINAL_DISABLE");',
    )
    graph = once(
        graph,
        "cache.initialization != initialization || cache.work",
        "cache.terminal != terminal || cache.initialization != initialization || cache.work",
    )
    graph = once(
        graph,
        "    const auto body = params.conditional.phGraph_out[0];",
        "    const auto loop_node = node;\n    const auto body = params.conditional.phGraph_out[0];",
    )
    graph = once(
        graph,
        "    qoco_ipm_pause();\n    qoco_ipm_current = nullptr;",
        """    qoco_ipm_pause();
    if (terminal) {
        capture.graph = root;
        capture.dependencies = {loop_node};
        qoco_ipm_resume();
        qoco_gpu_ipm_terminal(solver, cache.parameters);
        qoco_ipm_pause();
    }
    qoco_ipm_current = nullptr;""",
    )
    graph = once(
        graph,
        "    cache.initialization = initialization;",
        "    cache.initialization = initialization;\n    cache.terminal = terminal;",
    )
    graph = once(graph, "    return 1;\n}", "    return terminal ? 2 : 1;\n}")
    api = (root / "src/qoco_api.c").read_text()
    api = once(
        api,
        "  if (qoco_gpu_ipm_loop(solver)) {",
        "  const int gpu_ipm_result = qoco_gpu_ipm_loop(solver);\n  if (gpu_ipm_result) {",
    )
    api = once(
        api,
        """    if (solver->sol->status == QOCO_NUMERICAL_ERROR || solver->sol->status == QOCO_MAX_ITER)
      restore_best_iterate(solver);
    unscale_variables(work);""",
        """    if (gpu_ipm_result != 2) {
      if (solver->sol->status == QOCO_NUMERICAL_ERROR || solver->sol->status == QOCO_MAX_ITER)
        restore_best_iterate(solver);
      unscale_variables(work);
    }""",
    )
    patch = Path(__file__).resolve().parents[2] / "cpp/cuda/patches"
    sources = {
        "algebra/cuda/qoco_device_control.cuh": control,
        "algebra/cuda/cuda_linalg.cu": algebra,
        "algebra/cuda/qoco_ipm_cache.cuh": cache,
        "algebra/cuda/qoco_ipm_graph.cuh": graph,
        "algebra/cuda/qoco_ipm_terminal.cuh": (patch / "qoco_ipm_terminal.cuh").read_text(),
        "src/qoco_api.c": api,
    }
    for name, contents in sources.items():
        (root / name).write_text(contents)
    (root / "spacepdhcg-ipm-terminal.json").write_text(
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
