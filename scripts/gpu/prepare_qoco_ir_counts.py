#!/usr/bin/env python3
"""Keep refinement accounting on device; copy totals only for final/verbose output."""

import argparse
import hashlib
import json
from pathlib import Path

from prepare_qoco_device_control import once


def prepare(root):
    cache = (root / "algebra/cuda/qoco_ir_cache.cuh").read_text()
    cache = once(
        cache,
        "struct QocoIrParameters {",
        "struct QocoIrCounts { QOCOInt step, total; };\nstruct QocoIrParameters {",
    )
    runtime = (root / "algebra/cuda/qoco_ir_runtime.cuh").read_text()
    runtime = once(
        runtime, "    QocoIrCache cache;", "    QocoIrCache cache;\n    QocoIrCounts* counts{};"
    )
    runtime = once(
        runtime,
        "    auto* runtime = new QocoIrRuntime;",
        "    auto* runtime = new QocoIrRuntime;\n"
        "    CUDA_CHECK(cudaMalloc(&runtime->counts, sizeof(QocoIrCounts)));\n"
        "    CUDA_CHECK(cudaMemset(runtime->counts, 0, sizeof(QocoIrCounts)));",
    )
    runtime = once(
        runtime,
        "    qoco_ir_cache_clear(runtime->cache);",
        "    qoco_ir_cache_clear(runtime->cache);\n    CUDA_CHECK(cudaFree(runtime->counts));",
    )
    device = (root / "algebra/cuda/qoco_device_ir.cuh").read_text()
    device = once(
        device,
        "static void qoco_ir_solve(",
        '#include "qoco_ir_device_counts.cuh"\n\nstatic void qoco_ir_solve(',
    )
    device = once(
        device,
        "    QocoIrState host;\n"
        "    CUDA_CHECK(cudaMemcpyAsync(&host, runtime->state, sizeof(host), "
        "cudaMemcpyDeviceToHost, stream));\n"
        "    CUDA_CHECK(cudaStreamSynchronize(stream));\n"
        "    work->ir_iters += host.accepted;\n"
        '    if (getenv("SPACEPDHCG_TEST_QOCO_IR_CACHE_DISABLE")) qoco_ir_cache_clear(cache);',
        """    if (qoco_ir_counts_enabled()) {
        qoco_ir_add_count_kernel<<<1, 1, 0, stream>>>(runtime->counts, runtime->state);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(runtime->done, stream));
        CUDA_CHECK(cudaStreamWaitEvent(nullptr, runtime->done));
    } else {
        QocoIrState host;
        CUDA_CHECK(cudaMemcpyAsync(&host, runtime->state, sizeof(host), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        work->ir_iters += host.accepted;
    }
    if (getenv("SPACEPDHCG_TEST_QOCO_IR_CACHE_DISABLE")) {
        CUDA_CHECK(cudaStreamSynchronize(stream));
        qoco_ir_cache_clear(cache);
    }""",  # noqa: E501 - preserve the emitted CUDA source exactly.
    )
    api = (root / "src/qoco_api.c").read_text()
    api = once(
        api,
        "void qoco_gpu_reset_control(QOCOSolver*);",
        "void qoco_gpu_reset_control(QOCOSolver*);\n"
        "void qoco_gpu_ir_reset_counts(QOCOSolver*, int);\n"
        "void qoco_gpu_ir_report_counts(QOCOSolver*);\n"
        "void qoco_gpu_ir_finish_step(QOCOSolver*);",
    )
    api = once(
        api,
        "  qoco_gpu_reset_control(solver);",
        "  qoco_gpu_reset_control(solver);\n  qoco_gpu_ir_reset_counts(solver, 1);",
    )
    api = once(
        api,
        "    // Reset IR iteration counter for this IPM step.\n",
        "    // Reset IR iteration counter for this IPM step.\n"
        "    qoco_gpu_ir_reset_counts(solver, 0);\n",
    )
    api = once(
        api, "    solver->sol->ir_iters += work->ir_iters;", "    qoco_gpu_ir_finish_step(solver);"
    )
    if api.count("copy_solution(solver);") != 2:
        raise ValueError("Unexpected terminal solution paths")
    api = api.replace(
        "copy_solution(solver);", "copy_solution(solver);\n  qoco_gpu_ir_report_counts(solver);"
    )
    header = Path(__file__).resolve().parents[2] / "cpp/cuda/patches/qoco_ir_device_counts.cuh"
    sources = {
        "algebra/cuda/qoco_ir_cache.cuh": cache,
        "algebra/cuda/qoco_ir_runtime.cuh": runtime,
        "algebra/cuda/qoco_device_ir.cuh": device,
        "algebra/cuda/qoco_ir_device_counts.cuh": header.read_text(),
        "src/qoco_api.c": api,
    }
    for name, contents in sources.items():
        (root / name).write_text(contents)
    (root / "spacepdhcg-ir-counts.json").write_text(
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
