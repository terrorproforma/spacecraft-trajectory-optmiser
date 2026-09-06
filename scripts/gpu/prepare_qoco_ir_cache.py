#!/usr/bin/env python3
"""Retain conditional refinement graphs after the retained factorization patch.

Fixed sparsity and stable operator buffers are QOCO workspace invariants. Numeric
coefficients and RHS values remain device inputs; tolerance and iteration limit
are set on the solver stream before every replay rather than frozen in the graph.
"""

import argparse
import hashlib
import json
from pathlib import Path

from prepare_qoco_device_control import once


def prepare(root):
    runtime = (root / "algebra/cuda/qoco_ir_runtime.cuh").read_text()
    if "QocoFactorMemory* factor_memory" not in runtime:
        raise ValueError("Retained cuDSS allocations are required before IR graph reuse")
    runtime = once(
        runtime, "struct QocoIrState {", '#include "qoco_ir_cache.cuh"\n\nstruct QocoIrState {'
    )
    runtime = once(
        runtime, "struct QocoIrRuntime {", "struct QocoIrRuntime {\n    QocoIrCache cache;"
    )
    runtime = once(
        runtime,
        "    auto* runtime = new QocoIrRuntime;",
        "    auto* runtime = new QocoIrRuntime;\n"
        "    CUDA_CHECK(cudaMalloc(&runtime->cache.parameters, sizeof(QocoIrParameters)));",
    )
    runtime = once(
        runtime,
        "    for (auto input : runtime->range_inputs)",
        '    if (getenv("SPACEPDHCG_TEST_QOCO_IR_CACHE_TRACE"))\n'
        '        fprintf(stderr, "IR_CACHE builds=%zu launches=%zu\\n", '
        "runtime->cache.builds, runtime->cache.launches);\n"
        "    qoco_ir_cache_clear(runtime->cache);\n"
        "    CUDA_CHECK(cudaFree(runtime->cache.parameters));\n"
        "    for (auto input : runtime->range_inputs)",
    )
    anchor = "cudaGraphConditionalHandle handle, double tolerance, int maximum) {"
    if runtime.count(anchor) != 2:
        raise ValueError("Unexpected refinement decision signatures")
    runtime = runtime.replace(
        anchor,
        "cudaGraphConditionalHandle handle, const QocoIrParameters* parameters) {\n"
        "    const double tolerance = parameters->tolerance;\n"
        "    const int maximum = parameters->maximum;",
    )
    device = (root / "algebra/cuda/qoco_device_ir.cuh").read_text()
    device = once(
        device,
        "// Graphs are deliberately scoped to a single solve until refactorization reuse\n"
        "// can be proved. Runtime scratch and its stream live with the cuDSS handle.",
        "// Graphs and scratch live with the fixed-topology cuDSS workspace.\n"
        "// Every replay consumes current coefficient/RHS buffers and device parameters.",
    )
    device = once(
        device,
        "    cudaGraph_t graph, ended;",
        """    auto& cache = runtime->cache;
    if (cache.executable && (cache.work != work || cache.rhs != b ||
                             cache.solution != x || cache.count != count)) {
        CUDA_CHECK(cudaStreamSynchronize(stream));
        qoco_ir_cache_clear(cache);
    }
    qoco_ir_set_parameters<<<1, 1, 0, stream>>>(cache.parameters, tolerance, maximum);
    CUDA_CHECK(cudaGetLastError());
    if (!cache.executable) {
    cudaGraph_t graph, ended;""",
    )
    if device.count("runtime->state, norm, handle, tolerance, maximum") != 2:
        raise ValueError("Unexpected refinement decision launches")
    device = device.replace(
        "runtime->state, norm, handle, tolerance, maximum",
        "runtime->state, norm, handle, cache.parameters",
    )
    device = once(device, "    qoco_ir_exchange_stream(previous_stream);\n", "")
    device = once(
        device,
        "    CUDA_CHECK(cudaGraphLaunch(executable, stream));",
        """    cache.graph = graph;
    cache.executable = executable;
    cache.work = work; cache.rhs = b; cache.solution = x; cache.count = count;
    ++cache.builds;
    }
    qoco_ir_exchange_stream(previous_stream);
    CUDA_CHECK(cudaGraphLaunch(cache.executable, stream));
    ++cache.launches;""",
    )
    device = once(
        device,
        "    CUDA_CHECK(cudaGraphExecDestroy(executable));\n"
        "    CUDA_CHECK(cudaGraphDestroy(graph));",
        '    if (getenv("SPACEPDHCG_TEST_QOCO_IR_CACHE_DISABLE")) qoco_ir_cache_clear(cache);',
    )
    header = Path(__file__).resolve().parents[2] / "cpp/cuda/patches/qoco_ir_cache.cuh"
    sources = {
        "algebra/cuda/qoco_ir_runtime.cuh": runtime,
        "algebra/cuda/qoco_device_ir.cuh": device,
        "algebra/cuda/qoco_ir_cache.cuh": header.read_text(),
    }
    for name, contents in sources.items():
        (root / name).write_text(contents)
    (root / "spacepdhcg-ir-cache.json").write_text(
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
