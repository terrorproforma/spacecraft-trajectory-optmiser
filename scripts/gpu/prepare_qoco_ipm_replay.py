#!/usr/bin/env python3
"""Add device completion packets and nonblocking prepared replay after v125."""

# ruff: noqa: E501 -- Preserve CUDA replacement literals.
import argparse
import hashlib
import json
from pathlib import Path

from prepare_qoco_device_control import once


def prepare(root):
    cache = (root / "algebra/cuda/qoco_ipm_cache.cuh").read_text()
    cache = once(
        cache,
        '#include "qoco_ipm_parameters.cuh"',
        '#include "qoco_ipm_parameters.cuh"\n#include "qoco_gpu_replay.h"',
    )
    cache = once(
        cache,
        "    int* iterations{};",
        "    int* iterations{};\n    QocoGpuCompletion* completion{};\n    cudaStream_t replay_stream{};\n    bool replay_pending = false;",
    )
    cache = once(
        cache,
        "    CUDA_CHECK(cudaFree(cache.iterations));",
        "    CUDA_CHECK(cudaFree(cache.iterations));\n    CUDA_CHECK(cudaFree(cache.completion));",
    )
    algebra = (root / "algebra/cuda/cuda_linalg.cu").read_text()
    algebra += """
#include "qoco_ipm_completion.cuh"
extern "C" int qoco_gpu_ipm_resources_compatible(void* pointer) {
    auto* r = static_cast<QocoIpmResources*>(pointer);
    int device;
    return r && !r->active && cudaGetDevice(&device) == cudaSuccess &&
        r->device == device && r->owner == std::this_thread::get_id();
}
"""
    graph = (root / "algebra/cuda/qoco_ipm_graph.cuh").read_text()
    graph = once(
        graph,
        'extern "C" void qoco_gpu_ipm_terminal(QOCOSolver*, const QocoIpmParameters*);',
        'extern "C" void qoco_gpu_ipm_terminal(QOCOSolver*, const QocoIpmParameters*);\nextern "C" void qoco_gpu_ipm_completion(QOCOSolver*, const int*, const int*, const int*, QocoGpuCompletion*);',
    )
    graph = once(
        graph,
        "        CUDA_CHECK(cudaMalloc(&cache.iterations, sizeof(int)));",
        "        CUDA_CHECK(cudaMalloc(&cache.iterations, sizeof(int)));\n        CUDA_CHECK(cudaMalloc(&cache.completion, sizeof(QocoGpuCompletion)));",
    )
    graph = once(
        graph,
        "        qoco_gpu_ipm_terminal(solver, cache.parameters);",
        "        qoco_gpu_ipm_terminal(solver, cache.parameters);\n        qoco_gpu_ipm_completion(solver, cache.iterations, &s->ir->counts->step,\n            &s->ir->counts->total, cache.completion);",
    )
    graph += '\n#include "qoco_ipm_replay.cuh"\n'
    api = (root / "src/qoco_api.c").read_text()
    api = once(
        api,
        "QOCOInt qoco_solve(QOCOSolver* solver)\n{",
        'int qoco_gpu_ipm_finish_device(QOCOSolver*);\nQOCOInt qoco_solve(QOCOSolver* solver)\n{\n  if (qoco_gpu_ipm_finish_device(solver)) {\n    fprintf(stderr, "Pending GPU replay could not complete\\n"); exit(1);\n  }',
    )
    patch = Path(__file__).resolve().parents[2] / "cpp/cuda/patches"
    sources = {
        "algebra/cuda/qoco_ipm_cache.cuh": cache,
        "algebra/cuda/cuda_linalg.cu": algebra,
        "algebra/cuda/qoco_ipm_graph.cuh": graph,
        "src/qoco_api.c": api,
    }
    for name in ["qoco_gpu_replay.h", "qoco_ipm_completion.cuh", "qoco_ipm_replay.cuh"]:
        sources["algebra/cuda/" + name] = (patch / name).read_text()
    for name, contents in sources.items():
        (root / name).write_text(contents)
    (root / "spacepdhcg-ipm-replay.json").write_text(
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
