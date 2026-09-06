#!/usr/bin/env python3
"""Capture numerical factorization once, reusing retained vendor allocations."""

import argparse
import hashlib
import json
from pathlib import Path

from prepare_qoco_device_control import once


def prepare(root):
    path = root / "algebra/cuda/qoco_ir_runtime.cuh"
    text = path.read_text()
    text = once(
        text, "struct QocoIrState {", '#include "qoco_factor_runtime.cuh"\n\nstruct QocoIrState {'
    )
    text = once(
        text,
        "struct QocoIrRuntime {",
        "struct QocoIrRuntime {\n    QocoFactorMemory* factor_memory{};",
    )
    text = once(
        text,
        "    CUDSS_CHECK(set_stream(handle, runtime->stream));",
        "    CUDSS_CHECK(set_stream(handle, runtime->stream));\n"
        "    runtime->factor_memory = qoco_factor_memory_create(handle);",
    )
    text = once(
        text,
        "    delete runtime;",
        "    qoco_factor_memory_destroy(runtime->factor_memory);\n    delete runtime;",
    )
    text = once(
        text,
        "const auto result = g_cuda_funcs.cudssExecute(handle, phase, config, data, "
        "matrix, solution, rhs);",
        "const auto result = qoco_factor_execute(runtime->factor_memory, runtime->stream, "
        "handle, phase, config, data, matrix, solution, rhs);",
    )
    extension = Path(__file__).resolve().parents[2] / "cpp/cuda/patches/qoco_factor_runtime.cuh"
    sources = {"algebra/cuda/qoco_ir_runtime.cuh": text,
               "algebra/cuda/qoco_factor_runtime.cuh": extension.read_text()}
    for name, contents in sources.items():
        (root / name).write_text(contents)
    (root / "spacepdhcg-factor-graph.json").write_text(
        json.dumps({name: hashlib.sha256(value.encode()).hexdigest()
                    for name, value in sources.items()}, indent=2) + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", required=True, type=Path)
    prepare(parser.parse_args().destination)
