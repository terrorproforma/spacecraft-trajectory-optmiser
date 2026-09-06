#!/usr/bin/env python3
"""Install a diagnostic factor/solve graph probe in an isolated v110 QOCO tree."""

import argparse
import hashlib
import json
from pathlib import Path

from prepare_qoco_device_control import once


def prepare(root):
    path = root / "algebra/cuda/cudss_backend.cu"
    text = path.read_text()
    text = once(
        text,
        "struct LinSysData {",
        '#include <unordered_map>\n#include "qoco_factor_graph_memory.cuh"\n'
        "struct LinSysData {\n  QocoFactorMemory* factor_memory;",
    )
    anchor = "  linsys_data->ir = qoco_ir_create(linsys_data->handle, linsys_data->Kn);"
    text = once(
        text,
        anchor,
        anchor + "\n  linsys_data->factor_memory = qoco_factor_memory_create(linsys_data->handle);",
    )
    text = once(
        text,
        '#include "qoco_device_ir.cuh"',
        '#include "qoco_device_ir.cuh"\n#include "qoco_factor_graph_probe.cuh"',
    )
    text = once(
        text,
        "  cudss_solve_system(linsys_data, b, x);",
        "  cudss_solve_system(linsys_data, b, x);\n"
        "  qoco_factor_graph_probe::run(linsys_data, work, b, x);",
    )
    text = once(
        text,
        "static void cudss_cleanup(LinSysData* linsys_data)\n{",
        "static void cudss_cleanup(LinSysData* linsys_data)\n{\n"
        "  qoco_factor_graph_probe::cleanup(linsys_data);",
    )
    text = once(
        text,
        "    qoco_ir_destroy(linsys_data->ir);",
        "    qoco_ir_destroy(linsys_data->ir);\n"
        "    qoco_factor_memory_destroy(linsys_data->factor_memory);",
    )
    sources = {"algebra/cuda/cudss_backend.cu": text}
    tests = Path(__file__).resolve().parents[2] / "cpp/cuda/tests"
    for name in ("qoco_factor_graph_memory.cuh", "qoco_factor_graph_probe.cuh"):
        sources["algebra/cuda/" + name] = (tests / name).read_text()
    for name, contents in sources.items():
        (root / name).write_text(contents)
    (root / "spacepdhcg-factor-graph-probe.json").write_text(
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
