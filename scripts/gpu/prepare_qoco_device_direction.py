#!/usr/bin/env python3
"""Move the combined-direction NaN recovery decision into the GPU iterate update."""

import argparse
import hashlib
import json
from pathlib import Path

from prepare_qoco_device_control import once


def prepare(root):
    names = (
        "src/kkt.c",
        "algebra/cuda/cuda_linalg.cu",
        "algebra/cuda/qoco_device_control.cuh",
        "algebra/cuda/qoco_device_step_control.cuh",
    )
    texts = {name: (root / name).read_text() for name in names}
    s = texts["src/kkt.c"]
    s = once(
        s,
        "void qoco_gpu_note_alpha(QOCOWorkspace*, const double*);",
        "void qoco_gpu_note_alpha(QOCOWorkspace*, const double*);\n"
        "void qoco_gpu_scan_direction(QOCOWorkspace*);",
    )
    s = once(
        s,
        "  if (check_nan(work->xyz)) {",
        "  qoco_gpu_scan_direction(work);\n"
        "  if (!work->gpu_control && check_nan(work->xyz)) {",
    )
    texts["src/kkt.c"] = s
    s = texts["algebra/cuda/qoco_device_control.cuh"]
    s = once(
        s,
        "    int best_iter, best_valid, save, restored;",
        "    int best_iter, best_valid, save, restored;\n    int invalid_direction;",
    )
    texts["algebra/cuda/qoco_device_control.cuh"] = s
    s = texts["algebra/cuda/qoco_device_step_control.cuh"]
    s = 'extern "C" const int* qoco_gpu_direction_flag(QOCOWorkspace*);\n' + s
    s = once(
        s,
        "int n, int p, int m, const double* steps, double* alpha) {",
        "int n, int p, int m, const double* steps, double* alpha, const int* invalid) {\n"
        "    // Do not multiply a rejected NaN direction by zero: that still yields NaN.\n"
        "    if (invalid && *invalid) {\n"
        "        if (blockIdx.x == 0 && threadIdx.x == 0) *alpha = 0.0;\n"
        "        return;\n    }",
    )
    s = once(
        s,
        "    if (audit) reference = qoco_min(linesearch",
        "    if (audit && w->gpu_control && check_nan(w->xyz)) reference = 0.0;\n"
        "    else if (audit) reference = qoco_min(linesearch",
    )
    s = once(
        s,
        "        d->n, d->p, d->m, scalar, scalar + 2);",
        "        d->n, d->p, d->m, scalar, scalar + 2, qoco_gpu_direction_flag(w));",
    )
    texts["algebra/cuda/qoco_device_step_control.cuh"] = s
    texts["algebra/cuda/cuda_linalg.cu"] += '\n#include "qoco_device_direction.cuh"\n'
    extension = Path(__file__).resolve().parents[2] / "cpp/cuda/patches/qoco_device_direction.cuh"
    texts["algebra/cuda/qoco_device_direction.cuh"] = extension.read_text()
    # All anchors must match before mutating the isolated preparation tree.
    for name, text in texts.items():
        (root / name).write_text(text)
    (root / "spacepdhcg-device-direction.json").write_text(
        json.dumps(
            {name: hashlib.sha256(text.encode()).hexdigest() for name, text in texts.items()},
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", required=True, type=Path)
    prepare(parser.parse_args().destination)
