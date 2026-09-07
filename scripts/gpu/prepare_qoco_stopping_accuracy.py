#!/usr/bin/env python3
"""Correct Ruiz stopping metrics in a prepared GPU QOCO tree.

Apply after the IPM graph extensions. For x_orig=D*x, y_orig=E*y/k,
z_orig=F*z/k and s_orig=inv(F)*s, all termination quantities must be in
original problem units. Keep both complementarity and the actual objective
gap: a small s'z does not imply a small objective gap away from stationarity.
"""

import argparse
import hashlib
import json
from pathlib import Path

from prepare_qoco_device_control import once


def prepare(root: Path) -> dict[str, str]:
    path = root / "algebra/cuda/qoco_batched_stopping.cuh"
    text = path.read_text()
    text = once(text, "product_norm(f, s, u1, d->m, S);", "product_norm(fi, s, u1, d->m, S);")
    text = once(
        text,
        "    product_norm(xb, di, xb, d->n, PX);\n    dot(x, xb, d->n, XPX);",
        "    dot(x, xb, d->n, XPX);\n    product_norm(xb, di, xb, d->n, PX);",
    )
    text = once(
        text,
        "    if (d->m) { ew_product(s, f, u1, d->m); ew_product(z, f, u2, d->m); }\n"
        "    dot(u1, u2, d->m, GAP);",
        "    dot(s, z, d->m, GAP);",
    )
    start = text.index("    const double half = __dmul_rn(0.5, v[XPX]);")
    end = text.index("    if constexpr (Iteration)", start)
    text = (
        text[:start]
        + """    const double half = __dmul_rn(0.5, v[XPX]);
    const double primal = __dmul_rn(__dadd_rn(half, v[CTX]), kinv);
    const double dual = __dmul_rn(__dadd_rn(__dadd_rn(-half, -v[BTY]), -v[HTZ]), kinv);
    const double gap_rel = qoco_max(qoco_max(1.0, fabs(primal)), fabs(dual));
    const double complementarity = fabs(__dmul_rn(v[GAP], kinv));
    const double objective_gap = fabs(__dadd_rn(primal, -dual));
    out[0] = pres; out[1] = v[DUAL];
    out[2] = qoco_max(complementarity, objective_gap);
    out[3] = pres_rel; out[4] = __dmul_rn(dres_rel, kinv); out[5] = gap_rel;
"""
        + text[end:]
    )
    # Keep the explicit host-dispatched arithmetic audit consistent too.
    reference_path = root / "src/qoco_utils.c"
    reference = reference_path.read_text()
    reference = once(
        reference,
        "  ew_product(Fruiz_data, sdata, ubuff1, data->m);",
        "  ew_product(get_data_vectorf(work->scaling->Finvruiz), sdata, ubuff1, data->m);",
    )
    reference = once(
        reference,
        "  ew_product(xbuff, Dinvruiz_data, xbuff, data->n);\n"
        "  QOCOFloat Pxinf = inf_norm(xbuff, data->n);\n"
        "  QOCOFloat xPx = qoco_dot(xdata, xbuff, work->data->n);",
        "  QOCOFloat xPx = qoco_dot(xdata, xbuff, work->data->n);\n"
        "  ew_product(xbuff, Dinvruiz_data, xbuff, data->n);\n"
        "  QOCOFloat Pxinf = inf_norm(xbuff, data->n);",
    )
    reference = once(
        reference,
        "  ew_product(sdata, Fruiz_data, ubuff1, data->m);\n"
        "  ew_product(zdata, Fruiz_data, ubuff2, data->m);\n"
        "  QOCOFloat gap = qoco_dot(ubuff1, ubuff2, data->m);\n"
        "  gap *= work->scaling->kinv;",
        "  QOCOFloat gap = qoco_abs(qoco_dot(sdata, zdata, data->m) * work->scaling->kinv);",
    )
    reference = once(
        reference,
        "  QOCOFloat pobj = 0.5 * xPx + ctx;\n  QOCOFloat dobj = -0.5 * xPx - bty - htz;",
        "  QOCOFloat pobj = (0.5 * xPx + ctx) * work->scaling->kinv;\n"
        "  QOCOFloat dobj = (-0.5 * xPx - bty - htz) * work->scaling->kinv;\n"
        "  gap = qoco_max(gap, qoco_abs(pobj - dobj));",
    )
    # CPU priming needs the same zero-norm identity rule as GPU updates.
    equilibration_path = root / "src/equilibration.c"
    equilibration = equilibration_path.read_text()
    equilibration = once(
        equilibration, "    g = safe_div(1.0, g);", "    g = g > 1e-15 ? 1.0 / g : 1.0;"
    )
    if equilibration.count("temp = safe_div(1.0, temp);") != 3:
        raise ValueError("unexpected Ruiz reciprocal layout")
    equilibration = equilibration.replace(
        "temp = safe_div(1.0, temp);", "temp = temp > 1e-15 ? 1.0 / temp : 1.0;"
    )
    # An inaccurate terminal iterate can be worse than the saved best iterate,
    # just as an iteration-limit or numerical-error exit can be.
    terminal_path = root / "algebra/cuda/qoco_ipm_terminal.cuh"
    terminal = once(
        terminal_path.read_text(),
        "state->status == QOCO_NUMERICAL_ERROR || state->status == QOCO_MAX_ITER)",
        "state->status == QOCO_NUMERICAL_ERROR || state->status == QOCO_MAX_ITER ||\n"
        "        state->status == QOCO_SOLVED_INACCURATE)",
    )
    files = {
        path: text,
        reference_path: reference,
        equilibration_path: equilibration,
        terminal_path: terminal,
    }
    report = {}
    for destination, contents in files.items():
        destination.write_text(contents)
        report[str(destination.relative_to(root))] = hashlib.sha256(
            destination.read_bytes()
        ).hexdigest()
    (root / "spacepdhcg-stopping-accuracy.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", required=True, type=Path)
    print(json.dumps(prepare(parser.parse_args().destination), indent=2))
