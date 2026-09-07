#!/usr/bin/env python3
"""Add GPU objective-gap certification for translated primal coordinates.

Apply to a tree prepared with prepare_qoco_stopping_accuracy.py. For x=q+d,
gamma=.5*q'P*q+c'q and stationarity residual r, the original objectives are
primal=primal_shifted+gamma and dual=dual_shifted+gamma-q'r. This correction
matters away from exact stationarity. The ordinary objective output (slot 6)
retains solver-coordinate semantics; stopping gap and its scale use the original
objectives. All origin data is borrowed device storage, updated in stream order.
Bind before the first metric evaluation; rebinding different addresses thereafter
is rejected because complete IPM graphs may already borrow these addresses.
"""

import argparse
import hashlib
import json
from pathlib import Path

from prepare_qoco_device_control import once


def prepare(root: Path) -> dict[str, str]:
    files = {}
    path = root / "include/structs.h"
    files[path] = once(
        path.read_text(),
        "  void* gpu_control;",
        """  void* gpu_control;
  const double* stopping_origin;
  const double* stopping_offset;
  int stopping_origin_locked;""",
    )
    path = root / "src/qoco_api.c"
    files[path] = once(
        path.read_text(),
        "  work->gpu_control = NULL;",
        """  work->gpu_control = NULL;
  work->stopping_origin = NULL;
  work->stopping_offset = NULL;
  work->stopping_origin_locked = 0;""",
    )
    path = root / "algebra/cuda/qoco_batched_stopping.cuh"
    text = once(path.read_text(), "BTY, HTZ, Count,", "BTY, HTZ, ORIGIN_DUAL, Count,")
    text = once(
        text,
        "int m, double* out) {\n    const double pres",
        "int m, double* out, const double* offset) {\n    const double pres",
    )
    text = once(
        text,
        "    const double gap_rel = qoco_max(qoco_max(1.0, fabs(primal)), fabs(dual));",
        """    const double gamma = offset ? *offset : 0.0;
    const double original_primal = __dadd_rn(primal, gamma);
    const double original_dual = __dadd_rn(__dadd_rn(dual, gamma), -v[ORIGIN_DUAL]);
    const double gap_rel = qoco_max(qoco_max(1.0, fabs(original_primal)), fabs(original_dual));""",
    )
    text = once(
        text,
        "const double objective_gap = fabs(__dadd_rn(primal, -dual));",
        "const double objective_gap = fabs(__dadd_rn(__dadd_rn(primal, -dual), v[ORIGIN_DUAL]));",
    )
    text = once(
        text,
        "int m, double* out) {\n    finish_values<Iteration>(v, kinv, k, reg, m, out);",
        "int m, double* out, const double* offset) {\n"
        "    finish_values<Iteration>(v, kinv, k, reg, m, out, offset);",
    )
    text = once(
        text,
        "int m, double* out) {\n    finish_values<Iteration>(v, p->kinv, p->k, reg, m, out);",
        "int m, double* out, const double* offset) {\n"
        "    finish_values<Iteration>(v, p->kinv, p->k, reg, m, out, offset);",
    )
    text = once(
        text,
        "    auto* w = solver->work; auto* d = w->data; auto* scale = w->scaling;",
        "    auto* w = solver->work; auto* d = w->data; auto* scale = w->scaling;\n"
        "    w->stopping_origin_locked = 1;",
    )
    text = once(
        text,
        "    norm(xb, d->n, scratch + DUAL, partial);",
        """    norm(xb, d->n, scratch + DUAL, partial);
    // xb is the original-unit stationarity residual, including Ruiz k^-1.
    dot(w->stopping_origin, xb, w->stopping_origin ? d->n : 0, ORIGIN_DUAL);""",
    )
    if text.count("solver->settings->kkt_static_reg_P, d->m, result);") != 2:
        raise ValueError("unexpected finish launch layout")
    text = text.replace(
        "solver->settings->kkt_static_reg_P, d->m, result);",
        "solver->settings->kkt_static_reg_P, d->m, result, w->stopping_offset);",
    )
    text += (
        """
// Optional extension: no ownership transfer, download, allocation or sync.
extern "C" int qoco_gpu_set_stopping_origin(QOCOSolver* solver,
    const double* origin, const double* offset) {
    if (!solver || !solver->work || ((origin == nullptr) != (offset == nullptr))) return 1;
    auto* w = solver->work;
    if (w->stopping_origin_locked &&
        (w->stopping_origin != origin || w->stopping_offset != offset)) return 2;
    w->stopping_origin = origin; w->stopping_offset = offset;
    return 0;
}
// Explicit host arithmetic oracle only; never called by graph execution.
extern "C" double qoco_gpu_reference_origin_offset(QOCOSolver* solver) {
    double result = 0.0;
    if (solver->work->stopping_offset)
"""
        "        CUDA_CHECK(cudaMemcpy(&result, solver->work->stopping_offset, "
        "sizeof(result), cudaMemcpyDeviceToHost));\n"
        """    return result;
}
"""
    )
    files[path] = text
    path = root / "algebra/cuda/qoco_metric_graph.cuh"
    files[path] = once(
        path.read_text(),
        "key.pointers = {solver, w, d, scratch, handle,",
        "key.pointers = {solver, w, d, scratch, handle, w->stopping_origin, w->stopping_offset,",
    )
    path = root / "src/qoco_utils.c"
    text = once(
        path.read_text(),
        "void qoco_reference_stopping_metrics(QOCOSolver* solver, QOCOFloat* out)",
        "double qoco_gpu_reference_origin_offset(QOCOSolver* solver);\n\n"
        "void qoco_reference_stopping_metrics(QOCOSolver* solver, QOCOFloat* out)",
    )
    text = once(
        text,
        "  QOCOFloat dres = inf_norm(xbuff, data->n);",
        """  QOCOFloat dres = inf_norm(xbuff, data->n);
  QOCOFloat origin_dual = work->stopping_origin ?
      qoco_dot(work->stopping_origin, xbuff, data->n) : 0.0;""",
    )
    text = once(
        text,
        "  gap = qoco_max(gap, qoco_abs(pobj - dobj));",
        """  gap = qoco_max(gap, qoco_abs(pobj - dobj + origin_dual));
  QOCOFloat gamma = qoco_gpu_reference_origin_offset(solver);
  pobj += gamma; dobj += gamma - origin_dual;""",
    )
    files[path] = text
    report = {}
    for path, text in files.items():
        path.write_text(text)
        report[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    (root / "spacepdhcg-origin-stopping.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, required=True)
    print(json.dumps(prepare(parser.parse_args().destination), indent=2))
