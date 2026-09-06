#!/usr/bin/env python3
"""Prepare optional persistent device stopping/best/scalar control in an isolated tree."""

import argparse
import hashlib
import json
from pathlib import Path


def once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f"Expected one source anchor: {old[:100]}")
    return text.replace(old, new, 1)


def prepare(root):
    paths = {
        name: root / name
        for name in (
            "include/structs.h",
            "src/qoco_api.c",
            "src/qoco_utils.c",
            "src/kkt.c",
            "algebra/cuda/cuda_linalg.cu",
            "algebra/cuda/qoco_batched_stopping.cuh",
            "algebra/cuda/qoco_device_step_control.cuh",
            "algebra/cuda/qoco_device_combined_rhs.cuh",
        )
    }
    texts = {name: path.read_text() for name, path in paths.items()}
    if "gpu_control" in texts["include/structs.h"]:
        raise ValueError("Device control already installed")
    if "qoco_gpu_center_and_combine(solver);" not in texts["src/kkt.c"]:
        raise ValueError("Device combined RHS preparation is required")
    texts["include/structs.h"] = once(
        texts["include/structs.h"], "} QOCOWorkspace;", "  void* gpu_control;\n} QOCOWorkspace;"
    )
    s = texts["src/qoco_api.c"]
    s = once(
        s,
        '#include "qoco_api.h"',
        '#include "qoco_api.h"\n'
        "void qoco_gpu_reset_control(QOCOSolver*);\nvoid qoco_gpu_free_control(QOCOWorkspace*);",
    )
    s = once(
        s, "  work->gpu_device_io = 0;", "  work->gpu_control = NULL;\n  work->gpu_device_io = 0;"
    )
    s = once(
        s,
        "  initialize_ipm(solver);",
        "  qoco_gpu_reset_control(solver);\n  initialize_ipm(solver);",
    )
    s = once(
        s,
        "  // Free problem data.",
        "  qoco_gpu_free_control(solver->work);\n\n  // Free problem data.",
    )
    texts["src/qoco_api.c"] = s
    s = texts["src/qoco_utils.c"]
    for function, gpu in [
        ("check_stopping", "qoco_gpu_check_stopping"),
        ("restore_best_iterate", "qoco_gpu_restore_best"),
    ]:
        declaration = f"unsigned char {function}(QOCOSolver* solver)"
        s = once(s, declaration, f"unsigned char qoco_host_{function}(QOCOSolver* solver)")
        s += (
            f"\nunsigned char {gpu}(QOCOSolver*);\n{declaration} {{\n"
            f"  if (!solver->work->gpu_control) return qoco_host_{function}(solver);\n"
            '  const char* audit = getenv("SPACEPDHCG_TEST_QOCO_DEVICE_CONTROL_COMPARE");\n'
            '  const char* metrics = getenv("SPACEPDHCG_TEST_QOCO_BATCHED_STOPPING_COMPARE");\n'
            "  if (!((audit && audit[0]=='1') || (metrics && metrics[0]=='1'))) "
            f"return {gpu}(solver);\n"
            "  int incoming = solver->sol->status;\n"
            "  qoco_gpu_sync_control(solver);\n"
            "  solver->sol->status = incoming;\n"
            f"  unsigned char expected = qoco_host_{function}(solver);\n"
            "  QOCOWorkspace* w = solver->work;\n"
            "  double reference[] = {solver->settings->kkt_dynamic_reg, solver->sol->pres,\n"
            "    solver->sol->dres, solver->sol->gap, solver->sol->obj, w->mu, w->a,\n"
            "    w->best_metric, w->best_pres, w->best_dres, w->best_gap, w->best_obj};\n"
            "  int status=solver->sol->status, best_iter=w->best_iter, best_valid=w->best_valid;\n"
            "  solver->sol->status = incoming;\n"
            f"  unsigned char actual = {gpu}(solver);\n"
            "  double result[] = {solver->settings->kkt_dynamic_reg, solver->sol->pres,\n"
            "    solver->sol->dres, solver->sol->gap, solver->sol->obj, w->mu, w->a,\n"
            "    w->best_metric, w->best_pres, w->best_dres, w->best_gap, w->best_obj};\n"
            "  int bad = expected!=actual || status!=solver->sol->status\n"
            "    || best_iter!=w->best_iter || best_valid!=w->best_valid;\n"
            "  for (int i=0;i<12;++i)\n"
            "    if (!(result[i]==reference[i] || (isnan(result[i]) && isnan(reference[i]))\n"
            "      || (isfinite(result[i]) && isfinite(reference[i])\n"
            "          && fabs(result[i]-reference[i])<=2e-12*"
            "fmax(1.0,fabs(reference[i]))))) bad=1;\n"
            '  if (bad) { fprintf(stderr, "QOCO device control audit mismatch\\n"); exit(1); }\n'
            "  return actual;\n}\n"
        )
    s = once(
        s,
        '#include "qoco_utils.h"',
        '#include "qoco_utils.h"\nvoid qoco_gpu_sync_control(QOCOSolver*);',
    )
    texts["src/qoco_utils.c"] = s
    s = texts["src/kkt.c"]
    s = once(
        s,
        '#include "kkt.h"',
        '#include "kkt.h"\nvoid qoco_gpu_note_alpha(QOCOWorkspace*, const double*);',
    )
    s = once(s, "    work->a = 0.0;", "    work->a = 0.0;\n    qoco_gpu_note_alpha(work, NULL);")
    texts["src/kkt.c"] = s
    s = texts["algebra/cuda/qoco_batched_stopping.cuh"]
    s = once(
        s,
        "static void qoco_gpu_metrics(QOCOSolver* solver, double* output)",
        "static void qoco_gpu_metrics(QOCOSolver* solver, double* output, bool device = false)",
    )
    s = once(
        s,
        "    CUDA_CHECK(cudaMemcpy(output, result, results * sizeof(double), "
        "cudaMemcpyDeviceToHost));",
        "    if (device) CUDA_CHECK(cudaMemcpyAsync(output, result, results * sizeof(double), "
        "cudaMemcpyDeviceToDevice));\n"
        "    else CUDA_CHECK(cudaMemcpy(output, result, results * sizeof(double), "
        "cudaMemcpyDeviceToHost));",
    )
    texts["algebra/cuda/qoco_batched_stopping.cuh"] = s
    s = texts["algebra/cuda/qoco_device_step_control.cuh"]
    s = 'extern "C" void qoco_gpu_note_alpha(QOCOWorkspace*, const double*);\n' + s
    s = once(
        s,
        "    CUDA_CHECK(cudaMemcpy(&w->a, scalar + 2, sizeof(double), cudaMemcpyDeviceToHost));",
        "    qoco_gpu_note_alpha(w, scalar + 2);\n"
        "    if (!w->gpu_control || audit || solver->settings->verbose)\n"
        "        CUDA_CHECK(cudaMemcpy(&w->a, scalar + 2, sizeof(double), "
        "cudaMemcpyDeviceToHost));",
    )
    texts["algebra/cuda/qoco_device_step_control.cuh"] = s
    s = texts["algebra/cuda/qoco_device_combined_rhs.cuh"]
    a = s.index(
        "#ifdef SPACEPDHCG_QOCO_QUEUED_CENTERING_METADATA",
        s.index('extern "C" void qoco_gpu_center_and_combine'),
    )
    b = s.index("#endif", a) + len("#endif")
    s = (
        s[:a]
        + (
            "    if (!w->gpu_control || audit || solver->settings->verbose)\n"
            "        CUDA_CHECK(cudaMemcpy(&w->sigma, scalar + 5, sizeof(double), "
            "cudaMemcpyDeviceToHost));"
        )
        + s[b:]
    )
    texts["algebra/cuda/qoco_device_combined_rhs.cuh"] = s
    texts["algebra/cuda/cuda_linalg.cu"] += '\n#include "qoco_device_control.cuh"\n'
    extension = Path(__file__).resolve().parents[2] / "cpp/cuda/patches/qoco_device_control.cuh"
    texts["algebra/cuda/qoco_device_control.cuh"] = extension.read_text()
    # Validate every anchor before any file is changed.
    for name, text in texts.items():
        (root / name).write_text(text)
    (root / "spacepdhcg-device-control.json").write_text(
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
