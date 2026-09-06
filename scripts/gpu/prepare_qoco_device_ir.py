#!/usr/bin/env python3
"""Add experimental device-controlled refinement to an isolated prepared QOCO tree.

Requires the existing fused-KKT, queued-operator, device-scalar, and metric-graph
patches. This script is deliberately separate from normal runtime preparation.
"""

import argparse
import hashlib
import json
from pathlib import Path


def replace_once(text: str, old: str, new: str) -> str:
    if text.count(old) != 1:
        raise ValueError(f"Expected exactly one source anchor: {old[:100]}")
    return text.replace(old, new, 1)


def prepare(destination: Path) -> None:
    patch_root = Path(__file__).resolve().parents[2] / "cpp/cuda/patches"
    backend = destination / "algebra/cuda/cudss_backend.cu"
    algebra = destination / "algebra/cuda/cuda_linalg.cu"
    cone = destination / "src/cone.cu"
    fused = destination / "algebra/cuda/qoco_fused_kkt_product.cuh"
    sources = {p: p.read_text() for p in (backend, algebra, cone, fused)}
    if "qoco_ir_runtime.cuh" in sources[backend]:
        raise ValueError("Device refinement is already installed")
    if "qoco_metric_stream" not in sources[algebra]:
        raise ValueError("Prepared metric stream support is required")
    text = sources[backend]
    text = replace_once(
        text,
        "struct LinSysData {",
        '#include "qoco_ir_runtime.cuh"\n\nstruct LinSysData {\n  QocoIrRuntime* ir;',
    )
    anchor = "  CUDSS_CHECK(g_cuda_funcs.cudssCreate(&linsys_data->handle));"
    text = replace_once(
        text,
        anchor,
        anchor + "\n  linsys_data->ir = qoco_ir_create(linsys_data->handle, linsys_data->Kn);",
    )
    text = text.replace("g_cuda_funcs.cudssExecute(", "qoco_ir_execute(linsys_data->ir, ")
    anchor = "    g_cuda_funcs.cudssDestroy(linsys_data->handle);"
    text = replace_once(text, anchor, anchor + "\n    qoco_ir_destroy(linsys_data->ir);")
    text = replace_once(
        text,
        "QOCOFloat* residual_scratch)\n{",
        "QOCOFloat* residual_scratch, bool download = true)\n{",
    )
    text = replace_once(
        text,
        "  return inf_norm(residual_scratch, N);",
        "  return download ? inf_norm(residual_scratch, N) : 0.0;",
    )
    anchor = "static void cudss_solve(LinSysData* linsys_data"
    text = replace_once(text, anchor, '#include "qoco_device_ir.cuh"\n\n' + anchor)
    anchor = "  cudss_solve_system(linsys_data, b, x);"
    text = replace_once(
        text,
        anchor,
        anchor
        + """
  if (!getenv("SPACEPDHCG_TEST_QOCO_DEVICE_IR_DISABLE")) {
    qoco_ir_solve(linsys_data, work, b, x, ir_tol, max_ir_iters);
    return;
  }
""",
    )
    sources[backend] = text
    sources[algebra] += """
extern "C" cudaStream_t qoco_ir_exchange_stream(cudaStream_t stream) {
    const auto previous = qoco_metric_stream;
    qoco_metric_stream = stream;
    return previous;
}
extern "C" cudaStream_t qoco_ir_current_stream() { return qoco_metric_stream; }
extern "C" void qoco_ir_prepare_sparse(QOCOProblemData* data) {
    if (data->P) qoco_materialize_device_transpose(data->P);
    if (data->p) qoco_materialize_device_transpose(data->A);
    if (data->m) qoco_materialize_device_transpose(data->G);
}
extern "C" void qoco_ir_device_norm(const double* x, int count, double* scratch, cudaStream_t stream) {
    using namespace qoco_device_scalar;
    const int blocks = count <= 4096 ? 1 : std::min(256, (count - 1) / 256 + 1);
    double* output = scratch + 256;
    reduce<Maximum, false><<<blocks, 256, 0, stream>>>(x, count, blocks == 1 ? output : scratch);
    if (blocks > 1) reduce<Maximum, true><<<1, 256, 0, stream>>>(scratch, blocks, output);
    CUDA_CHECK(cudaGetLastError());
}
"""  # noqa: E501 - preserve the emitted C++ declaration exactly.
    sources[fused] = replace_once(
        sources[fused],
        "product<<<(count+255)/256,256>>>",
        "product<<<(count+255)/256,256,0,qoco_metric_stream>>>",
    )
    sources[cone] = replace_once(
        sources[cone],
        "void nt_multiply(QOCOFloat* W",
        'extern "C" cudaStream_t qoco_ir_current_stream();\n\nvoid nt_multiply(QOCOFloat* W',
    )
    # Both direct and inverse NT products respect the selected operator stream.
    if sources[cone].count("nt_multiply_kernel<<<blocks, threads>>>") != 2:
        raise ValueError("Unexpected NT multiplication launch layout")
    sources[cone] = sources[cone].replace(
        "nt_multiply_kernel<<<blocks, threads>>>",
        "nt_multiply_kernel<<<blocks, threads, 0, qoco_ir_current_stream()>>>",
    )
    for name in ("qoco_ir_runtime.cuh", "qoco_device_ir.cuh"):
        sources[destination / "algebra/cuda" / name] = (patch_root / name).read_text()
    # Validate all anchors before mutating the disposable prepared tree.
    for path, contents in sources.items():
        path.write_text(contents)
    manifest = {
        "experimental": True,
        "scope": "Device residual/backup/stop control; graph scoped to each linear solve",
        "files": {
            str(p.relative_to(destination)): hashlib.sha256(t.encode()).hexdigest()
            for p, t in sources.items()
        },
    }
    (destination / "spacepdhcg-device-ir.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, required=True)
    prepare(parser.parse_args().destination.resolve())
