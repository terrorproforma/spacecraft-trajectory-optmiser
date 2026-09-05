"""Prepare an isolated pinned QOCO-GPU source copy with scoped reduction handles.

Build the resulting copy with QOCO_ALGEBRA_BACKEND=cuda. The source repository is
read-only: QOCO's configure step writes a header into its source directory, so it
must never be configured against the shared pinned checkout for this experiment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path

PIN = "09f049597deef2a7ead15b3da19a9456ff7d4e53"


def patch_device_step_control(
    destination: Path, extension: Path, combined: bool = False
) -> None:
    """Queue cone step consumers while preserving the reference centering oracle."""
    for name, directory in (("qoco_device_step_cones.cuh", "src"),
                            ("qoco_device_step_control.cuh", "algebra/cuda")):
        shutil.copyfile(extension.with_name(name), destination / directory / name)
    path = destination / "src/cone.cu"
    text = path.read_text()
    before = "void compute_centering(QOCOSolver* solver)"
    if text.count(before) != 1:
        raise RuntimeError("unexpected centering definition")
    text = text.replace(
        before, 'extern "C" void qoco_reference_compute_centering(QOCOSolver* solver)'
    )
    text += '''
#include "qoco_device_step_cones.cuh"
extern "C" void qoco_gpu_compute_centering(QOCOSolver*);
void compute_centering(QOCOSolver* solver) { qoco_gpu_compute_centering(solver); }
'''
    path.write_text(text)
    path = destination / "src/kkt.c"
    text = path.read_text()
    start = text.index("  // Compute step-size.", text.index("void predictor_corrector("))
    end = text.index("\n}\n", start)
    text = text[:start] + "  qoco_gpu_take_step(solver);\n" + text[end:]
    before = "void predictor_corrector(QOCOSolver* solver)"
    text = text.replace(before, "void qoco_gpu_take_step(QOCOSolver* solver);\n\n" + before)
    if combined:
        for name, directory in (("qoco_device_combined_cones.cuh", "src"),
                                ("qoco_device_combined_rhs.cuh", "algebra/cuda")):
            shutil.copyfile(extension.with_name(name), destination / directory / name)
        cone = destination / "src/cone.cu"
        cone.write_text(cone.read_text() + '\n#include "qoco_device_combined_cones.cuh"\n')
        before = ("  // Compute centering parameter.\n  compute_centering(solver);\n\n"
                  "  // Construct rhs for combined direction.\n  construct_kkt_comb_rhs(work);")
        if text.count(before) != 1:
            raise RuntimeError("unexpected predictor centering/RHS sequence")
        text = text.replace(before, "  qoco_gpu_center_and_combine(solver);")
        before = "void qoco_gpu_take_step(QOCOSolver* solver);"
        text = text.replace(before, before + "\nvoid qoco_gpu_center_and_combine(QOCOSolver*);")
    path.write_text(text)


def patch_batched_stopping(destination: Path, iteration: bool = False) -> None:
    """Keep scalar metrics on device until one packet completes the calculation."""
    header = destination / "algebra/cuda/cudss_backend.h"
    text = header.read_text().replace(
        "  // cuBLAS function pointers",
        "  decltype(&::cublasSetPointerMode) cublasSetPointerMode;\n"
        "  decltype(&::cublasGetPointerMode) cublasGetPointerMode;\n"
        "  // cuBLAS function pointers",
    )
    header.write_text(text)
    backend = destination / "algebra/cuda/cudss_backend.cu"
    text = backend.read_text().replace(
        "  // Load cuBLAS functions",
        """  g_cuda_funcs.cublasSetPointerMode =
      reinterpret_cast<decltype(&::cublasSetPointerMode)>(
          dlsym(g_cublas_handle, "cublasSetPointerMode_v2"));
  g_cuda_funcs.cublasGetPointerMode =
      reinterpret_cast<decltype(&::cublasGetPointerMode)>(
          dlsym(g_cublas_handle, "cublasGetPointerMode_v2"));
  if (!g_cuda_funcs.cublasSetPointerMode || !g_cuda_funcs.cublasGetPointerMode) {
    fprintf(stderr, "QOCO cuBLAS pointer-mode symbols missing\\n");
    dlclose(g_cublas_handle); dlclose(g_cusparse_handle); dlclose(g_cudss_handle);
    return 0;
  }
  // Load cuBLAS functions""",
    )
    backend.write_text(text)
    path = destination / "src/qoco_utils.c"
    text = path.read_text()
    function = text.index("unsigned char check_stopping(QOCOSolver* solver)")
    begin = text.index("  QOCOFloat* xbuff", function)
    end = text.index("  // Composite progress metric", begin)
    reference = text[begin:end]
    for field in ("pres", "dres", "gap"):
        reference = reference.replace(f"  solver->sol->{field} = {field};\n", "")
    reference = (
        "void qoco_reference_stopping_metrics(QOCOSolver* solver, QOCOFloat* out)\n{\n"
        "  QOCOWorkspace* work = solver->work;\n"
        "  QOCOProblemData* data = work->data;\n" + reference
        + "  out[0] = pres; out[1] = dres; out[2] = gap;\n"
        "  out[3] = pres_rel; out[4] = dres_rel; out[5] = gap_rel;\n}\n\n"
        "void qoco_gpu_stopping_metrics(QOCOSolver* solver, double* out);\n\n"
    )
    replacement = """  QOCOFloat metrics[6];
  qoco_gpu_stopping_metrics(solver, metrics);
  const char* compare = getenv("SPACEPDHCG_TEST_QOCO_BATCHED_STOPPING_COMPARE");
  if (compare && compare[0] == '1') {
    QOCOFloat reference[6];
    qoco_reference_stopping_metrics(solver, reference);
    for (int i = 0; i < 6; ++i) {
      if (!isfinite(metrics[i]) || !isfinite(reference[i]) ||
          fabs(metrics[i] - reference[i]) > 2e-12 * fmax(1.0, fabs(reference[i]))) {
        fprintf(stderr, "QOCO batched stopping metric %d mismatch %.17g %.17g\\n",
                i, metrics[i], reference[i]);
        exit(1);
      }
    }
  }
  QOCOFloat pres = metrics[0], dres = metrics[1], gap = metrics[2];
  QOCOFloat pres_rel = metrics[3], dres_rel = metrics[4], gap_rel = metrics[5];
  solver->sol->pres = pres; solver->sol->dres = dres; solver->sol->gap = gap;

"""
    prefix = text[function:begin].replace(
        "  QOCOProblemData* data = solver->work->data;\n", ""
    )
    if iteration:
        reference += "void qoco_gpu_iteration_metrics(QOCOSolver* solver, double* out);\n\n"
        replacement = replacement.replace("metrics[6]", "metrics[8]").replace(
            "qoco_gpu_stopping_metrics(solver, metrics)",
            "qoco_gpu_iteration_metrics(solver, metrics)",
        ).replace("reference[6]", "reference[8]").replace("i < 6", "i < 8")
        replacement = replacement.replace(
            "    qoco_reference_stopping_metrics(solver, reference);",
            """    qoco_reference_stopping_metrics(solver, reference);
    reference[6] = compute_objective(work->data, work->x, work->xbuff,
        solver->settings->kkt_static_reg_P, work->scaling->k);
    reference[7] = compute_mu(work->s, work->z, work->data->m);""",
        )
        replacement += "  solver->sol->obj = metrics[6]; work->mu = metrics[7];\n\n"
        api_path = destination / "src/qoco_api.c"
        api = api_path.read_text()
        api_begin = api.index(
            "    // Compute objective function.", api.index("QOCOInt qoco_solve(")
        )
        api_end = api.index("    // Check stopping criteria.", api_begin)
        api_path.write_text(api[:api_begin] + api[api_end:])
    output = text[:function] + reference + prefix + replacement + text[end:]
    if iteration:
        output = '#include "kkt.h"\n' + output
    path.write_text(output)


def patch_cudss_abi(destination: Path) -> None:
    """Compile against the real API and reject a different runtime major/minor."""
    header = destination / "algebra/cuda/cudss_backend.h"
    contents, count = re.subn(
        r"  cudssStatus_t \(\*(cudss\w+)\)\([\s\S]*?\);",
        lambda match: f"  decltype(&::{match[1]}) {match[1]};",
        header.read_text(),
    )
    if count != 12:
        raise RuntimeError(f"unexpected cuDSS function table ({count} entries)")
    header.write_text(contents)
    path = destination / "algebra/cuda/cudss_backend.cu"
    contents = path.read_text()
    types = (
        "  cudaDataType_t indexType = CUDA_R_32I; // QOCOInt is int32_t\n"
        "  cudaDataType_t valueType_setup =\n"
        "      (sizeof(QOCOFloat) == 8) ? CUDA_R_64F : CUDA_R_32F;"
    )
    typed = (
        "#if CUDSS_VERSION >= 800\n"
        "  cudssDataType_t indexType = CUDSS_R_32I;\n"
        "  cudssDataType_t valueType_setup =\n"
        "      (sizeof(QOCOFloat) == 8) ? CUDSS_R_64F : CUDSS_R_32F;\n"
        "#else\n" + types + "\n#endif"
    )
    args = "csr_col_ind, csr_val, indexType,\n      valueType_setup"
    typed_args = (
        "csr_col_ind, csr_val,\n#if CUDSS_VERSION >= 800\n"
        "      indexType,\n#endif\n      indexType, valueType_setup"
    )
    guard = """  // The CSR signature changes in cuDSS 0.8. Never call a mismatched ABI.
  auto property = reinterpret_cast<decltype(&::cudssGetProperty)>(
      dlsym(g_cudss_handle, "cudssGetProperty"));
  int major = -1, minor = -1;
  if (!property || property(MAJOR_VERSION, &major) != CUDSS_STATUS_SUCCESS
      || property(MINOR_VERSION, &minor) != CUDSS_STATUS_SUCCESS
      || major != CUDSS_VERSION_MAJOR || minor != CUDSS_VERSION_MINOR) {
    fprintf(stderr, "QOCO cuDSS header/runtime ABI mismatch (%d.%d vs %d.%d)\\n",
            CUDSS_VERSION_MAJOR, CUDSS_VERSION_MINOR, major, minor);
    dlclose(g_cudss_handle);
    g_cudss_handle = nullptr;
    return 0;
  }

"""
    for before, after in (
        (types, typed),
        (args, typed_args),
        ("  // Load cuSPARSE\n", guard + "  // Load cuSPARSE\n"),
    ):
        if contents.count(before) != 1:
            raise RuntimeError(f"unexpected cuDSS ABI patch site: {before}")
        contents = contents.replace(before, after)
    path.write_text(contents)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--unmodified", action="store_true", help="prepare a matched build control")
    parser.add_argument(
        "--gather", action="store_true", help="use deterministic GPU sparse gathers"
    )
    parser.add_argument(
        "--original-handles", action="store_true", help="retain per-reduction handles"
    )
    parser.add_argument(
        "--correct-stopping", action="store_true", help="correct dual residual scaling"
    )
    parser.add_argument(
        "--deterministic", action="store_true", help="use deterministic cuDSS factors"
    )
    parser.add_argument(
        "--checked-cudss-abi", action="store_true", help="support and check cuDSS 0.7/0.8 APIs"
    )
    parser.add_argument(
        "--queued-operators",
        action="store_true",
        help="queue vector/gather operations within a solve",
    )
    parser.add_argument(
        "--device-cone-reductions", action="store_true",
        help="GPU cone reductions, retained scratch and SOC boundary fixes",
    )
    parser.add_argument(
        "--values-only-updates", action="store_true",
        help="retain sparse topology during numeric updates",
    )
    parser.add_argument(
        "--device-numeric-updates", action="store_true",
        help="expose device coefficient updates and GPU Ruiz equilibration",
    )
    parser.add_argument(
        "--device-scalar-reductions", action="store_true",
        help="reduce extrema and NaN checks on CUDA with retained scalar scratch",
    )
    parser.add_argument(
        "--batched-stopping", action="store_true",
        help="compute stopping metrics on GPU and return one scalar packet",
    )
    parser.add_argument(
        "--batched-iteration-scalars", action="store_true",
        help="include objective and mu in the stopping packet and share matrix products",
    )
    parser.add_argument("--device-step-control", action="store_true",
                        help="queue GPU centering and fused iterate updates")
    parser.add_argument("--device-combined-rhs", action="store_true",
                        help="fuse combined cone correction using device centering scalars")
    parser.add_argument("--queued-centering-metadata", action="store_true",
                        help="pin the workspace and queue the sigma metadata download")
    args = parser.parse_args()
    if args.unmodified and (
        args.gather
        or args.correct_stopping
        or args.deterministic
        or args.checked_cudss_abi
        or args.queued_operators
        or args.device_cone_reductions
        or args.values_only_updates
        or args.device_numeric_updates
        or args.device_scalar_reductions
        or args.batched_stopping
        or args.batched_iteration_scalars
        or args.device_step_control
        or args.device_combined_rhs
        or args.queued_centering_metadata
    ):
        parser.error("--unmodified cannot be combined with backend changes")
    if (args.queued_operators or args.device_cone_reductions) and args.original_handles:
        parser.error("queued operators and device cone reductions require scoped handles")
    if args.queued_operators and not args.gather:
        parser.error("--queued-operators requires --gather")
    if args.device_numeric_updates and not args.gather:
        parser.error("--device-numeric-updates requires --gather")
    if args.device_scalar_reductions and not args.device_cone_reductions:
        parser.error("--device-scalar-reductions requires --device-cone-reductions")
    if args.batched_stopping and not (args.device_scalar_reductions and args.queued_operators):
        parser.error(
            "--batched-stopping requires --device-scalar-reductions and --queued-operators"
        )
    if args.batched_stopping and not args.correct_stopping:
        parser.error("--batched-stopping requires --correct-stopping")
    if args.batched_iteration_scalars and not args.batched_stopping:
        parser.error("--batched-iteration-scalars requires --batched-stopping")
    if args.device_step_control and not args.batched_stopping:
        parser.error("--device-step-control requires --batched-stopping")
    if args.device_combined_rhs and not args.device_step_control:
        parser.error("--device-combined-rhs requires --device-step-control")
    if args.queued_centering_metadata and not args.device_combined_rhs:
        parser.error("--queued-centering-metadata requires --device-combined-rhs")
    source = args.source.resolve()
    destination = args.destination.resolve()
    if destination.is_relative_to(source) or source.is_relative_to(destination):
        parser.error("source and destination must be separate directory trees")
    commit = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    if commit != PIN:
        parser.error(f"expected reviewed QOCO commit {PIN}, got {commit}")
    if subprocess.check_output(["git", "-C", str(source), "status", "--porcelain"], text=True):
        parser.error("the pinned source must be clean before copying")
    destination.mkdir(parents=True, exist_ok=False)
    files = (
        subprocess.check_output(["git", "-C", str(source), "ls-files", "-z"]).decode().split("\0")
    )
    for relative in files:
        if not relative or not (source / relative).is_file():
            continue  # The benchmark-data submodule is not needed to build the library.
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / relative, target)
    extension = Path(__file__).resolve().parents[2] / "cpp/cuda/patches/qoco_reduction_scope.cuh"
    if args.unmodified:
        (destination / "spacepdhcg-provenance.json").write_text(
            json.dumps(
                {"upstream_commit": commit, "source": str(source), "unmodified": True}, indent=2
            )
            + "\n"
        )
        return
    shutil.copyfile(extension, destination / "algebra/cuda/qoco_reduction_scope.cuh")
    path = destination / "algebra/cuda/cuda_linalg.cu"
    original = path.read_text()
    modified = original.replace(
        '#include "cudss_backend.h"',
        '#include "cudss_backend.h"\n#include "qoco_reduction_scope.cuh"',
    )
    create = "    cublasHandle_t handle;\n    funcs->cublasCreate(&handle);"
    destroy = "    funcs->cublasDestroy(handle);"
    if modified.count(create) != 3 or modified.count(destroy) != 3:
        raise RuntimeError("unexpected upstream reduction implementation")
    if not args.original_handles:
        modified = modified.replace(
            create,
            "    bool temporary_handle;\n"
            "    cublasHandle_t handle = qoco_acquire_reduction_handle(&temporary_handle);",
        )
        modified = modified.replace(
            destroy, "    if (temporary_handle) funcs->cublasDestroy(handle);"
        )
    else:
        modified = original
    if args.gather:
        gather = extension.with_name("qoco_gather.cuh")
        shutil.copyfile(gather, destination / "algebra/cuda/qoco_gather.cuh")
        types_path = destination / "algebra/cuda/cuda_types.h"
        types = types_path.read_text().replace(
            "struct QOCOMatrix_ {",
            "struct QocoGpuGather;\nstruct QOCOMatrix_ {\n  QocoGpuGather* gather;",
        )
        types_path.write_text(types)
        replacements = {
            "static inline bool is_device_pointer": (
                '#include "qoco_gather.cuh"\n\nstatic inline bool is_device_pointer'
            ),
            "  QOCOMatrix* M = (QOCOMatrix*)qoco_malloc(sizeof(QOCOMatrix));": (
                "  QOCOMatrix* M = (QOCOMatrix*)qoco_malloc(sizeof(QOCOMatrix));\n"
                "  M->gather = nullptr;"
            ),
            "  return M;\n}\n\n// Construct x": (
                "  qoco_gpu_create_gather(M);\n  return M;\n}\n\n// Construct x"
            ),
            "  // Free host CSC": "  qoco_gpu_free_gather(A);\n  // Free host CSC",
            "                          (M->csc->n + 1) * sizeof(QOCOInt),\n"
            "                          cudaMemcpyHostToDevice));": (
                "                          (M->csc->n + 1) * sizeof(QOCOInt),\n"
                "                          cudaMemcpyHostToDevice));\n"
                "    qoco_gpu_refresh_gather(M);"
            ),
            "  QOCOInt col = blockIdx.x;\n  if (col >= M->n)": (
                "  QOCOInt col = blockIdx.x * blockDim.x + threadIdx.x;\n  if (col >= M->n)"
            ),
            "SpMtv_kernel<<<M->d_csc_host->n, 1>>>": (
                "SpMtv_kernel<<<(M->d_csc_host->n + 255) / 256, 256>>>"
            ),
        }
        for before, after in replacements.items():
            count = modified.count(before)
            expected = 2 if before.startswith("  QOCOInt col") else 1
            if count != expected:
                raise RuntimeError(f"unexpected upstream gather site ({count}): {before}")
            modified = modified.replace(before, after)
        for name, symmetric in (("USpMv", "true"), ("SpMv", "false")):
            start = modified.index(f"void {name}(const QOCOMatrix* M,")
            end = modified.index("\n}\n", start) + 3
            modified = (
                modified[:start]
                + f"void {name}(const QOCOMatrix* M, const QOCOFloat* v, QOCOFloat* r)\n"
                + "{\n"
                + f"  qoco_gpu_product<{symmetric}>(M, v, r);\n"
                + "}\n"
                + modified[end:]
            )
    device_solution = extension.with_name("qoco_device_solution.cuh")
    shutil.copyfile(device_solution, destination / "algebra/cuda/qoco_device_solution.cuh")
    modified = modified.replace(
        '#include "cudss_backend.h"',
        '#include "cudss_backend.h"\n#include "qoco_device_solution.cuh"',
    )
    if args.queued_operators:
        modified = modified.replace(
            '#include "qoco_reduction_scope.cuh"',
            '#define SPACEPDHCG_QOCO_QUEUED_OPERATORS 1\n#include "qoco_reduction_scope.cuh"',
        )
        # After gather replacement there are exactly six vector/transpose waits.
        # Setup, matrix refresh, cuDSS, and all host reads retain their barriers.
        if not args.gather or modified.count("cudaDeviceSynchronize()") != 6:
            raise RuntimeError("queued operators require reviewed gather/vector implementation")
        modified = modified.replace("cudaDeviceSynchronize()", "qoco_complete_vector_operation()")
        gather_path = destination / "algebra/cuda/qoco_gather.cuh"
        contents = gather_path.read_text()
        before, separator, after = contents.rpartition("cudaDeviceSynchronize()")
        if not separator or contents.count(separator) != 2:
            raise RuntimeError("unexpected gather completion sites")
        gather_path.write_text(before + "qoco_complete_vector_operation()" + after)
    if args.device_cone_reductions:
        modified = modified.replace(
            '#include "qoco_reduction_scope.cuh"',
            '#define SPACEPDHCG_QOCO_DEVICE_CONE_REDUCTIONS 1\n'
            '#include "qoco_reduction_scope.cuh"',
        )
        cone_path = destination / "src/cone.cu"
        cones = cone_path.read_text()
        # The pinned linear/boundary branches can step outside an SOC. Preserve
        # the quadratic formula, but enforce the first boundary of c+b*t+a*t^2.
        linear = "  if (qoco_abs(a) < 1e-14)\n    return alpha;"
        boundary = (
            "  if (c == 0.0) {\n    if (a >= 0.0)\n      return alpha;\n"
            "    else\n      return 0.0;\n  }"
        )
        for before, after in (
            (linear, "  if (a == 0.0)\n    return b < 0.0 ? qoco_min(alpha, -c / b) : alpha;"),
            (boundary, "  if (c == 0.0) {\n"
             "    if (b < 0.0 || (b == 0.0 && a < 0.0)) return 0.0;\n"
             "    return a < 0.0 ? qoco_min(alpha, -b / a) : alpha;\n  }"),
        ):
            if cones.count(before) != 1:
                raise RuntimeError("unexpected SOC step-length boundary implementation")
            cones = cones.replace(before, after)
        for signature in ("QOCOFloat linesearch(", "QOCOFloat cone_residual("):
            start = cones.index(signature)
            end = cones.index("\n}\n", start) + 3
            declaration = cones[start:cones.index("\n{", start)] + ";\n"
            cones = cones[:start] + declaration + cones[end:]
        replacement = extension.with_name("qoco_device_cone_reductions.cuh")
        shutil.copyfile(replacement, destination / "src/qoco_device_cone_reductions.cuh")
        cone_path.write_text(cones + '\n#include "qoco_device_cone_reductions.cuh"\n')
    if args.values_only_updates:
        # qoco_update_matrix_data changes coefficients only. Public full matrix
        # synchronization still uploads indices and refreshes gather topology.
        header_path = destination / "include/qoco_linalg.h"
        header = header_path.read_text()
        declaration = "void sync_matrix_to_device(QOCOMatrix* M);"
        if header.count(declaration) != 1:
            raise RuntimeError("unexpected matrix synchronization declaration")
        header_path.write_text(header.replace(
            declaration, declaration + "\nvoid sync_matrix_values_to_device(QOCOMatrix* M);"
        ))
        modified += """
void sync_matrix_values_to_device(QOCOMatrix* M)
{
  if (M && M->d_csc_host && M->csc->nnz > 0) {
    CUDA_CHECK(cudaMemcpy(M->d_csc_host->x, M->csc->x,
                          M->csc->nnz * sizeof(QOCOFloat), cudaMemcpyHostToDevice));
  }
}
"""
        api_path = destination / "src/qoco_api.c"
        api = api_path.read_text()
        for matrix in ("P", "A", "G"):
            before = f"  sync_matrix_to_device(data->{matrix});"
            if api.count(before) != 1:
                raise RuntimeError("unexpected QOCO numeric update synchronization")
            api = api.replace(before, f"  sync_matrix_values_to_device(data->{matrix});")
        api_path.write_text(api)
    if args.device_numeric_updates:
        # Initial Ruiz setup still uses host algebra. The pinned CUDA backend's
        # host scale_arrayf branch is absent, silently dropping cost scaling.
        start = modified.index("void scale_arrayf(const QOCOFloat*")
        end = modified.index("\n}\n", start)
        modified = (modified[:end] + "\n  else {\n"
                    "    for (QOCOInt i = 0; i < n; ++i) y[i] = s * x[i];\n  }"
                    + modified[end:])
        api_path = destination / "src/qoco_api.c"
        api = api_path.read_text()
        for matrix in ("A", "G"):
            before = f"{matrix}tx[i] = {matrix}xnew[data->{matrix}to{matrix}t[i]];"
            if api.count(before) != 1:
                raise RuntimeError("unexpected transpose update map")
            api = api.replace(
                before, f"{matrix}tx[data->{matrix}to{matrix}t[i]] = {matrix}xnew[i];"
            )
        start = api.index("    QOCOInt avoid = data->Pnum_nzadded")
        end = api.index("\n  }\n", start)
        api = api[:start] + """    QOCOInt source = 0, added = 0;
    for (QOCOInt i = 0; i < Pnnz; ++i) {
      if (added < data->Pnum_nzadded && i == data->Pnzadded_idx[added]) {
        Px[i] = 0.0;
        ++added;
      } else {
        Px[i] = Pxnew[source++];
      }
    }""" + api[end:]
        api_path.write_text(api)
        update_extension = extension.with_name("qoco_device_update.cuh")
        shutil.copyfile(update_extension, destination / "algebra/cuda/qoco_device_update.cuh")
        modified += '\n#include "qoco_device_update.cuh"\n'
    if args.device_scalar_reductions:
        scalar = extension.with_name("qoco_device_scalar.cuh")
        shutil.copyfile(scalar, destination / "algebra/cuda/qoco_device_scalar.cuh")
        modified = modified.replace(
            "QOCOFloat inf_norm(const QOCOFloat* x, QOCOInt n)",
            '#include "qoco_device_scalar.cuh"\n\n'
            "QOCOFloat inf_norm(const QOCOFloat* x, QOCOInt n)",
        )
        for function, operation in (("inf_norm", "Maximum"), ("min_abs_val", "Minimum")):
            start = modified.index(f"QOCOFloat {function}(const QOCOFloat* x,")
            begin = modified.index("  if (is_device_pointer(x)) {", start)
            end = modified.index("  else {", begin)
            modified = (modified[:begin] + "  if (is_device_pointer(x)) {\n"
                        "    return qoco_device_scalar::run<"
                        f"qoco_device_scalar::{operation}>(x, n);\n"
                        "  }\n" + modified[end:])
        start = modified.index("QOCOInt check_nan(const QOCOVectorf* x)")
        end = modified.index("\n}\n", start) + 3
        modified = (modified[:start] + "QOCOInt check_nan(const QOCOVectorf* x)\n{\n"
                    "  return static_cast<QOCOInt>(qoco_device_scalar::run<"
                    "qoco_device_scalar::HasNan>(x->d_data, x->len));\n}\n"
                    + modified[end:])
    if args.batched_stopping:
        stopping = extension.with_name("qoco_batched_stopping.cuh")
        shutil.copyfile(stopping, destination / "algebra/cuda/qoco_batched_stopping.cuh")
        if args.batched_iteration_scalars:
            modified += '\n#define SPACEPDHCG_QOCO_BATCHED_ITERATION 1\n'
        modified += '\n#include "qoco_batched_stopping.cuh"\n'
    if args.device_step_control:
        modified += '\n#include "qoco_device_step_control.cuh"\n'
        patch_device_step_control(destination, extension, args.device_combined_rhs)
    if args.device_combined_rhs:
        if args.queued_centering_metadata:
            modified += '\n#define SPACEPDHCG_QOCO_QUEUED_CENTERING_METADATA 1\n'
            api_path = destination / "src/qoco_api.c"
            api = api_path.read_text()
            for before, after in (
                ("solver->work = qoco_malloc(sizeof(QOCOWorkspace));",
                 "solver->work = qoco_gpu_allocate_workspace(sizeof(QOCOWorkspace));"),
                ("qoco_free(solver->work);", "qoco_gpu_free_workspace(solver->work);"),
            ):
                if api.count(before) != 1:
                    raise RuntimeError("unexpected workspace lifetime site")
                api = api.replace(before, after)
            before = "QOCOInt qoco_setup("
            api = api.replace(before, "void* qoco_gpu_allocate_workspace(size_t);\n"
                              "void qoco_gpu_free_workspace(void*);\n\n" + before)
            api_path.write_text(api)
        modified += '\n#include "qoco_device_combined_rhs.cuh"\n'
    path.write_text(modified)
    if args.correct_stopping:
        utils_path = destination / "src/qoco_utils.c"
        utils = utils_path.read_text()
        before = "  ew_product(Dinvruiz_data, xdata, xbuff, data->n);\n  QOCOFloat cinf"
        after = (
            "  ew_product(Dinvruiz_data, get_data_vectorf(data->c), xbuff, data->n);\n"
            "  QOCOFloat cinf"
        )
        if utils.count(before) != 1:
            raise RuntimeError("unexpected upstream stopping criterion")
        utils_path.write_text(utils.replace(before, after))
    if args.deterministic:
        backend_path = destination / "algebra/cuda/cudss_backend.cu"
        backend = backend_path.read_text()
        before = "  // Initialize cuSPARSE"
        after = (
            "  value = 1;\n"
            "  CUDSS_CHECK(g_cuda_funcs.cudssConfigSet(linsys_data->config,\n"
            "      CUDSS_CONFIG_DETERMINISTIC_MODE, &value, sizeof(value)));\n" + before
        )
        if backend.count(before) != 1:
            raise RuntimeError("unexpected cuDSS initialization")
        backend_path.write_text(backend.replace(before, after))
    if args.checked_cudss_abi:
        patch_cudss_abi(destination)
    if args.batched_stopping:
        patch_batched_stopping(destination, args.batched_iteration_scalars)
    provenance = {
        "upstream_commit": commit,
        "source": str(source),
        "original_cuda_linalg_sha256": hashlib.sha256(original.encode()).hexdigest(),
        "modified_cuda_linalg_sha256": hashlib.sha256(modified.encode()).hexdigest(),
        "extension_sha256": hashlib.sha256(extension.read_bytes()).hexdigest(),
        "device_solution_sha256": hashlib.sha256(device_solution.read_bytes()).hexdigest(),
        "gather": args.gather,
        "original_handles": args.original_handles,
        "queued_operators": args.queued_operators,
        "device_cone_reductions": args.device_cone_reductions,
        "values_only_updates": args.values_only_updates,
        "device_numeric_updates": args.device_numeric_updates,
        "device_scalar_reductions": args.device_scalar_reductions,
        "batched_stopping": args.batched_stopping,
        "batched_iteration_scalars": args.batched_iteration_scalars,
        "device_step_control": args.device_step_control,
        "device_combined_rhs": args.device_combined_rhs,
        "queued_centering_metadata": args.queued_centering_metadata,
        "correct_stopping": args.correct_stopping,
        "deterministic": args.deterministic,
        "checked_cudss_abi": args.checked_cudss_abi,
    }
    if args.gather:
        provenance["gather_sha256"] = hashlib.sha256(gather.read_bytes()).hexdigest()
    provenance["prepared_files_sha256"] = {
        relative: hashlib.sha256((destination / relative).read_bytes()).hexdigest()
        for relative in (
            "algebra/cuda/cuda_linalg.cu",
            "algebra/cuda/qoco_reduction_scope.cuh",
            "algebra/cuda/qoco_device_solution.cuh",
            "algebra/cuda/cuda_types.h",
            "algebra/cuda/cudss_backend.h",
            "algebra/cuda/cudss_backend.cu",
            "src/qoco_utils.c",
            *(["algebra/cuda/qoco_gather.cuh"] if args.gather else []),
            *(["src/cone.cu", "src/qoco_device_cone_reductions.cuh"]
              if args.device_cone_reductions else []),
            *(["include/qoco_linalg.h"] if args.values_only_updates else []),
            *(["src/qoco_api.c"]
              if args.values_only_updates or args.device_numeric_updates
              or args.batched_iteration_scalars or args.queued_centering_metadata else []),
            *(["algebra/cuda/qoco_device_update.cuh"] if args.device_numeric_updates else []),
            *(["algebra/cuda/qoco_device_scalar.cuh"] if args.device_scalar_reductions else []),
            *(["algebra/cuda/qoco_batched_stopping.cuh"] if args.batched_stopping else []),
            *(["src/kkt.c", "src/qoco_device_step_cones.cuh",
               "algebra/cuda/qoco_device_step_control.cuh"] if args.device_step_control else []),
            *(["src/qoco_device_combined_cones.cuh", "algebra/cuda/qoco_device_combined_rhs.cuh"]
              if args.device_combined_rhs else []),
        )
    }
    (destination / "spacepdhcg-provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
