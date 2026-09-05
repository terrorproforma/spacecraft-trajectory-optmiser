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
    args = parser.parse_args()
    if args.unmodified and (
        args.gather
        or args.correct_stopping
        or args.deterministic
        or args.checked_cudss_abi
        or args.queued_operators
        or args.device_cone_reductions
    ):
        parser.error("--unmodified cannot be combined with backend changes")
    if (args.queued_operators or args.device_cone_reductions) and args.original_handles:
        parser.error("queued operators and device cone reductions require scoped handles")
    if args.queued_operators and not args.gather:
        parser.error("--queued-operators requires --gather")
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
        )
    }
    (destination / "spacepdhcg-provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
