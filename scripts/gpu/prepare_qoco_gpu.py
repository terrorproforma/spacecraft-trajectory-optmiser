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


def patch_deferred_transposes(destination: Path, extension: Path) -> None:
    """Retain source matrices and materialize legacy transposes on demand."""
    path = destination / "algebra/cuda/cuda_types.h"
    text = path.read_text()
    marker = "struct QOCOMatrix_ {"
    if text.count(marker) != 1:
        raise RuntimeError("unexpected deferred matrix metadata site")
    path.write_text(text.replace(marker, "#define SPACEPDHCG_QOCO_DEFERRED_TRANSPOSES 1\n"
        + marker + "\n  int reference_count;\n  QOCOMatrix* transpose_source;\n"
        "  QOCOInt* transpose_map;\n  int transpose_values_pending;"))
    shutil.copyfile(extension, destination / "algebra/cuda/qoco_deferred_transpose.cuh")
    path = destination / "algebra/cuda/cuda_linalg.cu"
    text = path.read_text()
    replacements = {
        '#include "qoco_lazy_host_mirror.cuh"':
            'static void qoco_materialize_device_transpose(const QOCOMatrix*);\n'
            '#include "qoco_lazy_host_mirror.cuh"',
        '#include "qoco_gpu_transpose.cuh"':
            '#include "qoco_gpu_transpose.cuh"\n#include "qoco_deferred_transpose.cuh"',
        "  M->host_values_pending = 0;":
            "  M->host_values_pending = 0;\n  M->reference_count = 1;\n"
            "  M->transpose_source = nullptr;\n  M->transpose_map = nullptr;\n"
            "  M->transpose_values_pending = 0;",
        "  qoco_gpu_free_gather(A);":
            "  if (--A->reference_count) return;\n"
            "  auto* retained_source = A->transpose_source;\n  qoco_gpu_free_gather(A);",
        "  qoco_free(A);\n  CUDA_CHECK(cudaGetLastError());":
            "  qoco_free(A);\n  free_qoco_matrix(retained_source);\n"
            "  CUDA_CHECK(cudaGetLastError());",
        "QOCOCscMatrix* get_csc_matrix(const QOCOMatrix* M)\n{":
            "QOCOCscMatrix* get_csc_matrix(const QOCOMatrix* M)\n{\n"
            "  qoco_materialize_device_transpose(M);",
    }
    for function in ["USpMv", "SpMv", "SpMtv"]:
        before = f"void {function}(const QOCOMatrix* M, const QOCOFloat* v, QOCOFloat* r)\n{{"
        replacements[before] = before + "\n  qoco_materialize_device_transpose(M);"
    for before, after in replacements.items():
        if text.count(before) != 1:
            raise RuntimeError(f"unexpected deferred transpose site: {before}")
        text = text.replace(before, after)
    path.write_text(text)
    path = destination / "src/qoco_api.c"
    text = path.read_text()
    if text.count("qoco_gpu_transpose_lazy(") != 3:
        raise RuntimeError("unexpected deferred transpose setup sites")
    text = text.replace("qoco_gpu_transpose_lazy(", "qoco_gpu_transpose_deferred(")
    # The internal constructor retains its mutable source metadata.
    text = text.replace("qoco_gpu_transpose_deferred(const QOCOMatrix*",
                        "qoco_gpu_transpose_deferred(QOCOMatrix*")
    path.write_text(text)


def patch_lazy_transpose_mirrors(destination: Path, extension: Path) -> None:
    """Materialize transpose host arrays only through an explicit CPU access."""
    path = destination / "algebra/cuda/cuda_types.h"
    text = path.read_text()
    marker = "struct QOCOMatrix_ {"
    if text.count(marker) != 1:
        raise RuntimeError("unexpected lazy matrix metadata site")
    path.write_text(text.replace(marker, "#define SPACEPDHCG_QOCO_LAZY_HOST_MIRRORS 1\n"
                                 + marker + "\n  int lazy_host_mirror;\n"
                                 "  mutable int host_values_pending;"))
    shutil.copyfile(extension, destination / "algebra/cuda/qoco_lazy_host_mirror.cuh")
    path = destination / "algebra/cuda/cuda_linalg.cu"
    text = path.read_text()
    replacements = {
        '#include "qoco_gpu_transpose.cuh"':
            '#include "qoco_lazy_host_mirror.cuh"\n#include "qoco_gpu_transpose.cuh"',
        "  M->gather = nullptr;":
            "  M->gather = nullptr;\n  M->lazy_host_mirror = 0;\n  M->host_values_pending = 0;",
        "    return M->csc;": "    qoco_materialize_host_mirror(M);\n    return M->csc;",
        "void sync_matrix_to_device(QOCOMatrix* M)\n{":
            "void sync_matrix_to_device(QOCOMatrix* M)\n{\n"
            "  if (qoco_device_mirror_current(M)) return;",
        "void sync_matrix_values_to_device(QOCOMatrix* M)\n{":
            "void sync_matrix_values_to_device(QOCOMatrix* M)\n{\n"
            "  if (qoco_device_mirror_current(M)) return;",
    }
    for before, after in replacements.items():
        if before.startswith("void sync_matrix_values_to_device") and before not in text:
            continue  # Present only with --values-only-updates.
        if text.count(before) != 1:
            raise RuntimeError(f"unexpected lazy mirror site: {before}")
        text = text.replace(before, after)
    path.write_text(text)
    path = destination / "src/qoco_api.c"
    text = path.read_text()
    if text.count("qoco_gpu_transpose(") != 3:
        raise RuntimeError("unexpected lazy transpose setup sites")
    path.write_text(text.replace("qoco_gpu_transpose(", "qoco_gpu_transpose_lazy("))
    path = destination / "algebra/cuda/qoco_device_update.cuh"
    if path.exists():
        text = path.read_text()
        marker = "    double result[9]{};"
        if text.count(marker) != 1:
            raise RuntimeError("unexpected device-update mirror invalidation site")
        path.write_text(text.replace(marker, marker + """
    // Device updates own these values. A later explicit CPU inspection must
    // refresh a previously materialized cache, including after a failed update.
    for (auto* matrix : {data->At, data->Gt})
        if (matrix && matrix->lazy_host_mirror) matrix->host_values_pending = 1;"""))


def patch_gpu_transposes(destination: Path, extension: Path) -> None:
    """Construct owned transpose matrices from existing GPU gather ordering."""
    shutil.copyfile(extension, destination / "algebra/cuda/qoco_gpu_transpose.cuh")
    path = destination / "algebra/cuda/cuda_linalg.cu"
    text = path.read_text()
    marker = '#include "qoco_gather.cuh"'
    if text.count(marker) != 1:
        raise RuntimeError("unexpected GPU transpose include site")
    path.write_text(text.replace(marker, marker + '\n#include "qoco_gpu_transpose.cuh"'))
    path = destination / "src/qoco_api.c"
    text = path.read_text()
    start = text.index("  // When creating transposed matrices,")
    end = text.index("  // Compute scaling statistics", start)
    profile = '  qoco_gpu_setup_mark("transposes");\n'
    profile = profile if profile in text[start:end] else ""
    text = text[:start] + """  data->At = qoco_gpu_transpose(data->A, data->AtoAt);
  data->Gt = qoco_gpu_transpose(data->G, data->GtoGt);

""" + profile + text[end:]
    text = text.replace('#include "backend.h"', '#include "backend.h"\n'
                        'extern QOCOMatrix* qoco_gpu_transpose(const QOCOMatrix*, QOCOInt*);')
    path.write_text(text)


def patch_restore_inaccurate_best(destination: Path) -> None:
    """Return the best qualified iterate when a solve exits inaccurately."""
    path = destination / "src/qoco_api.c"
    text = path.read_text()
    start = text.index("QOCOInt qoco_solve(QOCOSolver* solver)")
    end = text.index("\n}\n", start) + 3
    body = text[start:end]
    marker = "if (solver->sol->status == QOCO_NUMERICAL_ERROR) {"
    if body.count(marker) != 1:
        raise RuntimeError("unexpected inaccurate best-iterate restoration site")
    body = body.replace(marker, """if (solver->sol->status == QOCO_NUMERICAL_ERROR ||
          (solver->sol->status == QOCO_SOLVED_INACCURATE && work->best_valid &&
           work->best_metric <= 1.0)) {""")
    body = body.replace(
        "// On numerical error, restore the best iterate seen so far. The helper",
        "// On numerical error or a qualified inaccurate exit, restore the best iterate.\n"
        "      // The helper",
    )
    path.write_text(text[:start] + body + text[end:]
                    + "\nint qoco_restores_inaccurate_best(void) { return 1; }\n")


def patch_vector_arena(destination: Path, extension: Path) -> None:
    """Own post-analysis scratch vectors in one zeroed device allocation."""
    path = destination / "include/structs.h"
    text = path.read_text()
    marker = "} QOCOWorkspace;"
    if text.count(marker) != 1:
        raise RuntimeError("unexpected QOCO workspace declaration")
    path.write_text(text.replace(marker, "  void* gpu_vector_arena;\n" + marker))
    path = destination / "algebra/cuda/cuda_types.h"
    text = path.read_text()
    start = text.index("struct QOCOVectorf_ {")
    end = text.index("\n};", start)
    path.write_text(text[:end] + "\n  int arena_owned;" + text[end:])
    path = destination / "algebra/cuda/cuda_linalg.cu"
    text = path.read_text()
    start = text.index("QOCOVectorf* new_qoco_vectorf(")
    end = text.index("\n}\n", start)
    body = text[start:end].replace("  v->len = n;", "  v->arena_owned = 0;\n  v->len = n;")
    text = text[:start] + body + text[end:]
    start = text.index("void free_qoco_vectorf(")
    end = text.index("\n}\n", start)
    body = text[start:end].replace("if (x->d_data)", "if (x->d_data && !x->arena_owned)")
    path.write_text(text[:start] + body + text[end:] + '\n#include "qoco_vector_arena.cuh"\n')
    path = destination / "src/qoco_api.c"
    text = path.read_text()
    text = text.replace('#include "backend.h"', '''#include "backend.h"
extern void* qoco_gpu_vector_arena_create(int n, int m, int p, int wn, int nt);
extern QOCOVectorf* qoco_gpu_vector_arena_vector(void* arena, int length);
extern void qoco_gpu_vector_arena_finish(void* arena);
extern void qoco_gpu_vector_arena_destroy(void* arena);''')
    text = text.replace("  QOCOWorkspace* work = solver->work;",
                        "  QOCOWorkspace* work = solver->work;\n"
                        "  work->gpu_vector_arena = NULL;", 1)
    start = text.index("  // Allocate primal and dual variables.")
    end = text.index("  // Allocate solution struct.", start)
    body = text[start:end]
    body, count = re.subn(r"new_qoco_vectorf\(NULL, ([^)]+)\)",
                         r"qoco_gpu_vector_arena_vector(work->gpu_vector_arena, \1)", body)
    if count != 26:
        raise RuntimeError(f"unexpected arena vector count: {count}")
    body = ("  work->gpu_vector_arena = qoco_gpu_vector_arena_create(n, m, p, Wnnz, m + nsoc);\n"
            + body)
    body += "  qoco_gpu_vector_arena_finish(work->gpu_vector_arena);\n\n"
    text = text[:start] + body + text[end:]
    text = text.replace("  qoco_free(solver->work);",
                        "  qoco_gpu_vector_arena_destroy(solver->work->gpu_vector_arena);\n"
                        "  qoco_free(solver->work);")
    path.write_text(text)
    shutil.copyfile(extension, destination / "algebra/cuda/qoco_vector_arena.cuh")


def patch_gpu_kkt(destination: Path, extension: Path) -> None:
    """Replace host KKT/CSR assembly and update-map construction with CUDA."""
    path = destination / "algebra/cuda/cudss_backend.cu"
    text = path.read_text()
    start = text.index("  // Allocate memory for mappings to KKT matrix")
    end = text.index("  // Store CSR data array.", start)
    text = text[:start] + """  // Host mapping slots stay null; build GPU CSR maps directly.
  linsys_data->nt2kkt = nullptr;
  linsys_data->ntdiag2kkt = nullptr;
  linsys_data->PregtoKKT = nullptr;
  linsys_data->AttoKKT = nullptr;
  linsys_data->GttoKKT = nullptr;
  const int kkt_nnz = qoco_gpu_kkt::build(data, settings, linsys_data, Wnnz);
  QOCOInt* csr_row_ptr = linsys_data->d_csr_rows;
  QOCOInt* csr_col_ind = linsys_data->d_csr_columns;
  QOCOFloat* csr_val = linsys_data->d_csr_val;

""" + text[end:]
    text = text.replace("(int64_t)Kcsc->nnz, csr_row_ptr", "(int64_t)kkt_nnz, csr_row_ptr")
    start = text.index("  // CSR structure stays owned until vendor matrix destruction.")
    end = text.index("  return linsys_data;", start)
    text = text[:start] + text[end:]
    start = text.index("static void csc_to_csr_device(")
    end = text.index("\n}\n", start) + 3
    text = text[:start] + text[end:]
    marker = "static LinSysData* cudss_setup(QOCOProblemData* data, QOCOSettings* settings,"
    if text.count(marker) != 1:
        raise RuntimeError("unexpected GPU KKT extension site")
    path.write_text(text.replace(marker, '#include "qoco_gpu_kkt.cuh"\n\n' + marker))
    shutil.copyfile(extension, destination / "algebra/cuda/qoco_gpu_kkt.cuh")


def patch_ruiz_vector_sync(destination: Path) -> None:
    """Legacy host equilibration must publish scaled c/b/h as well as matrices."""
    path = destination / "src/equilibration.c"
    text = path.read_text()
    before = "  sync_vector_to_device(scaling->Finvruiz);"
    if text.count(before) != 1:
        raise RuntimeError("unexpected host Ruiz synchronization site")
    path.write_text(text.replace(before, before + """
  // CPU equilibration changes these vectors too. Publish them before solving
  // or refreshing the factorization, including zero-pass re-equilibration.
  sync_vector_to_device(data->c);
  sync_vector_to_device(data->b);
  sync_vector_to_device(data->h);"""))


def patch_device_io(destination: Path, extension: Path) -> None:
    """Negotiate device output and retain accepted primal starts on CUDA."""
    header = destination / "include/structs.h"
    text = header.read_text()
    before = "} QOCOWorkspace;"
    if text.count(before) != 1:
        raise RuntimeError("unexpected workspace extension site")
    header.write_text(text.replace(
        before, "  int gpu_device_io;\n  int gpu_primal_saved;\n" + before))
    path = destination / "src/qoco_api.c"
    text = path.read_text()
    before = "  QOCOWorkspace* work = solver->work;"
    if before not in text[:text.index("QOCOInt qoco_solve")]:
        raise RuntimeError("unexpected workspace initialization site")
    text = text.replace(before, before + "\n  work->gpu_device_io = 0;\n"
                        "  work->gpu_primal_saved = 0;", 1)
    before = "  copy_arrayf(x0, get_data_vectorf(work->x0), work->data->n);"
    if text.count(before) != 1:
        raise RuntimeError("unexpected host primal-start site")
    text = text.replace(before, "  work->gpu_primal_saved = 0;\n" + before)
    path.write_text(text)
    path = destination / "src/qoco_utils.c"
    text = path.read_text()
    before = "void copy_solution(QOCOSolver* solver)"
    if text.count(before) != 1:
        raise RuntimeError("unexpected solution export site")
    text = text.replace(before, "void qoco_reference_copy_solution(QOCOSolver* solver)")
    text += '''
int qoco_gpu_finish_device_solution(QOCOSolver* solver);
void copy_solution(QOCOSolver* solver) {
  const int device = qoco_gpu_finish_device_solution(solver);
  if (device < 0) { fprintf(stderr, "QOCO device solution completion failed\\n"); exit(1); }
  if (!device) qoco_reference_copy_solution(solver);
  else solver->sol->solve_time_sec = get_elapsed_time_sec(&(solver->work->solve_timer));
}
'''
    path.write_text(text)
    shutil.copyfile(extension, destination / "algebra/cuda/qoco_device_io.cuh")
    path = destination / "algebra/cuda/cuda_linalg.cu"
    text = path.read_text()
    if text.count('#include "qoco_device_solution.cuh"') != 1:
        raise RuntimeError("unexpected device extension include site")
    path.write_text(text.replace('#include "qoco_device_solution.cuh"',
                                '#include "qoco_device_solution.cuh"\n'
                                '#include "qoco_device_io.cuh"'))


def patch_solve_state(destination: Path) -> None:
    """Keep allocation reuse without retaining a previous problem's best iterate."""
    path = destination / "src/qoco_api.c"
    text = path.read_text()
    start = text.index("QOCOInt qoco_solve(QOCOSolver* solver)")
    before = "  start_timer(&(work->solve_timer));"
    body = text[start:]
    if body.count(before) != 1:
        raise RuntimeError("unexpected QOCO solve-state reset site")
    body = body.replace(
        before, """  // Best iterates are valid only for this solve's coefficients and scaling.
  // Keep device allocations and the explicitly requested primal warm start.
  work->best_valid = 0;
  work->best_iter = -1;
  work->best_metric = 0.0;
  work->ir_iters = 0;
  solver->sol->iters = 0;
  solver->sol->ir_iters = 0;
  solver->sol->status = QOCO_UNSOLVED;

""" + before)
    path.write_text(text[:start] + body)


def patch_setup_profile(destination: Path) -> None:
    """Insert optional completion-fenced stage markers into setup only."""
    sites = {
        "src/qoco_api.c": (
            ("  // Validate problem data.", "  qoco_gpu_setup_mark(NULL);\n", None),
            ("  // Equilibrate data.", None, "copy_input"),
            ("  // Compute scaling statistics before equilibration", None, "transposes"),
            ("  // Regularize P.", None, "ruiz"),
            ("  // Compute number of nonzeros in upper triangular NT", None, "regularize_P"),
            ("  // Set up linear system data.", None, "cone_indices"),
            ("  // Allocate primal and dual variables.", None, "linsys_return"),
            ("  stop_timer(&setup_timer);", None, "workspace_vectors"),
        ),
        "algebra/cuda/cudss_backend.cu": (
            ("  // Load CUDA libraries dynamically", None, "pre_vendor"),
            ("  // Allocate vector buffers", None, "vendor_handles"),
            ("  // Construct KKT matrix (no permutation", None, "vendor_buffers"),
            ("  // Convert KKT matrix from CSC", None, "assemble_KKT"),
            ("  // Build nt2kktcsr and ntdiag2kktcsr mappings", None, "convert_KKT"),
            ("  // Run analysis phase.", None, "create_csr"),
            ("  // Free CSR structure arrays", None, "symbolic_analysis"),
            ("  return linsys_data;", None, "vendor_finish"),
        ),
    }
    for relative, markers in sites.items():
        path = destination / relative
        text = path.read_text()
        start = text.index("QOCOInt qoco_setup(" if relative.endswith(".c")
                           else "static LinSysData* cudss_setup(")
        end = text.index("\n}\n", start) + 3
        body = text[start:end]
        for anchor, code, stage in markers:
            if anchor == "  // Free CSR structure arrays" and anchor not in body:
                anchor = "  // CSR structure stays owned until vendor matrix destruction."
            if body.count(anchor) != 1:
                raise RuntimeError(f"unexpected setup profile site: {anchor}")
            body = body.replace(anchor, (code or f'  qoco_gpu_setup_mark("{stage}");\n')
                                + anchor)
        declaration = 'extern "C" ' if relative.endswith(".cu") else ""
        if relative.endswith(".cu"):
            original = '''  CUDSS_CHECK(g_cuda_funcs.cudssExecute(
      linsys_data->handle, CUDSS_PHASE_ANALYSIS, linsys_data->config,
      linsys_data->data, linsys_data->K_csr, linsys_data->d_xyz_matrix,
      linsys_data->d_rhs_matrix));'''
            if body.count(original) != 1:
                raise RuntimeError("unexpected analysis phase for split profiling")
            body = body.replace(original, "  if (qoco_gpu_setup_profiling()) {\n"
                                + original.replace("CUDSS_PHASE_ANALYSIS", "CUDSS_PHASE_REORDERING")
                                + '\n    qoco_gpu_setup_mark("vendor_reordering");\n'
                                + original.replace("CUDSS_PHASE_ANALYSIS",
                                                   "CUDSS_PHASE_SYMBOLIC_FACTORIZATION")
                                + "\n  } else {\n" + original + "\n  }\n")
            declaration = 'extern "C" int qoco_gpu_setup_profiling();\n' + declaration
            stats = '''  if (qoco_gpu_setup_profiling()) {
    auto get_stats = reinterpret_cast<decltype(&::cudssDataGet)>(
        dlsym(g_cudss_handle, "cudssDataGet"));
    int64_t nnz = -1, flops = -1;
    size_t nnz_bytes = 0, flops_bytes = 0;
    if (get_stats) {
      const int nnz_status = get_stats(linsys_data->handle, linsys_data->data,
          CUDSS_DATA_LU_NNZ, &nnz, sizeof(nnz), &nnz_bytes);
      int flops_status = CUDSS_STATUS_NOT_SUPPORTED;
#if CUDSS_VERSION >= 800
      flops_status = get_stats(linsys_data->handle, linsys_data->data,
          CUDSS_DATA_FLOPS, &flops, sizeof(flops), &flops_bytes);
#endif
      fprintf(stderr, "{\\"case\\":\\"qoco_factor_statistics\\","
          "\\"nnz\\":%lld,\\"flops\\":%lld,\\"nnz_status\\":%d,\\"flops_status\\":%d,"
          "\\"nnz_bytes\\":%zu,\\"flops_bytes\\":%zu}\\n",
          (long long)nnz, (long long)flops, nnz_status, flops_status, nnz_bytes, flops_bytes);
    }
  }
'''
            body = body.replace('  qoco_gpu_setup_mark("symbolic_analysis");',
                                '  qoco_gpu_setup_mark("symbolic_analysis");\n' + stats)
        text = (text[:start] + declaration + "void qoco_gpu_setup_mark(const char*);\n\n"
                + body + text[end:])
        path.write_text(text)


def patch_gpu_ordering(destination: Path) -> None:
    """Supply an experimental GPU ordering instead of generic vendor reordering."""
    path = destination / "algebra/cuda/cudss_backend.cu"
    text = path.read_text()
    before = "static LinSysData* cudss_setup("
    text = text.replace(before, 'extern "C" void qoco_gpu_degree_ordering(int, '
                        'const int*, const int*, int*);\n\n' + before)
    before = "  // Run analysis phase."
    if text.count(before) != 1:
        raise RuntimeError("unexpected vendor analysis site")
    text = text.replace(before, '''  int* device_permutation = nullptr;
  CUDA_CHECK(cudaMalloc(&device_permutation, linsys_data->Kn * sizeof(int)));
  qoco_gpu_degree_ordering(linsys_data->Kn, csr_row_ptr, csr_col_ind, device_permutation);
  auto set_permutation = reinterpret_cast<decltype(&::cudssDataSet)>(
      dlsym(g_cudss_handle, "cudssDataSet"));
  if (!set_permutation) { fprintf(stderr, "cuDSS user permutation API unavailable\\n"); exit(1); }
  CUDSS_CHECK(set_permutation(linsys_data->handle, linsys_data->data,
      CUDSS_DATA_USER_PERM, device_permutation, linsys_data->Kn * sizeof(int)));

''' + before)
    before = "  // Free CSR structure arrays"
    text = text.replace(before, "  CUDA_CHECK(cudaFree(device_permutation));\n\n" + before)
    path.write_text(text)


def patch_setup_lifetimes(destination: Path) -> None:
    """Own vendor matrix inputs for their full lifetime and release host scratch."""
    path = destination / "algebra/cuda/cudss_backend.cu"
    text = path.read_text()
    before = "  QOCOFloat* d_csr_val;"
    if text.count(before) != 1:
        raise RuntimeError("unexpected CSR ownership field")
    text = text.replace(before, before + "\n  QOCOInt* d_csr_rows;\n  QOCOInt* d_csr_columns;")
    before = "  linsys_data->d_csr_val = csr_val;"
    text = text.replace(before, before + "\n  linsys_data->d_csr_rows = csr_row_ptr;\n"
                        "  linsys_data->d_csr_columns = csr_col_ind;")
    before = ("  // Free CSR structure arrays - cuDSS uses them during analysis\n"
              "  cudaFree(csr_row_ptr);\n  cudaFree(csr_col_ind);")
    if text.count(before) != 1:
        raise RuntimeError("unexpected CSR structure lifetime")
    text = text.replace(before, "  // CSR structure stays owned until vendor matrix destruction.\n"
                        "  free_qoco_csc_matrix(Kcsc);")
    start = text.index("  // Create dense matrix wrappers for solution and RHS vectors")
    end = text.index("\n  return linsys_data;", start)
    wrappers = text[start:end]
    text = text[:start] + text[end:]
    before = "  // Run analysis phase."
    if text.count(before) != 1:
        raise RuntimeError("unexpected analysis wrapper placement")
    text = text.replace(before, wrappers + "\n" + before)
    before = "  cudaFree(linsys_data->d_csr_val);"
    text = text.replace(before, before + "\n  cudaFree(linsys_data->d_csr_rows);\n"
                        "  cudaFree(linsys_data->d_csr_columns);")
    path.write_text(text)


def patch_trajectory_ordering(destination: Path, with_tree: bool = False) -> None:
    path = destination / "algebra/cuda/cudss_backend.cu"
    text = path.read_text()
    before = "static LinSysData* cudss_setup("
    text = text.replace(before, 'extern "C" int qoco_gpu_trajectory_ordering(int, int, '
                        'const int*, const int*, int*);\n\n' + before)
    before = "  // Run analysis phase."
    if text.count(before) != 1:
        raise RuntimeError("unexpected trajectory ordering site")
    text = text.replace(before, '''  int* trajectory_permutation = nullptr;
  CUDA_CHECK(cudaMalloc(&trajectory_permutation, linsys_data->Kn * sizeof(int)));
  const int trajectory_ordered = qoco_gpu_trajectory_ordering(linsys_data->Kn, data->n,
      csr_row_ptr, csr_col_ind, trajectory_permutation);
  if (trajectory_ordered < 0) {
    fprintf(stderr, "Invalid trajectory ordering metadata\\n"); exit(1);
  }
  if (trajectory_ordered > 0) {
    auto set_permutation = reinterpret_cast<decltype(&::cudssDataSet)>(
        dlsym(g_cudss_handle, "cudssDataSet"));
    if (!set_permutation) { fprintf(stderr, "cuDSS user permutation API unavailable\\n"); exit(1); }
    CUDSS_CHECK(set_permutation(linsys_data->handle, linsys_data->data,
        CUDSS_DATA_USER_PERM, trajectory_permutation, linsys_data->Kn * sizeof(int)));
  }

''' + before)
    before = "  // CSR structure stays owned until vendor matrix destruction."
    if text.count(before) != 1:
        raise RuntimeError("trajectory ordering requires corrected setup lifetimes")
    text = text.replace(before, "  CUDA_CHECK(cudaFree(trajectory_permutation));\n\n" + before)
    if with_tree:
        text = text.replace('extern "C" int qoco_gpu_trajectory_ordering(int, int, '
                            'const int*, const int*, int*);',
                            'extern "C" int qoco_gpu_trajectory_ordering_with_tree(int, int, '
                            'const int*, const int*, int*, int**, int*);')
        text = text.replace("  const int trajectory_ordered = qoco_gpu_trajectory_ordering(",
                            "  int* trajectory_tree = nullptr;\n  int trajectory_levels = 0;\n"
                            "  const int trajectory_ordered = "
                            "qoco_gpu_trajectory_ordering_with_tree(")
        text = text.replace("      csr_row_ptr, csr_col_ind, trajectory_permutation);",
                            "      csr_row_ptr, csr_col_ind, trajectory_permutation,\n"
                            "      &trajectory_tree, &trajectory_levels);")
        before = ("        CUDSS_DATA_USER_PERM, trajectory_permutation, "
                  "linsys_data->Kn * sizeof(int)));")
        if text.count(before) != 1:
            raise RuntimeError("unexpected trajectory permutation submission")
        text = text.replace(before, before + '''
    CUDSS_CHECK(g_cuda_funcs.cudssConfigSet(linsys_data->config,
        CUDSS_CONFIG_ND_NLEVELS, &trajectory_levels, sizeof(trajectory_levels)));
    CUDSS_CHECK(set_permutation(linsys_data->handle, linsys_data->data,
#if CUDSS_VERSION >= 800
        CUDSS_DATA_USER_ND_PARTITION_TREE,
#else
        CUDSS_DATA_USER_ELIMINATION_TREE,
#endif
        trajectory_tree,
        ((1 << trajectory_levels) - 1) * sizeof(int)));''')
        text = text.replace("  CUDA_CHECK(cudaFree(trajectory_permutation));",
                            "  CUDA_CHECK(cudaFree(trajectory_permutation));\n"
                            "  CUDA_CHECK(cudaFree(trajectory_tree));")
    path.write_text(text)


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
    parser.add_argument("--multiblock-factorization", action="store_true",
                        help="experimental cuDSS multiblock factorization algorithm")
    parser.add_argument("--superpanels", action="store_true",
                        help="experimental cuDSS superpanel optimization")
    parser.add_argument("--reset-solve-state", action="store_true",
                        help="reset per-solve history (already enabled for all patched builds)")
    parser.add_argument("--device-io", action="store_true",
                        help="offer opt-in device solution output and GPU primal warm starts")
    parser.add_argument("--gpu-kkt", action="store_true",
                        help="assemble KKT CSR and numerical-update maps on CUDA")
    parser.add_argument("--gpu-transposes", action="store_true",
                        help="construct constraint transposes and their update maps on CUDA")
    parser.add_argument("--lazy-transpose-mirrors", action="store_true",
                        help="defer transpose host copies and skip unchanged mirror uploads")
    parser.add_argument("--deferred-transposes", action="store_true",
                        help="materialize compatibility transposes only on explicit access")
    parser.add_argument("--vector-arena", action="store_true",
                        help="experimental GPU scratch-vector arena (not qualified for promotion)")
    parser.add_argument("--restore-inaccurate-best", action="store_true",
                        help="return the saved best qualified iterate on inaccurate exits")
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
    parser.add_argument("--profile-setup", action="store_true",
                        help="expose optional completion-fenced setup stage diagnostics")
    parser.add_argument("--gpu-degree-ordering", action="store_true",
                        help="experimental GPU static-degree KKT ordering (not AMD)")
    parser.add_argument("--setup-lifetimes", action="store_true",
                        help="fix KKT scratch ownership and vendor matrix lifetimes")
    parser.add_argument("--trajectory-ordering", action="store_true",
                        help="experimental GPU separators from explicit trajectory index maps")
    parser.add_argument("--trajectory-tree", action="store_true",
                        help="submit GPU separator sizes with the permutation (cuDSS 0.7.1+)")
    args = parser.parse_args()
    if args.unmodified and (
        args.gather
        or args.correct_stopping
        or args.deterministic
        or args.multiblock_factorization
        or args.superpanels
        or args.reset_solve_state
        or args.device_io
        or args.gpu_kkt
        or args.gpu_transposes
        or args.lazy_transpose_mirrors
        or args.deferred_transposes
        or args.vector_arena
        or args.restore_inaccurate_best
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
        or args.profile_setup
        or args.gpu_degree_ordering
        or args.setup_lifetimes
        or args.trajectory_ordering
        or args.trajectory_tree
    ):
        parser.error("--unmodified cannot be combined with backend changes")
    if (args.queued_operators or args.device_cone_reductions) and args.original_handles:
        parser.error("queued operators and device cone reductions require scoped handles")
    if args.queued_operators and not args.gather:
        parser.error("--queued-operators requires --gather")
    if args.device_numeric_updates and not args.gather:
        parser.error("--device-numeric-updates requires --gather")
    if args.gpu_transposes and not args.gather:
        parser.error("--gpu-transposes requires --gather")
    if args.lazy_transpose_mirrors and not args.gpu_transposes:
        parser.error("--lazy-transpose-mirrors requires --gpu-transposes")
    if args.deferred_transposes and not (args.lazy_transpose_mirrors and args.gpu_kkt):
        parser.error("--deferred-transposes requires --lazy-transpose-mirrors and --gpu-kkt")
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
    if args.gpu_degree_ordering and not args.checked_cudss_abi:
        parser.error("--gpu-degree-ordering requires --checked-cudss-abi")
    if args.profile_setup and not args.checked_cudss_abi:
        parser.error("--profile-setup requires --checked-cudss-abi")
    if args.trajectory_ordering and (not args.checked_cudss_abi or not args.setup_lifetimes):
        parser.error("--trajectory-ordering requires --checked-cudss-abi and --setup-lifetimes")
    if args.trajectory_ordering and args.gpu_degree_ordering:
        parser.error("choose only one ordering strategy")
    if args.trajectory_tree and not args.trajectory_ordering:
        parser.error("--trajectory-tree requires --trajectory-ordering")
    if args.multiblock_factorization and not args.checked_cudss_abi:
        parser.error("--multiblock-factorization requires --checked-cudss-abi")
    if args.superpanels and not args.checked_cudss_abi:
        parser.error("--superpanels requires --checked-cudss-abi")
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
    if args.profile_setup:
        shutil.copyfile(extension.with_name("qoco_setup_profile.cuh"),
                        destination / "algebra/cuda/qoco_setup_profile.cuh")
        modified += '\n#include "qoco_setup_profile.cuh"\n'
    if args.gpu_degree_ordering:
        shutil.copyfile(extension.with_name("qoco_gpu_ordering.cuh"),
                        destination / "algebra/cuda/qoco_gpu_ordering.cuh")
        modified += '\n#include "qoco_gpu_ordering.cuh"\n'
    if args.trajectory_ordering:
        shutil.copyfile(extension.with_name("qoco_trajectory_ordering.cuh"),
                        destination / "algebra/cuda/qoco_trajectory_ordering.cuh")
        modified += '\n#include "qoco_trajectory_ordering.cuh"\n'
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
    # --unmodified returned above. Every patched build must isolate recovery
    # history, independent of which performance experiments are selected.
    patch_solve_state(destination)
    patch_ruiz_vector_sync(destination)
    if args.device_io:
        patch_device_io(destination, extension.with_name("qoco_device_io.cuh"))
    if args.superpanels:
        backend_path = destination / "algebra/cuda/cudss_backend.cu"
        backend = backend_path.read_text()
        before = "  int value = 0;\n  CUDSS_CHECK(g_cuda_funcs.cudssConfigSet(linsys_data->config,"
        if backend.count(before) != 1:
            raise RuntimeError("unexpected cuDSS superpanel configuration site")
        backend_path.write_text(backend.replace(before, before.replace("value = 0", "value = 1")))
    if args.multiblock_factorization:
        backend_path = destination / "algebra/cuda/cudss_backend.cu"
        backend = backend_path.read_text()
        before = "  // Initialize cuSPARSE"
        if backend.count(before) != 1:
            raise RuntimeError("unexpected cuDSS factorization configuration site")
        backend_path.write_text(backend.replace(before, '''#if CUDSS_VERSION >= 800
  cudssFactorizationAlg_t factor_algorithm = CUDSS_FACTORIZATION_ALG_MULTIBLOCK;
#else
  cudssAlgType_t factor_algorithm = CUDSS_ALG_1;
#endif
  CUDSS_CHECK(g_cuda_funcs.cudssConfigSet(linsys_data->config,
      CUDSS_CONFIG_FACTORIZATION_ALG, &factor_algorithm, sizeof(factor_algorithm)));
''' + before))
    if args.batched_stopping:
        patch_batched_stopping(destination, args.batched_iteration_scalars)
    if args.gpu_degree_ordering:
        patch_gpu_ordering(destination)
    if args.setup_lifetimes:
        patch_setup_lifetimes(destination)
    if args.trajectory_ordering:
        patch_trajectory_ordering(destination, args.trajectory_tree)
    if args.profile_setup:
        patch_setup_profile(destination)
    if args.gpu_kkt:
        if not args.setup_lifetimes or args.profile_setup:
            raise RuntimeError(
                "GPU KKT requires setup lifetimes and excludes legacy setup profiling")
        patch_gpu_kkt(destination, extension.with_name("qoco_gpu_kkt.cuh"))
    if args.vector_arena:
        patch_vector_arena(destination, extension.with_name("qoco_vector_arena.cuh"))
    if args.restore_inaccurate_best:
        patch_restore_inaccurate_best(destination)
    if args.gpu_transposes:
        patch_gpu_transposes(destination, extension.with_name("qoco_gpu_transpose.cuh"))
    if args.lazy_transpose_mirrors:
        patch_lazy_transpose_mirrors(destination, extension.with_name("qoco_lazy_host_mirror.cuh"))
    if args.deferred_transposes:
        patch_deferred_transposes(destination, extension.with_name("qoco_deferred_transpose.cuh"))
    provenance = {
        "upstream_commit": commit,
        "source": str(source),
        "original_cuda_linalg_sha256": hashlib.sha256(original.encode()).hexdigest(),
        "modified_cuda_linalg_sha256": hashlib.sha256(
            (destination / "algebra/cuda/cuda_linalg.cu").read_bytes()).hexdigest(),
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
        "profile_setup": args.profile_setup,
        "gpu_degree_ordering": args.gpu_degree_ordering,
        "setup_lifetimes": args.setup_lifetimes,
        "trajectory_ordering": args.trajectory_ordering,
        "trajectory_tree": args.trajectory_tree,
        "correct_stopping": args.correct_stopping,
        "deterministic": args.deterministic,
        "multiblock_factorization": args.multiblock_factorization,
        "superpanels": args.superpanels,
        "reset_solve_state": True,
        "device_io": args.device_io,
        "gpu_kkt": args.gpu_kkt,
        "gpu_transposes": args.gpu_transposes,
        "lazy_transpose_mirrors": args.lazy_transpose_mirrors,
        "deferred_transposes": args.deferred_transposes,
        "vector_arena": args.vector_arena,
        "restore_inaccurate_best": args.restore_inaccurate_best,
        "host_ruiz_vector_sync": True,
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
            "src/qoco_api.c",
            "src/equilibration.c",
            *(["include/structs.h", "algebra/cuda/qoco_device_io.cuh"]
              if args.device_io else []),
            *(["algebra/cuda/qoco_gpu_kkt.cuh"] if args.gpu_kkt else []),
            *(["algebra/cuda/qoco_gpu_transpose.cuh"] if args.gpu_transposes else []),
            *(["algebra/cuda/qoco_lazy_host_mirror.cuh"] if args.lazy_transpose_mirrors else []),
            *(["algebra/cuda/qoco_deferred_transpose.cuh"] if args.deferred_transposes else []),
            *(["include/structs.h", "algebra/cuda/qoco_vector_arena.cuh"]
              if args.vector_arena else []),
            *(["algebra/cuda/qoco_gather.cuh"] if args.gather else []),
            *(["src/cone.cu", "src/qoco_device_cone_reductions.cuh"]
              if args.device_cone_reductions else []),
            *(["include/qoco_linalg.h"] if args.values_only_updates else []),
            *(["algebra/cuda/qoco_device_update.cuh"] if args.device_numeric_updates else []),
            *(["algebra/cuda/qoco_device_scalar.cuh"] if args.device_scalar_reductions else []),
            *(["algebra/cuda/qoco_batched_stopping.cuh"] if args.batched_stopping else []),
            *(["src/kkt.c", "src/qoco_device_step_cones.cuh",
               "algebra/cuda/qoco_device_step_control.cuh"] if args.device_step_control else []),
            *(["src/qoco_device_combined_cones.cuh", "algebra/cuda/qoco_device_combined_rhs.cuh"]
              if args.device_combined_rhs else []),
            *(["algebra/cuda/qoco_setup_profile.cuh"] if args.profile_setup else []),
            *(["algebra/cuda/qoco_gpu_ordering.cuh"] if args.gpu_degree_ordering else []),
            *(["algebra/cuda/qoco_trajectory_ordering.cuh"] if args.trajectory_ordering else []),
        )
    }
    (destination / "spacepdhcg-provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
