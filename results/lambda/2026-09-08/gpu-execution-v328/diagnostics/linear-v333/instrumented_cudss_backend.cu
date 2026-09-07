/**
 * @file cudss_backend.cu
 * @author Govind M. Chari <govindchari1@gmail.com>
 *
 * @section LICENSE
 *
 * Copyright (c) 2025, Govind M. Chari
 * This source code is licensed under the BSD 3-Clause License
 */

#include "cudss_backend.h"
#include <dlfcn.h>
#include <stdlib.h>

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,         \
              cudaGetErrorString(err));                                        \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

#define CUDSS_CHECK(call)                                                      \
  do {                                                                         \
    cudssStatus_t err = call;                                                  \
    if (err != CUDSS_STATUS_SUCCESS) {                                         \
      const char* err_str =                                                    \
          (err == CUDSS_STATUS_NOT_INITIALIZED)    ? "NOT_INITIALIZED"         \
          : (err == CUDSS_STATUS_ALLOC_FAILED)     ? "ALLOC_FAILED"            \
          : (err == CUDSS_STATUS_INVALID_VALUE)    ? "INVALID_VALUE"           \
          : (err == CUDSS_STATUS_NOT_SUPPORTED)    ? "NOT_SUPPORTED"           \
          : (err == CUDSS_STATUS_EXECUTION_FAILED) ? "EXECUTION_FAILED"        \
          : (err == CUDSS_STATUS_INTERNAL_ERROR)   ? "INTERNAL_ERROR"          \
                                                   : "UNKNOWN";                  \
      fprintf(stderr, "cuDSS error at %s:%d: status %d (%s)\n", __FILE__,      \
              __LINE__, (int)err, err_str);                                    \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

// Global function pointer structure
static CudaLibFuncs g_cuda_funcs = {0};
static void* g_cudss_handle = NULL;
static void* g_cusparse_handle = NULL;
static void* g_cublas_handle = NULL;
static int g_libs_loaded = 0;

// Global accessor for function pointers (for use in cuda_linalg.cu)
CudaLibFuncs* get_cuda_funcs(void) { return &g_cuda_funcs; }

// Load CUDA libraries using dlopen
int load_cuda_libraries(void)
{
  if (g_libs_loaded) {
    return 1; // Already loaded
  }

  // Load cuDSS
  g_cudss_handle = dlopen("libcudss.so", RTLD_LAZY);
  if (!g_cudss_handle) {
    g_cudss_handle = dlopen("libcudss.so.1", RTLD_LAZY);
  }
  if (!g_cudss_handle) {
    fprintf(stderr, "Failed to load cuDSS: %s\n", dlerror());
    return 0;
  }

  // The CSR signature changes in cuDSS 0.8. Never call a mismatched ABI.
  auto property = reinterpret_cast<decltype(&::cudssGetProperty)>(
      dlsym(g_cudss_handle, "cudssGetProperty"));
  int major = -1, minor = -1;
  if (!property || property(MAJOR_VERSION, &major) != CUDSS_STATUS_SUCCESS
      || property(MINOR_VERSION, &minor) != CUDSS_STATUS_SUCCESS
      || major != CUDSS_VERSION_MAJOR || minor != CUDSS_VERSION_MINOR) {
    fprintf(stderr, "QOCO cuDSS header/runtime ABI mismatch (%d.%d vs %d.%d)\n",
            CUDSS_VERSION_MAJOR, CUDSS_VERSION_MINOR, major, minor);
    dlclose(g_cudss_handle);
    g_cudss_handle = nullptr;
    return 0;
  }

  // Load cuSPARSE
  g_cusparse_handle = dlopen("libcusparse.so", RTLD_LAZY);
  if (!g_cusparse_handle) {
    g_cusparse_handle = dlopen("libcusparse.so.11", RTLD_LAZY);
  }
  if (!g_cusparse_handle) {
    g_cusparse_handle = dlopen("libcusparse.so.12", RTLD_LAZY);
  }
  if (!g_cusparse_handle) {
    fprintf(stderr, "Failed to load cuSPARSE: %s\n", dlerror());
    dlclose(g_cudss_handle);
    return 0;
  }

  // Load cuBLAS
  g_cublas_handle = dlopen("libcublas.so", RTLD_LAZY);
  if (!g_cublas_handle) {
    g_cublas_handle = dlopen("libcublas.so.11", RTLD_LAZY);
  }
  if (!g_cublas_handle) {
    g_cublas_handle = dlopen("libcublas.so.12", RTLD_LAZY);
  }
  if (!g_cublas_handle) {
    fprintf(stderr, "Failed to load cuBLAS: %s\n", dlerror());
    dlclose(g_cudss_handle);
    dlclose(g_cusparse_handle);
    return 0;
  }

  // Load cuDSS functions
  g_cuda_funcs.cudssCreate =
      (typeof(g_cuda_funcs.cudssCreate))dlsym(g_cudss_handle, "cudssCreate");
  g_cuda_funcs.cudssConfigCreate =
      (typeof(g_cuda_funcs.cudssConfigCreate))dlsym(g_cudss_handle,
                                                    "cudssConfigCreate");
  g_cuda_funcs.cudssDataCreate = (typeof(g_cuda_funcs.cudssDataCreate))dlsym(
      g_cudss_handle, "cudssDataCreate");
  g_cuda_funcs.cudssConfigSet = (typeof(g_cuda_funcs.cudssConfigSet))dlsym(
      g_cudss_handle, "cudssConfigSet");
  g_cuda_funcs.cudssMatrixCreateCsr =
      (typeof(g_cuda_funcs.cudssMatrixCreateCsr))dlsym(g_cudss_handle,
                                                       "cudssMatrixCreateCsr");
  g_cuda_funcs.cudssExecute =
      (typeof(g_cuda_funcs.cudssExecute))dlsym(g_cudss_handle, "cudssExecute");
  g_cuda_funcs.cudssMatrixCreateDn =
      (typeof(g_cuda_funcs.cudssMatrixCreateDn))dlsym(g_cudss_handle,
                                                      "cudssMatrixCreateDn");
  g_cuda_funcs.cudssMatrixSetValues =
      (typeof(g_cuda_funcs.cudssMatrixSetValues))dlsym(g_cudss_handle,
                                                       "cudssMatrixSetValues");
  g_cuda_funcs.cudssMatrixDestroy =
      (typeof(g_cuda_funcs.cudssMatrixDestroy))dlsym(g_cudss_handle,
                                                     "cudssMatrixDestroy");
  g_cuda_funcs.cudssDataDestroy = (typeof(g_cuda_funcs.cudssDataDestroy))dlsym(
      g_cudss_handle, "cudssDataDestroy");
  g_cuda_funcs.cudssConfigDestroy =
      (typeof(g_cuda_funcs.cudssConfigDestroy))dlsym(g_cudss_handle,
                                                     "cudssConfigDestroy");
  g_cuda_funcs.cudssDestroy =
      (typeof(g_cuda_funcs.cudssDestroy))dlsym(g_cudss_handle, "cudssDestroy");

  if (!g_cuda_funcs.cudssCreate || !g_cuda_funcs.cudssConfigCreate ||
      !g_cuda_funcs.cudssDataCreate || !g_cuda_funcs.cudssMatrixCreateCsr ||
      !g_cuda_funcs.cudssExecute || !g_cuda_funcs.cudssMatrixCreateDn ||
      !g_cuda_funcs.cudssMatrixSetValues || !g_cuda_funcs.cudssMatrixDestroy ||
      !g_cuda_funcs.cudssDataDestroy || !g_cuda_funcs.cudssConfigDestroy ||
      !g_cuda_funcs.cudssDestroy || !g_cuda_funcs.cudssConfigSet) {
    fprintf(stderr, "Failed to resolve cuDSS symbols: %s\n", dlerror());
    dlclose(g_cudss_handle);
    dlclose(g_cusparse_handle);
    return 0;
  }

  // Load cuSPARSE functions
  g_cuda_funcs.cusparseCreate = (typeof(g_cuda_funcs.cusparseCreate))dlsym(
      g_cusparse_handle, "cusparseCreate");
  g_cuda_funcs.cusparseCreateMatDescr =
      (typeof(g_cuda_funcs.cusparseCreateMatDescr))dlsym(
          g_cusparse_handle, "cusparseCreateMatDescr");
  g_cuda_funcs.cusparseSetMatType =
      (typeof(g_cuda_funcs.cusparseSetMatType))dlsym(g_cusparse_handle,
                                                     "cusparseSetMatType");
  g_cuda_funcs.cusparseSetMatIndexBase =
      (typeof(g_cuda_funcs.cusparseSetMatIndexBase))dlsym(
          g_cusparse_handle, "cusparseSetMatIndexBase");
  g_cuda_funcs.cusparseDestroy = (typeof(g_cuda_funcs.cusparseDestroy))dlsym(
      g_cusparse_handle, "cusparseDestroy");
  g_cuda_funcs.cusparseDestroyMatDescr =
      (typeof(g_cuda_funcs.cusparseDestroyMatDescr))dlsym(
          g_cusparse_handle, "cusparseDestroyMatDescr");

  if (!g_cuda_funcs.cusparseCreate || !g_cuda_funcs.cusparseCreateMatDescr ||
      !g_cuda_funcs.cusparseSetMatType ||
      !g_cuda_funcs.cusparseSetMatIndexBase || !g_cuda_funcs.cusparseDestroy ||
      !g_cuda_funcs.cusparseDestroyMatDescr) {
    fprintf(stderr, "Failed to resolve cuSPARSE symbols: %s\n", dlerror());
    dlclose(g_cudss_handle);
    dlclose(g_cusparse_handle);
    dlclose(g_cublas_handle);
    return 0;
  }

  g_cuda_funcs.cublasSetWorkspace = reinterpret_cast<decltype(&::cublasSetWorkspace)>(
      dlsym(g_cublas_handle, "cublasSetWorkspace_v2"));
  if (!g_cuda_funcs.cublasSetWorkspace) return 0;
  g_cuda_funcs.cublasSetStream = reinterpret_cast<decltype(&::cublasSetStream)>(
      dlsym(g_cublas_handle, "cublasSetStream_v2"));
  g_cuda_funcs.cublasGetStream = reinterpret_cast<decltype(&::cublasGetStream)>(
      dlsym(g_cublas_handle, "cublasGetStream_v2"));
  if (!g_cuda_funcs.cublasSetStream || !g_cuda_funcs.cublasGetStream) {
    dlclose(g_cublas_handle); dlclose(g_cusparse_handle);
    dlclose(g_cudss_handle); return 0;
  }
  g_cuda_funcs.cublasSetPointerMode =
      reinterpret_cast<decltype(&::cublasSetPointerMode)>(
          dlsym(g_cublas_handle, "cublasSetPointerMode_v2"));
  g_cuda_funcs.cublasGetPointerMode =
      reinterpret_cast<decltype(&::cublasGetPointerMode)>(
          dlsym(g_cublas_handle, "cublasGetPointerMode_v2"));
  if (!g_cuda_funcs.cublasSetPointerMode || !g_cuda_funcs.cublasGetPointerMode) {
    fprintf(stderr, "QOCO cuBLAS pointer-mode symbols missing\n");
    dlclose(g_cublas_handle); dlclose(g_cusparse_handle); dlclose(g_cudss_handle);
    return 0;
  }
  // Load cuBLAS functions
  g_cuda_funcs.cublasCreate = (typeof(g_cuda_funcs.cublasCreate))dlsym(
      g_cublas_handle, "cublasCreate_v2");
  if (!g_cuda_funcs.cublasCreate) {
    g_cuda_funcs.cublasCreate = (typeof(g_cuda_funcs.cublasCreate))dlsym(
        g_cublas_handle, "cublasCreate");
  }
  g_cuda_funcs.cublasDdot =
      (typeof(g_cuda_funcs.cublasDdot))dlsym(g_cublas_handle, "cublasDdot_v2");
  if (!g_cuda_funcs.cublasDdot) {
    g_cuda_funcs.cublasDdot =
        (typeof(g_cuda_funcs.cublasDdot))dlsym(g_cublas_handle, "cublasDdot");
  }
  g_cuda_funcs.cublasDestroy = (typeof(g_cuda_funcs.cublasDestroy))dlsym(
      g_cublas_handle, "cublasDestroy_v2");
  if (!g_cuda_funcs.cublasDestroy) {
    g_cuda_funcs.cublasDestroy = (typeof(g_cuda_funcs.cublasDestroy))dlsym(
        g_cublas_handle, "cublasDestroy");
  }
  g_cuda_funcs.cublasIdamin = (typeof(g_cuda_funcs.cublasIdamin))dlsym(
      g_cublas_handle, "cublasIdamin_v2");
  if (!g_cuda_funcs.cublasIdamin) {
    g_cuda_funcs.cublasIdamin = (typeof(g_cuda_funcs.cublasIdamin))dlsym(
        g_cublas_handle, "cublasIdamin");
  }
  g_cuda_funcs.cublasIdamax = (typeof(g_cuda_funcs.cublasIdamax))dlsym(
      g_cublas_handle, "cublasIdamax_v2");
  if (!g_cuda_funcs.cublasIdamax) {
    g_cuda_funcs.cublasIdamax = (typeof(g_cuda_funcs.cublasIdamax))dlsym(
        g_cublas_handle, "cublasIdamax");
  }

  if (!g_cuda_funcs.cublasCreate || !g_cuda_funcs.cublasDdot ||
      !g_cuda_funcs.cublasDestroy || !g_cuda_funcs.cublasIdamin ||
      !g_cuda_funcs.cublasIdamax) {
    fprintf(stderr, "Failed to resolve cuBLAS symbols: %s\n", dlerror());
    dlclose(g_cudss_handle);
    dlclose(g_cusparse_handle);
    dlclose(g_cublas_handle);
    return 0;
  }

  g_libs_loaded = 1;
  return 1;
}

// Contains data for linear system.
#include "qoco_ir_runtime.cuh"

struct LinSysData {
  QocoIrRuntime* ir;

  /** KKT matrix in CSR form (device) for cuDSS. */
  cudssMatrix_t K_csr;

  /** Number of rows/columns of KKT matrix. */
  QOCOInt Kn;

  /** cuDSS handle. */
  cudssHandle_t handle;

  /** cuDSS config. */
  cudssConfig_t config;

  /** cuDSS data. */
  cudssData_t data;

  /** Buffer of size n + m + p (device) - used for b and x vectors. */
  QOCOFloat* d_rhs_matrix_data;

  /** Buffer of size n + m + p (device). */
  QOCOFloat* d_xyz_matrix_data;

  /** Mapping from elements in the Nesterov-Todd scaling matrix to elements in
   * the KKT matrix. */
  QOCOInt* nt2kkt;

  /** Mapping from elements on the main diagonal of the Nesterov-Todd scaling
   * matrices to elements in the KKT matrix. Used for regularization.*/
  QOCOInt* ntdiag2kkt;

  /** Mapping from elements in regularized P to elements in the KKT matrix. */
  QOCOInt* PregtoKKT;

  /** Mapping from elements in At to elements in the KKT matrix. */
  QOCOInt* AttoKKT;

  /** Mapping from elements in Gt to elements in the KKT matrix. */
  QOCOInt* GttoKKT;

  QOCOInt Wnnz;

  /** Number of constraints (m) - stored for use in factor */
  QOCOInt m;

  /** Static regularization for the (1,1) P block. */
  QOCOFloat kkt_static_reg_P;

  /** Static regularization for the (2,2) A block. */
  QOCOFloat kkt_static_reg_A;

  /** Static regularization for the (3,3) G block. */
  QOCOFloat kkt_static_reg_G;

  /** cuSPARSE handle. */
  cusparseHandle_t cusparse_handle;

  /** Matrix description. */
  cusparseMatDescr_t descr;

  /** cuDSS dense matrix wrappers for solution and RHS vectors. */
  cudssMatrix_t d_rhs_matrix;
  cudssMatrix_t d_xyz_matrix;

  /** CSR data array (used for updating NT block and cudss_update_data) */
  QOCOFloat* d_csr_val;
  QOCOInt* d_csr_rows;
  QOCOInt* d_csr_columns;

  /** Mapping from NT block indices to CSR KKT matrix indices (device) */
  QOCOInt* d_nt2kktcsr;
  /** Mapping from NT diagonal indices to CSR KKT matrix indices (device) */
  QOCOInt* d_ntdiag2kktcsr;

  /** Device buffer for WtW values (persistent, reused across iterations) */
  QOCOFloat* d_WtW;

  /** Mapping from P, A, G indices to CSR KKT matrix indices (device) */
  QOCOInt* d_PregtoKKTcsr;
  QOCOInt* d_AttoKKTcsr;
  QOCOInt* d_GttoKKTcsr;
};

// Convert CSC to CSR on CPU and copy to GPU

extern "C" int qoco_gpu_trajectory_ordering_with_tree(int, int, const int*, const int*, int*, int**, int*);

#include "qoco_gpu_kkt.cuh"

static LinSysData* cudss_setup(QOCOProblemData* data, QOCOSettings* settings,
                               QOCOInt Wnnz, QOCOFloat* analysis_time_sec)
{
  // Load CUDA libraries dynamically
  if (!load_cuda_libraries()) {
    fprintf(stderr, "Failed to load CUDA libraries\n");
    return NULL;
  }

  LinSysData* linsys_data = (LinSysData*)qoco_malloc(sizeof(LinSysData));

  linsys_data->Kn = data->n + data->m + data->p;

  // Initialize cuDSS
  CUDSS_CHECK(g_cuda_funcs.cudssCreate(&linsys_data->handle));
  linsys_data->ir = qoco_ir_create(linsys_data->handle, linsys_data->Kn);
  CUDSS_CHECK(g_cuda_funcs.cudssConfigCreate(&linsys_data->config));
  CUDSS_CHECK(
      g_cuda_funcs.cudssDataCreate(linsys_data->handle, &linsys_data->data));
  int value = 0;
  CUDSS_CHECK(g_cuda_funcs.cudssConfigSet(linsys_data->config,
                                          CUDSS_CONFIG_USE_SUPERPANELS,
                                          (void*)&value, sizeof(int)));

  // Initialize cuSPARSE
  g_cuda_funcs.cusparseCreate(&linsys_data->cusparse_handle);
  g_cuda_funcs.cusparseCreateMatDescr(&linsys_data->descr);
  g_cuda_funcs.cusparseSetMatType(linsys_data->descr,
                                  CUSPARSE_MATRIX_TYPE_GENERAL);
  g_cuda_funcs.cusparseSetMatIndexBase(linsys_data->descr,
                                       CUSPARSE_INDEX_BASE_ZERO);

  // Allocate vector buffers
  CUDA_CHECK(cudaMalloc(&linsys_data->d_rhs_matrix_data,
                        sizeof(QOCOFloat) * linsys_data->Kn));
  CUDA_CHECK(cudaMalloc(&linsys_data->d_xyz_matrix_data,
                        sizeof(QOCOFloat) * linsys_data->Kn));
  linsys_data->Wnnz = Wnnz;
  linsys_data->kkt_static_reg_P = settings->kkt_static_reg_P;
  linsys_data->kkt_static_reg_A = settings->kkt_static_reg_A;
  linsys_data->kkt_static_reg_G = settings->kkt_static_reg_G;

  // Host mapping slots stay null; build GPU CSR maps directly.
  linsys_data->nt2kkt = nullptr;
  linsys_data->ntdiag2kkt = nullptr;
  linsys_data->PregtoKKT = nullptr;
  linsys_data->AttoKKT = nullptr;
  linsys_data->GttoKKT = nullptr;
  const int kkt_nnz = qoco_gpu_kkt::build(data, settings, linsys_data, Wnnz);
  QOCOInt* csr_row_ptr = linsys_data->d_csr_rows;
  QOCOInt* csr_col_ind = linsys_data->d_csr_columns;
  QOCOFloat* csr_val = linsys_data->d_csr_val;

  // Store CSR data array.
  linsys_data->d_csr_val = csr_val;
  linsys_data->d_csr_rows = csr_row_ptr;
  linsys_data->d_csr_columns = csr_col_ind;

  // Determine data types
#if CUDSS_VERSION >= 800
  cudssDataType_t indexType = CUDSS_R_32I;
  cudssDataType_t valueType_setup =
      (sizeof(QOCOFloat) == 8) ? CUDSS_R_64F : CUDSS_R_32F;
#else
  cudaDataType_t indexType = CUDA_R_32I; // QOCOInt is int32_t
  cudaDataType_t valueType_setup =
      (sizeof(QOCOFloat) == 8) ? CUDA_R_64F : CUDA_R_32F;
#endif

  // KKT matrix is symmetric (upper triangular stored)
  CUDSS_CHECK(g_cuda_funcs.cudssMatrixCreateCsr(
      &linsys_data->K_csr, (int64_t)linsys_data->Kn, (int64_t)linsys_data->Kn,
      (int64_t)kkt_nnz, csr_row_ptr, NULL, csr_col_ind, csr_val,
#if CUDSS_VERSION >= 800
      indexType,
#endif
      indexType, valueType_setup, CUDSS_MTYPE_SYMMETRIC, CUDSS_MVIEW_UPPER,
      CUDSS_BASE_ZERO));

  // Create dense matrix wrappers for solution and RHS vectors (column vectors)
  // Note: d_rhs_matrix wraps d_rhs_matrix_data, d_xyz_matrix wraps
  // d_xyz_matrix_data
  CUDSS_CHECK(g_cuda_funcs.cudssMatrixCreateDn(
      &linsys_data->d_rhs_matrix, (int64_t)linsys_data->Kn, 1,
      (int64_t)linsys_data->Kn, linsys_data->d_rhs_matrix_data, valueType_setup,
      CUDSS_LAYOUT_COL_MAJOR));
  CUDSS_CHECK(g_cuda_funcs.cudssMatrixCreateDn(
      &linsys_data->d_xyz_matrix, (int64_t)linsys_data->Kn, 1,
      (int64_t)linsys_data->Kn, linsys_data->d_xyz_matrix_data, valueType_setup,
      CUDSS_LAYOUT_COL_MAJOR));

  int* trajectory_permutation = nullptr;
  CUDA_CHECK(cudaMalloc(&trajectory_permutation, linsys_data->Kn * sizeof(int)));
  int* trajectory_tree = nullptr;
  int trajectory_levels = 0;
  const int trajectory_ordered = qoco_gpu_trajectory_ordering_with_tree(linsys_data->Kn, data->n,
      csr_row_ptr, csr_col_ind, trajectory_permutation,
      &trajectory_tree, &trajectory_levels);
  if (trajectory_ordered < 0) {
    fprintf(stderr, "Invalid trajectory ordering metadata\n"); exit(1);
  }
  if (trajectory_ordered > 0) {
    auto set_permutation = reinterpret_cast<decltype(&::cudssDataSet)>(
        dlsym(g_cudss_handle, "cudssDataSet"));
    if (!set_permutation) { fprintf(stderr, "cuDSS user permutation API unavailable\n"); exit(1); }
    CUDSS_CHECK(set_permutation(linsys_data->handle, linsys_data->data,
        CUDSS_DATA_USER_PERM, trajectory_permutation, linsys_data->Kn * sizeof(int)));
    CUDSS_CHECK(g_cuda_funcs.cudssConfigSet(linsys_data->config,
        CUDSS_CONFIG_ND_NLEVELS, &trajectory_levels, sizeof(trajectory_levels)));
    CUDSS_CHECK(set_permutation(linsys_data->handle, linsys_data->data,
#if CUDSS_VERSION >= 800
        CUDSS_DATA_USER_ND_PARTITION_TREE,
#else
        CUDSS_DATA_USER_ELIMINATION_TREE,
#endif
        trajectory_tree,
        ((1 << trajectory_levels) - 1) * sizeof(int)));
  }

  // Run analysis phase.
  QOCOTimer analysis_timer;
  start_timer(&analysis_timer);
  CUDSS_CHECK(qoco_ir_execute(linsys_data->ir, 
      linsys_data->handle, CUDSS_PHASE_ANALYSIS, linsys_data->config,
      linsys_data->data, linsys_data->K_csr, linsys_data->d_xyz_matrix,
      linsys_data->d_rhs_matrix));
  // cudssExecute is asynchronous; synchronize before stopping the timer so the
  // measured time reflects the completed analysis phase.
  cudaDeviceSynchronize();
  stop_timer(&analysis_timer);
  if (analysis_time_sec) {
    *analysis_time_sec = get_elapsed_time_sec(&analysis_timer);
  }

  CUDA_CHECK(cudaFree(trajectory_permutation));
  CUDA_CHECK(cudaFree(trajectory_tree));

  return linsys_data;
}

// CUDA kernel to directly update CSR values for NT blocks
__global__ void
update_csr_nt_blocks_kernel(const QOCOFloat* WtW, // NT block values (on GPU)
                            QOCOFloat* csr_val, // CSR values to update (on GPU)
                            const QOCOInt* nt2kktcsr,
                            QOCOInt Wnnz)
{
  QOCOInt idx = blockIdx.x * blockDim.x + threadIdx.x;

  // Update NT block values
  if (idx < Wnnz) {
    QOCOInt csr_idx = nt2kktcsr[idx];
    csr_val[csr_idx] = -WtW[idx];
  }
}

// CUDA kernel to update NT diagonal regularization
__global__ void
update_csr_nt_diag_kernel(QOCOFloat* csr_val, // CSR values to update (on GPU)
                          const QOCOInt* ntdiag2kktcsr,
                          QOCOFloat kkt_static_reg_G, QOCOInt m)
{
  QOCOInt idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < m) {
    QOCOInt csr_idx = ntdiag2kktcsr[idx];
    csr_val[csr_idx] -= kkt_static_reg_G;
  }
}

__global__ void set_nt_zero_kernel(double* Kx, const int* nt2kkt, int Wnnz,
                                   int m)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid < Wnnz) {
    Kx[nt2kkt[tid]] = 0.0;
  }
}

__global__ void set_nt_identity_kernel(double* Kx, const int* nt2kkt,
                                       const int* ntdiag2kkt, int Wnnz, int m)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < m) {
    Kx[ntdiag2kkt[tid]] = -1.0;
  }
}

// CUDA kernel to update CSR values for P, A, G matrices
__global__ void update_csr_matrix_data_kernel(
    const QOCOFloat* matrix_val, // New matrix values (on GPU)
    QOCOFloat* csr_val,          // CSR values to update (on GPU)
    const QOCOInt* mat2kktcsr, QOCOInt nnz)
{
  QOCOInt idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < nnz) {
    QOCOInt csr_idx = mat2kktcsr[idx];
    csr_val[csr_idx] = matrix_val[idx];
  }
}

extern "C" int qoco_gpu_ipm_capturing();
static void qoco_ipm_factor(LinSysData*);
static void qoco_ipm_linear(LinSysData*, QOCOWorkspace*, const double*, double*, double, int);
static void cudss_factor(LinSysData* linsys_data, QOCOInt n,
                         QOCOFloat kkt_dynamic_reg)
{
  (void)n;
  (void)kkt_dynamic_reg;
  if (qoco_gpu_ipm_capturing()) { qoco_ipm_factor(linsys_data); return; }

  CUDSS_CHECK(qoco_ir_execute(linsys_data->ir, 
      linsys_data->handle, CUDSS_PHASE_FACTORIZATION, linsys_data->config,
      linsys_data->data, linsys_data->K_csr, linsys_data->d_xyz_matrix,
      linsys_data->d_rhs_matrix));
}

static void cudss_solve_system(LinSysData* linsys_data, const QOCOFloat* rhs,
                               QOCOFloat* sol)
{
  CUDA_CHECK(cudaMemcpy(linsys_data->d_rhs_matrix_data, rhs,
                        linsys_data->Kn * sizeof(QOCOFloat),
                        cudaMemcpyDeviceToDevice));

  CUDA_CHECK(cudaMemset(linsys_data->d_xyz_matrix_data, 0,
                        linsys_data->Kn * sizeof(QOCOFloat)));

  CUDSS_CHECK(qoco_ir_execute(linsys_data->ir, 
      linsys_data->handle, CUDSS_PHASE_SOLVE, linsys_data->config,
      linsys_data->data, linsys_data->K_csr, linsys_data->d_xyz_matrix,
      linsys_data->d_rhs_matrix));

  if (sol != linsys_data->d_xyz_matrix_data) {
    CUDA_CHECK(cudaMemcpy(sol, linsys_data->d_xyz_matrix_data,
                          linsys_data->Kn * sizeof(QOCOFloat),
                          cudaMemcpyDeviceToDevice));
  }
}

/**
 * @brief Computes norm(K_true*x - b, inf) on device.
 *
 * The factorization uses the statically regularized KKT matrix. The residual
 * used for iterative refinement is computed against the unregularized KKT
 * product, matching the builtin backend behavior. The residual b - K_true*x is
 * written to residual_scratch.
 */
static QOCOFloat compute_linsys_residual(LinSysData* linsys_data,
                                         QOCOWorkspace* work,
                                         const QOCOFloat* b,
                                         const QOCOFloat* x,
                                         QOCOFloat* residual_scratch, bool download = true)
{
  QOCOFloat* nt_scaling = get_data_vectorf(work->nt_scaling);
  QOCOInt* nt_scaling_soc_idx = get_data_vectori(work->nt_scaling_soc_idx);
  QOCOInt* soc_idx = get_data_vectori(work->soc_idx);
  QOCOFloat* xbuff = get_data_vectorf(work->xbuff);
  QOCOFloat* ubuff1 = get_data_vectorf(work->ubuff1);
  QOCOFloat* ubuff2 = get_data_vectorf(work->ubuff2);
  QOCOInt n = work->data->n;
  QOCOInt N = linsys_data->Kn;

  // d_rhs_matrix_data is scratch here; cudss_solve_system overwrites it before
  // every cuDSS solve.
  kkt_multiply((QOCOFloat*)x, linsys_data->d_rhs_matrix_data, work->data,
               nt_scaling, nt_scaling_soc_idx, soc_idx, xbuff, ubuff1, ubuff2);

  // data->P stores P + eps_P * I, so remove the P regularization from the
  // product before measuring the true KKT residual.
  qoco_axpy(x, linsys_data->d_rhs_matrix_data, linsys_data->d_rhs_matrix_data,
            -linsys_data->kkt_static_reg_P, n);

  // residual_scratch = b - K_true*x.
  qoco_axpy(linsys_data->d_rhs_matrix_data, b, residual_scratch, -1.0, N);

  return download ? inf_norm(residual_scratch, N) : 0.0;
}

#ifdef QOCO_LOGGING
static void log_linsys_error(LinSysData* linsys_data, QOCOWorkspace* work,
                             const QOCOFloat* b, const QOCOFloat* x,
                             QOCOFloat* residual_scratch, const char* label,
                             FILE* f)
{
  QOCOFloat res =
      compute_linsys_residual(linsys_data, work, b, x, residual_scratch);
  fprintf(f, "  (%s): %.4e\n", label, res);
}
#endif

#include "qoco_device_ir.cuh"
// Isolated diagnostic only: synchronous snapshots, never used for performance.
#include <unistd.h>
template<class T> static void dump_device(FILE* f, const T* data, size_t n) {
    std::vector<T> host(n);
    if (n) CUDA_CHECK(cudaMemcpy(host.data(), data, n*sizeof(T), cudaMemcpyDeviceToHost));
    if (fwrite(host.data(),sizeof(T),n,f)!=n) { perror("snapshot write"); exit(1); }
}
static void dump_linear(LinSysData* s,QOCOWorkspace* w,const double* b,const double* x,int index,int stage) {
    const char* root=getenv("SPACEPDHCG_DIAGNOSTIC_LINEAR_DIR"); if(!root) return;
    CUDA_CHECK(cudaDeviceSynchronize());
    compute_linsys_residual(s,w,b,x,s->d_xyz_matrix_data,false);
    CUDA_CHECK(cudaDeviceSynchronize());
    int nnz=0; CUDA_CHECK(cudaMemcpy(&nnz,s->d_csr_rows+s->Kn,sizeof(int),cudaMemcpyDeviceToHost));
    char path[4096]; snprintf(path,sizeof(path),"%s/linear-%d-%04d-%d.bin",root,getpid(),index,stage);
    FILE* f=fopen(path,"wb"); if(!f){perror(path);exit(1);}
    int header[]={s->Kn,nnz,w->data->n,w->data->p,w->data->m,w->data->l,w->data->nsoc,w->nt_scaling_nnz};
    fwrite(header,sizeof(int),8,f);
    double reg[]={s->kkt_static_reg_P,s->kkt_static_reg_A,s->kkt_static_reg_G}; fwrite(reg,sizeof(double),3,f);
    dump_device(f,s->d_csr_rows,s->Kn+1);dump_device(f,s->d_csr_columns,nnz);dump_device(f,s->d_csr_val,nnz);
    dump_device(f,b,s->Kn);dump_device(f,x,s->Kn);dump_device(f,s->d_xyz_matrix_data,s->Kn);
    dump_device(f,get_data_vectorf(w->nt_scaling),w->nt_scaling_nnz);
    dump_device(f,get_data_vectori(w->nt_scaling_soc_idx),w->data->nsoc);
    dump_device(f,get_data_vectori(w->soc_idx),w->data->nsoc);
    dump_device(f,get_data_vectori(w->data->q),w->data->nsoc);
    fclose(f);
}


static void cudss_solve(LinSysData* linsys_data, QOCOWorkspace* work,
                        QOCOVectorf* b_vec, QOCOVectorf* x_vec,
                        QOCOFloat ir_tol, QOCOInt max_ir_iters)
{
  QOCOFloat* x = get_data_vectorf(x_vec);
  QOCOFloat* b = get_data_vectorf(b_vec);
  QOCOFloat* residual = linsys_data->d_xyz_matrix_data;

  if (qoco_gpu_ipm_capturing()) {
    qoco_ipm_linear(linsys_data, work, b, x, ir_tol, max_ir_iters); return;
  }
  // Initial solve. Store the current solution in x; d_xyz_matrix_data remains
  // available as residual/correction scratch after this copy.
  cudss_solve_system(linsys_data, b, x);
  static int diagnostic_index=0;
  const int index=diagnostic_index++;
  dump_linear(linsys_data,work,b,x,index,0);
  if (!getenv("SPACEPDHCG_TEST_QOCO_DEVICE_IR_DISABLE")) {
    qoco_ir_solve(linsys_data, work, b, x, ir_tol, max_ir_iters);
    dump_linear(linsys_data,work,b,x,index,1);
    return;
  }


#ifdef QOCO_LOGGING
  FILE* log_f = fopen("qoco_log.txt", "a");
  if (log_f) {
    log_linsys_error(linsys_data, work, b, x, residual, "initial solve",
                     log_f);
  }
#endif

  QOCOFloat* best_sol = get_data_vectorf(work->xyzbuff1);
  QOCOFloat best_res = compute_linsys_residual(linsys_data, work, b, x,
                                               residual);
  copy_arrayf(x, best_sol, linsys_data->Kn);

  QOCOInt ir_count = 0;
  QOCOFloat res = best_res;

  for (QOCOInt i = 0; i < max_ir_iters; ++i) {
    if (res < ir_tol) {
      break;
    }

    // residual currently holds b - K_true*x. Solve K_reg*dx = residual.
    cudss_solve_system(linsys_data, residual, linsys_data->d_xyz_matrix_data);

    // x_new = x_old + dx.
    qoco_axpy(linsys_data->d_xyz_matrix_data, x, x, 1.0, linsys_data->Kn);

    QOCOFloat new_res = compute_linsys_residual(linsys_data, work, b, x,
                                                residual);

#ifdef QOCO_LOGGING
    if (log_f) {
      fprintf(log_f, "  (refinement): %.4e\n", new_res);
    }
#endif

    if (new_res >= best_res) {
      copy_arrayf(best_sol, x, linsys_data->Kn);
      break;
    }

    ir_count++;
    best_res = new_res;
    copy_arrayf(x, best_sol, linsys_data->Kn);
    res = new_res;
  }

  work->ir_iters += ir_count;

#ifdef QOCO_LOGGING
  if (log_f)
    fclose(log_f);
#endif
}

void cudss_set_nt_identity(LinSysData* linsys_data, QOCOInt m)
{
  int Wnnz = linsys_data->Wnnz;

  int N = max(Wnnz, m);
  int blockSize = 256;
  int gridSize = (N + blockSize - 1) / blockSize;

  if (m > 0) {
    set_nt_zero_kernel<<<gridSize, blockSize>>>(
        linsys_data->d_csr_val, linsys_data->d_nt2kktcsr, Wnnz, m);

    if (!qoco_gpu_ipm_capturing()) CUDA_CHECK(cudaDeviceSynchronize());

    set_nt_identity_kernel<<<gridSize, blockSize>>>(
        linsys_data->d_csr_val, linsys_data->d_nt2kktcsr,
        linsys_data->d_ntdiag2kktcsr, Wnnz, m);
    CUDA_CHECK(cudaGetLastError());

    update_csr_nt_diag_kernel<<<gridSize, blockSize>>>(
        linsys_data->d_csr_val, linsys_data->d_ntdiag2kktcsr,
        linsys_data->kkt_static_reg_G, m);
    CUDA_CHECK(cudaGetLastError());
    if (!qoco_gpu_ipm_capturing()) CUDSS_CHECK(g_cuda_funcs.cudssMatrixSetValues(linsys_data->K_csr,
                                                  linsys_data->d_csr_val));
    if (!qoco_gpu_ipm_capturing()) CUDA_CHECK(cudaDeviceSynchronize());
  }
}

static void cudss_update_nt(LinSysData* linsys_data, QOCOVectorf* WtW_vec,
                            QOCOFloat kkt_static_reg_G, QOCOInt m)
{
  QOCOFloat* WtW = get_data_vectorf(WtW_vec);
  // Update CSR matrix values on GPU directly for NT blocks
  if (linsys_data->Wnnz > 0 && linsys_data->d_nt2kktcsr) {
    // Copy WtW to device from host
    CUDA_CHECK(cudaMemcpyAsync(linsys_data->d_WtW, WtW,
                          linsys_data->Wnnz * sizeof(QOCOFloat),
                          cudaMemcpyDeviceToDevice));

    // Update NT blocks directly in CSR
    QOCOInt threadsPerBlock = 256;
    QOCOInt numBlocks_nt =
        (linsys_data->Wnnz + threadsPerBlock - 1) / threadsPerBlock;
    update_csr_nt_blocks_kernel<<<numBlocks_nt, threadsPerBlock>>>(
        linsys_data->d_WtW, linsys_data->d_csr_val, linsys_data->d_nt2kktcsr,
        linsys_data->Wnnz);
    CUDA_CHECK(cudaGetLastError());
  }

  // Update diagonal regularization separately
  if (m > 0 && linsys_data->d_ntdiag2kktcsr) {
    QOCOInt threadsPerBlock = 256;
    QOCOInt numBlocks_diag = (m + threadsPerBlock - 1) / threadsPerBlock;
    update_csr_nt_diag_kernel<<<numBlocks_diag, threadsPerBlock>>>(
        linsys_data->d_csr_val, linsys_data->d_ntdiag2kktcsr, kkt_static_reg_G,
        m);
    CUDA_CHECK(cudaGetLastError());
  }
  CUDSS_CHECK(g_cuda_funcs.cudssMatrixSetValues(linsys_data->K_csr,
                                                linsys_data->d_csr_val));
  if (!qoco_gpu_ipm_capturing()) CUDA_CHECK(cudaDeviceSynchronize());
}

static void cudss_update_data(LinSysData* linsys_data, QOCOProblemData* data)
{
  // Update P, A, G directly in CSR matrix on GPU
  QOCOInt threadsPerBlock = 256;

  // Update P in CSR matrix
  if (data->P && linsys_data->d_PregtoKKTcsr) {
    QOCOCscMatrix* Pcsc = get_csc_matrix(data->P);
    QOCOInt Pnnz = get_nnz(data->P);

    // Update CSR KKT matrix
    QOCOInt numBlocks = (Pnnz + threadsPerBlock - 1) / threadsPerBlock;
    update_csr_matrix_data_kernel<<<numBlocks, threadsPerBlock>>>(
        Pcsc->x, linsys_data->d_csr_val, linsys_data->d_PregtoKKTcsr, Pnnz);
    CUDA_CHECK(cudaGetLastError());
  }

  // Update A in CSR matrix
  if (data->p > 0) {
    QOCOCscMatrix* Acsc = get_csc_matrix(data->A);
    QOCOInt Annz = get_nnz(data->A);

    // Update CSR KKT matrix
    QOCOInt numBlocksA = (Annz + threadsPerBlock - 1) / threadsPerBlock;
    update_csr_matrix_data_kernel<<<numBlocksA, threadsPerBlock>>>(
        Acsc->x, linsys_data->d_csr_val, linsys_data->d_AttoKKTcsr, Annz);
    CUDA_CHECK(cudaGetLastError());
  }

  // Update G in CSR matrix
  if (data->m > 0) {

    QOCOCscMatrix* Gcsc = get_csc_matrix(data->G);
    QOCOInt Gnnz = get_nnz(data->G);

    // Update CSR KKT matrix
    QOCOInt numBlocksG = (Gnnz + threadsPerBlock - 1) / threadsPerBlock;
    update_csr_matrix_data_kernel<<<numBlocksG, threadsPerBlock>>>(
        Gcsc->x, linsys_data->d_csr_val, linsys_data->d_GttoKKTcsr, Gnnz);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
  }
}


extern "C" int qoco_gpu_update_kkt_values(QOCOSolver* solver, cudaStream_t stream) {
    auto* s = solver->linsys_data; auto* data = solver->work->data;
    QOCOMatrix* matrices[] = {data->P, data->A, data->G};
    const QOCOInt* maps[] = {s->d_PregtoKKTcsr, s->d_AttoKKTcsr, s->d_GttoKKTcsr};
    for (int i=0;i<3;++i) if (matrices[i] && maps[i]) {
        const auto* matrix = matrices[i]->d_csc_host;
        if (matrix && matrix->nnz) update_csr_matrix_data_kernel<<<(matrix->nnz+255)/256,256,0,stream>>>(
            matrix->x,s->d_csr_val,maps[i],matrix->nnz);
    }
    return cudaGetLastError()==cudaSuccess ? 0 : 4;
}

#include "qoco_ipm_graph.cuh"

static void cudss_cleanup(LinSysData* linsys_data)
{
  qoco_ipm_cache_destroy(linsys_data->ir->ipm);
  if (g_libs_loaded) {
    g_cuda_funcs.cudssMatrixDestroy(linsys_data->K_csr);
    g_cuda_funcs.cudssMatrixDestroy(linsys_data->d_rhs_matrix);
    g_cuda_funcs.cudssMatrixDestroy(linsys_data->d_xyz_matrix);
    g_cuda_funcs.cudssDataDestroy(linsys_data->handle, linsys_data->data);
    g_cuda_funcs.cudssConfigDestroy(linsys_data->config);
    g_cuda_funcs.cudssDestroy(linsys_data->handle);
    qoco_ir_destroy(linsys_data->ir);
    g_cuda_funcs.cusparseDestroy(linsys_data->cusparse_handle);
    g_cuda_funcs.cusparseDestroyMatDescr(linsys_data->descr);
  }
  cudaFree(linsys_data->d_rhs_matrix_data);
  cudaFree(linsys_data->d_xyz_matrix_data);
  qoco_free(linsys_data->nt2kkt);
  qoco_free(linsys_data->ntdiag2kkt);
  qoco_free(linsys_data->PregtoKKT);
  qoco_free(linsys_data->AttoKKT);
  qoco_free(linsys_data->GttoKKT);
  cudaFree(linsys_data->d_csr_val);
  cudaFree(linsys_data->d_csr_rows);
  cudaFree(linsys_data->d_csr_columns);
  cudaFree(linsys_data->d_nt2kktcsr);
  cudaFree(linsys_data->d_ntdiag2kktcsr);
  cudaFree(linsys_data->d_WtW);
  cudaFree(linsys_data->d_PregtoKKTcsr);
  cudaFree(linsys_data->d_AttoKKTcsr);
  cudaFree(linsys_data->d_GttoKKTcsr);
  qoco_free(linsys_data);
}

static const char* cudss_name() { return "cuda/cuDSS"; }

LinSysBackend backend = {.linsys_name = cudss_name,
                         .linsys_setup = cudss_setup,
                         .linsys_set_nt_identity = cudss_set_nt_identity,
                         .linsys_update_nt = cudss_update_nt,
                         .linsys_update_data = cudss_update_data,
                         .linsys_factor = cudss_factor,
                         .linsys_solve = cudss_solve,
                         .linsys_cleanup = cudss_cleanup};
