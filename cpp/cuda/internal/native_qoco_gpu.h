#pragma once

#include <cuda_runtime.h>
#include <cstdint>

struct QocoAuditCsc {
    int rows, columns, nonzeros;
    const int* offsets;
    const int* indices;
    const double* values;
};

struct QocoAuditInput {
    QocoAuditCsc quadratic, equality, conic, dual_map;
    const double* objective;
    const double* equality_rhs;
    const double* conic_rhs;
    int nonnegative, soc_count;
    const int* soc_sizes;
};

struct QocoAuditResult {
    double primal, dual, absolute_primal, absolute_dual, dual_cone, complementarity;
};

struct QocoAuditTransfers {
    std::uint64_t h2d_count{}, h2d_bytes{}, d2h_count{}, d2h_bytes{};
};

// Audit-owned allocations only, excluding opaque QOCO/cuDSS workspaces.
struct QocoAuditMemory {
    std::uint64_t allocations{}, bytes{}, peak_bytes{};
};

struct QocoGpuAudit;

cudaError_t qoco_gpu_audit_create(const QocoAuditInput&, bool host_solution,
                                cudaStream_t, QocoGpuAudit**);
cudaError_t qoco_gpu_audit_update(QocoGpuAudit*, const QocoAuditInput&, cudaStream_t);
cudaError_t qoco_gpu_audit_upload_solution(QocoGpuAudit*, const double*, const double*,
                                         const double*, cudaStream_t,
                                         const double**, const double**, const double**);
// Inputs are unscaled device vectors. The mapped dual is written on the same
// stream. Only the six audit scalars are downloaded; no CPU numerical work.
cudaError_t qoco_gpu_audit_run(QocoGpuAudit*, const double*, const double*, const double*,
                             double* mapped_dual, cudaStream_t, QocoAuditResult*);
QocoAuditTransfers qoco_gpu_audit_transfers(const QocoGpuAudit*);
QocoAuditMemory qoco_gpu_audit_memory(const QocoGpuAudit*);
void qoco_gpu_audit_destroy(QocoGpuAudit*);
